#![recursion_limit = "256"]

use std::path::PathBuf;
use std::thread::available_parallelism;
use burn::tensor::backend::AutodiffBackend;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng, rng};
use rlgymppo::rlgym::{Env, GameState, SharedInfoProvider};
use rlgymppo::rocketsim::{Arena, ArenaEvent, CarBodyConfig, GameMode, Team, init_from_default};
use rlgymppo::{any_terminal, default_adamw_optimizer, weighted_state, GaeEstimator, LearnerConfig, PpoLearnerConfig, SelfPlayConfig, SkillTrackerConfig};
use rlgymppo_utils::actions::DefaultAction;
use rlgymppo_utils::obs::AdvancedObs;
use rlgymppo_utils::shared_info::{SharedInfoReport, SharedInfoRng};
use rlgymppo_utils::state_setters::{KickoffState, RandomState, WeightedState};
use rlgymppo_utils::terminal::{
    AnyTerminal, NoTouchCondition, OnGoalCondition, RandomGameEndedCondition,
};
use rlgymppo_utils::{AvgTracker, Report};

mod rewards;

pub struct SharedInfo {
    rng: SmallRng,
    metrics: Report,
}

impl Default for SharedInfo {
    fn default() -> Self {
        Self {
            rng: SmallRng::seed_from_u64(rng().next_u64()),
            metrics: Report::default(),
        }
    }
}

impl SharedInfoProvider for SharedInfo {
    fn reset(&mut self, _initial_state: &GameState) {}

    fn update(&mut self, game_state: &GameState) {
        for (info, state) in &game_state.cars {
            let dist_to_ball = state.pos.distance(game_state.ball.pos);
            self.metrics["Player/Distance to ball"] += AvgTracker::from(dist_to_ball);

            self.metrics["Player/In Air Ratio"] += AvgTracker::from(!state.is_on_ground);
            self.metrics["Player/Demoed Ratio"] += AvgTracker::from(state.is_demoed);

            self.metrics["Player/Speed"] += AvgTracker::from(state.vel.length());

            let dir_to_ball = (game_state.ball.pos - state.pos).normalize_or_zero();
            let speed_towards_ball = state.vel.dot(dir_to_ball).max(0.0);
            self.metrics["Player/Speed Towards Ball"] += AvgTracker::from(speed_towards_ball);

            self.metrics["Player/Boost"] += AvgTracker::from(state.boost);

            let has_touched = game_state.events.iter().any(
                |event| matches!(event, ArenaEvent::CarHitBall(hit) if hit.car_idx == info.idx),
            );
            self.metrics["Player/Ball Touch Ratio"] += AvgTracker::from(has_touched);
        }

        for event in &game_state.events {
            if let ArenaEvent::CarHitBall(car_hit_ball) = event {
                self.metrics["Player/Touch Height"] +=
                    AvgTracker::from(car_hit_ball.contact_point.z);
            }
        }
    }
}

impl SharedInfoRng for SharedInfo {
    type Rng = SmallRng;

    fn rng(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

impl SharedInfoReport for SharedInfo {
    fn report(&mut self) -> &mut Report {
        &mut self.metrics
    }
}

const MIN_GAME_DURATION: u64 = 60 * 120;
const MAX_GAME_DURATION: u64 = 3 * 60 * 120;
type GameEndCond = RandomGameEndedCondition<MIN_GAME_DURATION, MAX_GAME_DURATION>;

const MAX_NO_TOUCH_DURATION: u64 = 30 * 120;

#[allow(clippy::type_complexity)]
pub fn create_env(
    game_id: Option<usize>,
) -> Env<
    WeightedState<SharedInfo>,
    AdvancedObs<3>,
    DefaultAction<6, 8, 1>,
    rewards::CombinedRewards<SharedInfo>,
    AnyTerminal<SharedInfo>,
    NoTouchCondition<MAX_NO_TOUCH_DURATION>,
    SharedInfo,
> {
    let game_id = game_id.unwrap_or(0);

    let n_team_players = 1 + game_id % 3;

    let mut arena = Arena::new(GameMode::Soccar);

    for _ in 0..n_team_players {
        arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
        arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
    }

    let state_setter = if n_team_players == 1 {
        // 1s state (More kickoffs)
        weighted_state![
                KickoffState, 0.6;
                RandomState<true, false, true>, 0.15;
                RandomState<true, true, true>, 0.15;
                RandomState<true, true, false>, 0.1;
            ]
    } else {
        // 2s and 3s state
        weighted_state![
                KickoffState, 0.4;
                RandomState<true, false, true>, 0.4;
                RandomState<true, true, true>, 0.3;
                RandomState<true, true, false>, 0.1;
            ]
    };


    Env::new(
        arena,
        state_setter,
        AdvancedObs,
        DefaultAction::default(),
        rewards::RewardPresets::get_scoring_rewards(),
        any_terminal![OnGoalCondition, GameEndCond],
        NoTouchCondition::default(),
        SharedInfo::default(),
    )
}

pub fn default_config<B: AutodiffBackend>(
    device: B::Device,
    render_device: B::Device,
    pipelined_collection_device: Option<B::Device>,
    skill_tracker_device: Option<B::Device>,
    async_skill_tracker: bool,
) -> LearnerConfig<B> {
    let timesteps_per_iteration = 100_000;
    let batch_size = timesteps_per_iteration;
    let mini_batch_size = 100_000;
    let gpu_timestep_buffer_size = batch_size;
    let truncation_value_batch_size = batch_size;
    let lr = 2e-4;
    let num_pools = 2;

    LearnerConfig {
        render: false,
        render_game_id: 0,
        num_pools,
        num_threads_per_pool: available_parallelism().unwrap().get() / num_pools,
        num_games_per_pool: 512 / num_pools,
        timesteps_per_save: 100_000_000,
        checkpoints_limit: None,
        checkpoints_folder: PathBuf::from("runs/checkpoints-annie-v0.3"),
        ppo: PpoLearnerConfig {
            gamma: 0.99,
            lambda: 0.95,
            timesteps_per_iteration,
            batch_size,
            mini_batch_size,
            gpu_timestep_buffer_size,
            truncation_value_batch_size,
            epochs: 2,
            learning_rate: lr,
            entropy_scale: 0.024,
            gae_estimator: GaeEstimator::TerminationTime,
            ..Default::default()
        },
        self_play: SelfPlayConfig {
            save_policy_versions: true,
            ts_per_version: 500_000_000,
            max_old_versions: 10,
            max_saved_versions: None,
            train_against_old_versions: true,
            train_against_old_chance: 0.15,
        },
        skill_tracker: SkillTrackerConfig {
            enabled: true,
            num_arenas: 12,
            update_interval: 5_000_000 / timesteps_per_iteration,
            async_eval: async_skill_tracker,
            ..Default::default()
        },
        shared_head_layer_sizes: vec![512; 2],
        policy_layer_sizes: vec![512; 2],
        critic_layer_sizes: vec![512; 3],
        device,
        render_device,
        skill_tracker_device,
        pipelined_collection_device,
        #[cfg(feature = "wandb")]
        wandb_project_name: Some("annie-v1".into()),
        #[cfg(feature = "wandb")]
        wandb_run_name: Some("annie-v1-0".into()),
        metrics_jsonl_path: Some(PathBuf::from("runs/metrics-annie-v0.3.jsonl")),
        ..Default::default()
    }
}

pub fn run<B: AutodiffBackend>(
    device: B::Device,
    render_device: B::Device,
    pipelined_collection_device: Option<B::Device>,
    skill_tracker_device: Option<B::Device>,
    async_skill_tracker: bool,
) {
    init_from_default(cfg!(not(debug_assertions))).unwrap();

    let mut learner = default_config::<B>(
        device,
        render_device,
        pipelined_collection_device,
        skill_tracker_device,
        async_skill_tracker,
    )
        .init(create_env, default_adamw_optimizer::<B>());
    learner.load();
    learner.learn();
}

#[cfg(not(any(
    feature = "torch",
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "wgpu",
    feature = "vulkan",
    feature = "flex",
    feature = "candle"
)))]
compile_error!(
    "enable exactly one backend feature to run this example, e.g. `cargo run -p rlgymppo-trainer --example run --features torch`"
);

#[cfg(any(
    all(
        feature = "torch",
        any(
            feature = "cuda",
            feature = "metal",
            feature = "rocm",
            feature = "wgpu",
            feature = "vulkan",
            feature = "flex",
            feature = "candle"
        )
    ),
    all(
        feature = "cuda",
        any(
            feature = "metal",
            feature = "rocm",
            feature = "wgpu",
            feature = "vulkan",
            feature = "flex",
            feature = "candle"
        )
    ),
    all(
        feature = "metal",
        any(
            feature = "rocm",
            feature = "wgpu",
            feature = "vulkan",
            feature = "flex",
            feature = "candle"
        )
    ),
    all(
        feature = "rocm",
        any(
            feature = "wgpu",
            feature = "vulkan",
            feature = "flex",
            feature = "candle"
        )
    ),
    all(
        feature = "wgpu",
        any(feature = "vulkan", feature = "flex", feature = "candle")
    ),
    all(feature = "vulkan", any(feature = "flex", feature = "candle")),
    all(feature = "flex", feature = "candle"),
))]
compile_error!(
    "enable only one backend feature to run this example; backend features are mutually exclusive"
);

fn main() {
    #[cfg(feature = "torch")]
    {
        use burn::backend::LibTorch;
        use burn::backend::libtorch::LibTorchDevice;
        use rlgymppo::burn::backend::Autodiff;

        run::<Autodiff<LibTorch>>(
            LibTorchDevice::Cuda(0),
            LibTorchDevice::Cpu,
            None,
            //Some(LibTorchDevice::Cuda(0)),
            Some(LibTorchDevice::Cpu),
            true,
        );
    }

    #[cfg(feature = "cuda")]
    {
        use burn::backend::Cuda;
        use burn::backend::cuda::CudaDevice;
        use rlgymppo::burn::backend::Autodiff;

        run::<Autodiff<Cuda>>(CudaDevice::new(0), CudaDevice::default(), None, None, false);
    }

    #[cfg(feature = "metal")]
    {
        use burn::backend::Metal;
        use burn::backend::wgpu::WgpuDevice;
        use rlgymppo::burn::backend::Autodiff;

        run::<Autodiff<Metal>>(
            WgpuDevice::default(),
            WgpuDevice::default(),
            None,
            None,
            false,
        );
    }

    #[cfg(feature = "rocm")]
    {
        use burn::backend::Rocm;
        use burn::backend::rocm::RocmDevice;
        use rlgymppo::burn::backend::Autodiff;

        run::<Autodiff<Rocm>>(RocmDevice::new(0), RocmDevice::default(), None, None, false);
    }

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::Wgpu;
        use burn::backend::wgpu::WgpuDevice;
        use rlgymppo::burn::backend::Autodiff;

        run::<Autodiff<Wgpu>>(
            WgpuDevice::default(),
            WgpuDevice::default(),
            None,
            Some(WgpuDevice::Cpu),
            true,
        );
    }

    #[cfg(feature = "flex")]
    {
        use burn::backend::Flex;
        use rlgymppo::burn::backend::Autodiff;

        run::<Autodiff<Flex>>(Default::default(), Default::default(), None, None, true);
    }

    #[cfg(feature = "candle")]
    {
        use burn::backend::Candle;
        use burn::backend::candle::CandleDevice;
        use rlgymppo::burn::backend::Autodiff;

        run::<Autodiff<Candle>>(
            CandleDevice::default(),
            CandleDevice::default(),
            None,
            Some(CandleDevice::Cpu),
            true,
        );
    }
}
