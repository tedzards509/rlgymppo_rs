#![allow(unused_imports)]
use rlgymppo_utils::combined_rewards;
use rlgymppo_utils::rewards::{
    AirReward, BallTouchReward, BumpReward, BumpedPenalty, DemoReward,
    DemoedPenalty, FaceBallReward, GoalReward, PickupBoostReward, SaveBoostReward,
    StrongTouchReward, TouchAccelReward, VelocityBallToGoalReward, VelocityReward,
    VelocityToBallReward, WavedashReward
};
pub use rlgymppo_utils::rewards::{CombinedRewards, ZeroSumReward};
use player_rewards::AnnieVelocityReward;

mod player_rewards;

pub struct RewardPresets;

#[allow(dead_code)]
impl RewardPresets {
    /// Rewards to learn driving towards the ball and touching it
    /// See https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/making_a_good_bot.md#what-rewards-should-i-use-in-the-early-stages
    /// Usually combined with a high learning rate
    pub fn get_touch_ball_rewards<SI>() -> CombinedRewards<SI>
    where
        SI: rlgymppo_utils::shared_info::SharedInfoReport,
    {
        combined_rewards!(
            "Reward/Touch ball", BallTouchReward => 50.0;
            "Reward/Speed to ball", VelocityToBallReward => 5.0;
            "Reward/Face Ball", FaceBallReward => 1.0;
            "Reward/In Air", AirReward => 0.25;
            "Reward/Player velocity", AnnieVelocityReward => 0.25;
        )
    }

    /// Rewards to learn to hit the ball toward the goal
    /// See https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/making_a_good_bot.md#learning-to-score
    /// Usually combined with a high learning rate
    pub fn get_scoring_rewards<SI>() -> CombinedRewards<SI>
    where
        SI: rlgymppo_utils::shared_info::SharedInfoReport + 'static,
    {
        combined_rewards!(
            "Reward/Learn touch ball", Self::get_touch_ball_rewards() => 1.0;
            "Reward/Goal", GoalReward::new(-0.2) => 10.0; // Barely punish getting scored on for now
            "Reward/Ball to goal", VelocityBallToGoalReward => 2.0;
            "Reward/Touch ball", BallTouchReward => 1.0;
            "Reward/Speed to ball", VelocityToBallReward => 1.0;
            "Reward/Face Ball", FaceBallReward => 0.1;
            "Reward/In Air", AirReward => 0.25;
            "Reward/Player velocity", AnnieVelocityReward => 0.25; // Update to just use length of velocity / max speed
        )
    }
}
