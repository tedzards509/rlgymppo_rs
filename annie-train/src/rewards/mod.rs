#![allow(unused_imports)]
use player_rewards::{
    AnnieDefensivePositioningReward, AnnieOffensivePositioningReward, AnnieVelocityReward,
};
use rlgymppo_utils::combined_rewards;
use rlgymppo_utils::rewards::{
    AirReward, BallTouchReward, BumpReward, BumpedPenalty, DemoReward, DemoedPenalty,
    FaceBallReward, GoalReward, PickupBoostReward, SaveBoostReward, StrongTouchReward,
    TouchAccelReward, VelocityBallToGoalReward, VelocityReward, VelocityToBallReward,
    WavedashReward,
};
pub use rlgymppo_utils::rewards::{CombinedRewards, ZeroSumReward};

use crate::rewards::player_rewards::{AnnieDistantTeammateReward, AnnieNearbyTeammateReward};

mod player_rewards;

pub struct RewardPresets;

#[allow(dead_code)]
impl RewardPresets {
    /// Rewards to learn driving towards the ball and touching it
    /// See https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/making_a_good_bot.md#what-rewards-should-i-use-in-the-early-stages
    /// Usually combined with a high learning rate for ~100M steps at my network size
    pub fn get_touch_ball_rewards<SI>() -> CombinedRewards<SI>
    where
        SI: rlgymppo_utils::shared_info::SharedInfoReport,
    {
        combined_rewards!(
             // NOTE: Should be strong touch from the start.
            "Reward/Touch ball", BallTouchReward => 50.0;
            "Reward/Speed to ball", VelocityToBallReward => 5.0;
            "Reward/Face Ball", FaceBallReward => 1.0;
            "Reward/In Air", AirReward => 0.25;
            // NOTE: Should probably be removed
            // "Reward/Player velocity", AnnieVelocityReward => 0.25;
        )
    }

    /// Rewards to learn to hit the ball toward the goal
    /// See https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/making_a_good_bot.md#learning-to-score
    /// Usually combined with a high learning rate
    pub fn get_scoring_rewards<SI>() -> CombinedRewards<SI>
    where
        SI: rlgymppo_utils::shared_info::SharedInfoReport,
    {
        combined_rewards!(
            "Reward/Goal", GoalReward::new(-0.2) => 20.0; // Barely punish getting scored on for now
            "Reward/Ball to goal", VelocityBallToGoalReward => 5.0;
            // NOTE: Think about the values more.
            "Reward/Touch ball (hard)", StrongTouchReward::new(0.0, 3600.0) => 50.0; // Disincentivize dribbling again
            "Reward/Speed to ball", VelocityToBallReward => 1.0;
            "Reward/Face Ball", FaceBallReward => 0.1;
            "Reward/In Air", AirReward => 0.25;
            "Reward/Defensive positioning", AnnieDefensivePositioningReward::default() => 0.25;
            "Reward/Offensive positioning", AnnieOffensivePositioningReward::default() => 0.25;
            // NOTE: Maybe replace with energy reward
            "Reward/Player velocity", AnnieVelocityReward => 0.25;
        )
    }

    /// Sparser rewards
    /// At this point I'm just watching the bot and trying to put band-aids on what I don't like.
    pub fn get_sparser_rewards_v0<SI>() -> CombinedRewards<SI>
    where
        SI: rlgymppo_utils::shared_info::SharedInfoReport,
    {
        combined_rewards!(
            "Reward/Goal", GoalReward::new(-0.7) => 20.0; // Punish getting scored more but aggression>passiveness
            "Reward/Ball to goal", VelocityBallToGoalReward => 2.0;
            // NOTE: Think about the values more.
            "Reward/Touch ball (hard)", StrongTouchReward::new(0.0, 3600.0) => 10.0;
            "Reward/Speed to ball", VelocityToBallReward => 0.5;
            "Reward/Face Ball", FaceBallReward => 0.1;
            "Reward/In Air", AirReward => 0.25;
            "Reward/Defensive positioning", AnnieDefensivePositioningReward::default() => 0.25;
            "Reward/Offensive positioning", AnnieOffensivePositioningReward::default() => 0.25;
            "Reward/Nearby teammates", AnnieNearbyTeammateReward::new(2000.0) => -0.5;
            "Reward/Distant teammates", AnnieDistantTeammateReward::new(7000.0, 3000.0) => -0.5;
            // NOTE: Replace with energy reward
            "Reward/Player velocity", VelocityReward => 0.25;
        )
    }
}
