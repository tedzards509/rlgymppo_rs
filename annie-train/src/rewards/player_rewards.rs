use rlgymppo::rlgym::{GameState, Reward};
use rlgymppo::rocketsim::{consts, CarState};

/*
# Brainstorm:
- Control
    Close to ball and similar velocity
- Close to teammate punishment
- No opponent can reach the ball reward

Mechanics:
- Hook shot / Cutting
- Dribbling + Cool flicks
- Aerial

*/



/// Reward for general velocity.
///
/// Returns the norm of the car's velocity, normalized by the maximum car speed (2300) so the output is roughly in [0, 1].
#[derive(Default)]
pub struct AnnieVelocityReward;

impl AnnieVelocityReward {
    fn get_reward(car: &CarState) -> f32 {
        if car.is_demoed {
            return 0.0;
        }

        car.vel.length() / consts::car::MAX_SPEED
    }
}

impl<SI> Reward<SI> for AnnieVelocityReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(_, car)| Self::get_reward(car))
            .collect()
    }
}
