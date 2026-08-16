#![allow(dead_code)]
use rlgymppo::rlgym::{GameState, Reward};
use rlgymppo::rocketsim::shared::Aabb;
use rlgymppo::rocketsim::{BallState, CarInfo, CarState, GameMode, Team, consts};
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
- Pinches

Fun: Einna (Backwards only 1v1)

*/

/// Danger of the ball towards a team's goal, based on its distance to the goal center.
///
/// Returns ~1.0 at the own goal center, 0.5 at midfield, and ~0.0 at the opponent's goal,
/// smoothly mapped via a sigmoid with the given exponent.
///
/// This can be improved in the future to account for more things like:
/// - The balls velocity
/// - Nearby players
fn danger_to_goal(team: Team, ball: BallState, exponent: Option<f32>) -> f32 {
    let exponent = exponent.unwrap_or(5.0);
    let aabb: Aabb = consts::arena::get_aabb(GameMode::Soccar);
    let goal_center = consts::goal::get_goal_face_center(team);
    let dist_to_goal = (ball.pos - goal_center).length();

    // Normalized distance: +1.0 at the teams goal center, -1.0 at the opponents
    let norm_dist_to_goal = 1.0 - (dist_to_goal / aabb.max.y).min(2.0);

    1.0 / (1.0 + (-exponent * norm_dist_to_goal).exp())
}

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

/// Reward used to punish being too close to a teammate.
///
/// Returns 0 if at least at max_distance from all teammates. Returns 1 if "inside" all teammates.
#[derive(Default)]
pub struct AnnieNearbyTeammateReward {
    pub max_distance: f32,
}

impl AnnieNearbyTeammateReward {
    pub fn new(max_distance: f32) -> Self {
        AnnieNearbyTeammateReward { max_distance }
    }

    fn get_reward(&self, info: &CarInfo, car: &CarState, cars: &Vec<(CarInfo, CarState)>) -> f32 {
        if car.is_demoed {
            return 0.0;
        }

        let mut closest_proximity = f32::INFINITY;
        let mut num_teammates = 0;

        for (other_info, other_car) in cars {
            if other_info.team != info.team || other_info.idx == info.idx {
                continue;
            }

            num_teammates += 1;
            let dist = car.pos.distance(other_car.pos);
            let proximity = 1.0 - dist / self.max_distance;
            closest_proximity = closest_proximity.min(proximity);
        }

        if num_teammates == 0 {
            return 0.0;
        }

        closest_proximity.clamp(0.0, 1.0)
    }
}

impl<SI> Reward<SI> for AnnieNearbyTeammateReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(info, car)| self.get_reward(info, car, &state.cars))
            .collect()
    }
}

/// Reward used to punish being too far away from a teammate.
///
/// Returns 0 if at most min_distance from all teammates.
#[derive(Default)]
pub struct AnnieDistantTeammateReward {
    pub min_distance: f32,
    pub ramp_up: f32,
}

impl AnnieDistantTeammateReward {
    pub fn new(min_distance: f32, ramp_up: f32) -> Self {
        AnnieDistantTeammateReward {
            min_distance,
            ramp_up,
        }
    }

    fn get_reward(&self, info: &CarInfo, car: &CarState, cars: &Vec<(CarInfo, CarState)>) -> f32 {
        if car.is_demoed {
            return 0.0;
        }

        let mut furthest_distance = -1f32;
        let mut num_teammates = 0;

        for (other_info, other_car) in cars {
            if other_info.team != info.team || other_info.idx == info.idx {
                continue;
            }

            num_teammates += 1;
            let distance = (car.pos.distance(other_car.pos) - self.min_distance) / self.ramp_up;
            furthest_distance = furthest_distance.max(distance);
        }

        if num_teammates == 0 {
            return 0.0;
        }

        furthest_distance.clamp(0.0, 1.0)
    }
}

impl<SI> Reward<SI> for AnnieDistantTeammateReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(info, car)| self.get_reward(info, car, &state.cars))
            .collect()
    }
}

/// Reward for having a member of your team between the ball and goal.
///
/// Returns the largest -1 * normalized(car->ball) * normalized(car->own goal) per team for all members of the team.
/// Roughly within (-1, 1), scaled by danger_to_goal.
#[derive(Default)]
pub struct AnnieDefensivePositioningReward {
    pub exponent: Option<f32>,
}

impl AnnieDefensivePositioningReward {
    pub fn with_exponent(exponent: f32) -> Self {
        Self {
            exponent: Some(exponent),
        }
    }
}

impl<SI> Reward<SI> for AnnieDefensivePositioningReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        let goal_centers = [
            consts::goal::get_goal_face_center(Team::Blue),
            consts::goal::get_goal_face_center(Team::Orange),
        ];
        let mut team_best = [-1f32; 2];

        for (info, car) in &state.cars {
            if car.is_demoed {
                continue;
            }

            // IDEA: Scale normalized vectors and clamp dot product to get a plateau at good values instead of always trying to be right inside of goal
            let ball_dir = (state.ball.pos - car.pos).normalize_or_zero();
            let goal_dir = (goal_centers[info.team as usize] - car.pos).normalize_or_zero();
            let score = -ball_dir.dot(goal_dir);
            team_best[info.team as usize] = team_best[info.team as usize].max(score);
        }

        let danger = [
            danger_to_goal(Team::Blue, state.ball, self.exponent),
            danger_to_goal(Team::Orange, state.ball, self.exponent),
        ];

        state
            .cars
            .iter()
            .map(|(info, _)| team_best[info.team as usize] * danger[info.team as usize])
            .collect()
    }
}

#[derive(Default)]
/// Reward for having a member of your team behind the ball relative to the opponents goal.
///
/// Returns the largest -1 * normalized(ball->car) * normalized(ball->opponent goal) per team for all members of the team.
/// Roughly within (-1, 1), scaled by danger_to_goal(opponent_goal).
pub struct AnnieOffensivePositioningReward {
    pub exponent: Option<f32>,
}

impl AnnieOffensivePositioningReward {
    pub fn with_exponent(exponent: f32) -> Self {
        Self {
            exponent: Some(exponent),
        }
    }
}

impl<SI> Reward<SI> for AnnieOffensivePositioningReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        let goal_centers = [
            consts::goal::get_goal_face_center(Team::Orange),
            consts::goal::get_goal_face_center(Team::Blue),
        ];
        let mut team_best = [-1f32; 2];

        for (info, car) in &state.cars {
            if car.is_demoed {
                continue;
            }

            // IDEA: Scale normalized vectors and clamp dot product to get a plateau at good values instead of always trying to be right inside of goal
            let to_car = (car.pos - state.ball.pos).normalize_or_zero();
            let to_goal = (goal_centers[info.team as usize] - state.ball.pos).normalize_or_zero();
            let score = -to_car.dot(to_goal);
            team_best[info.team as usize] = team_best[info.team as usize].max(score);
        }

        let danger = [
            danger_to_goal(Team::Orange, state.ball, self.exponent),
            danger_to_goal(Team::Blue, state.ball, self.exponent),
        ];

        state
            .cars
            .iter()
            .map(|(info, _)| team_best[info.team as usize] * danger[info.team as usize])
            .collect()
    }
}

// TODO: Tests
