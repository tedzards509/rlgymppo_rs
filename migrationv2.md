
```sh
export LIBTORCH=/home/ted/.libtorch/libtorch-rocm-6.4-2_9/
export LD_LIBRARY_PATH=/home/ted/.libtorch/libtorch-rocm-6.4-2_9/lib:$LD_LIBRARY_PATH
export LIBTORCH_BYPASS_VERSION_CHECK=1
cargo run -p annie-train --no-default-features --features torch,tui
```

Goal:
Pipelining over A/B memory one is filled with experience, one is frozen and used for training

Current training loop (sequentially):
- nodiff_model <- model (?What does mode.valid() do?)
- Collect steps
    - Decide on if self-play iteration
        - If self-play, clone random old version from self.version_mgr and initialize into self_play
        - Else None
    - memory, metrics <- self.collector.run(nodiff_model, self_play)
- Train model
    - self.ppo.learn(model, memory, rng, metrics, stats, is_first_iteration)
- Metrics report, skill tracker


Closest-to-ball reward
Energy reward

## Energy reward
Some reference formulas
```py
CAR_MASS = 180
BALL_MASS = 30
GRAVITATIONAL_ACCELERATION = 6.5
BOOST_CONSUMPTION_RATE = 33.3
BOOST_ACCELERATION = 9.91666


def potential_energy(df, is_player):
    m = CAR_MASS if is_player else BALL_MASS
    g = GRAVITATIONAL_ACCELERATION
    h = df["pos_z"] / 100

    return m * g * h


def kinetic_energy(df, is_player):
    vel_xyz = df[["vel_x", "vel_y", "vel_z"]] / 100
    v = np.linalg.norm(vel_xyz.values, axis=-1)
    m = CAR_MASS if is_player else BALL_MASS

    return 0.5 * m * v ** 2


def boost_energy(df):
    boost = df["boost_amount"]
    a = BOOST_ACCELERATION
    m = CAR_MASS
    t = boost / BOOST_CONSUMPTION_RATE
    
    return 0.5 * m * a ** 2 * t ** 2
```
Opti:
```py
# energy reward
            if self.energy_reward_w != 0:
                # max_energy is supersonic at ceiling, use to norm, ignore jump/dodge and boost
                max_energy = (MASS * GRAVITY * (CEILING_Z - 17)) + (0.5 * MASS * (CAR_MAX_SPEED * CAR_MAX_SPEED))
                energy = 0
                # add height PE
                energy += 1.1 * MASS * GRAVITY * player.car_data.position[2]
                # add KE
                velocity = np.linalg.norm(player.car_data.linear_velocity)
                energy += 0.5 * MASS * (velocity * velocity)
                # add boost
                energy += 7.97e5 * player.boost_amount * 100
                if player.has_jump:
                    energy += 0.8 * 0.5 * MASS * (292 * 292)
                if player.has_flip:
                    dodge_impulse = 500 + (velocity / 17) if velocity <= 1700 else (600 - (velocity - 1700))
                    # cheat a bit to encourage the dodge usage
                    dodge_impulse = max(dodge_impulse - 25, 0)
                    energy += 0.9 * 0.5 * MASS * (dodge_impulse * dodge_impulse)

                norm_energy = energy / max_energy
                if player.is_demoed:
                    norm_energy = 0
                player_self_rewards[i] += norm_energy * self.energy_reward_w

```

Greg:
```py
def energy_reward():
        """
        energy reward
        """
        energy_weight = 0.1
        energy_reward = 0
        
        # max_energy is supersonic at ceiling, use to norm, ignore jump/dodge and boost
        max_energy = (MASS * GRAVITY * (CEILING_Z - 17)) + (0.5 * MASS * (CAR_MAX_SPEED * CAR_MAX_SPEED))
        energy_reward = 0
        # add height PE
        energy_reward += 1.1 * MASS * GRAVITY * player_position[2]
        # add KE
        velocity = np.linalg.norm(player_velocity)
        energy_reward += 0.5 * MASS * (velocity * velocity)
        # add boost
        energy_reward += 7.97e5 * player.boost_amount * 100
        if player.has_jump:
            energy_reward += 0.8 * 0.5 * MASS * (292 * 292)
        if player.has_flip:
            dodge_impulse = 500 + (velocity / 17) if velocity <= 1700 else (600 - (velocity - 1700))
            # cheat a bit to encourage the dodge usage
            dodge_impulse = max(dodge_impulse - 25, 0)
            energy_reward += 0.9 * 0.5 * MASS * (dodge_impulse * dodge_impulse)

        # this is some demo logic that I haven't figured out lol
        norm_energy = energy_reward / max_energy
        if player.is_demoed:
            norm_energy = 0

        energy_reward = norm_energy * energy_weight
```