"""Basic smoke entrypoint for the KrakenLiveEnv environment."""

import numpy as np

import env as env_module
from env import KrakenLiveEnv


def main() -> None:
    env_module.time.sleep = lambda *_args, **_kwargs: None

    env = KrakenLiveEnv()
    print(f"Loaded environment: {env.__class__.__name__}")

    obs, info = env.reset()
    print(f"Reset observation shape: {obs.shape}")
    print(f"Reset info: {info}")

    assert obs.shape == env.observation_space.shape

    for step_num in range(1, 6):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        has_nan = bool(np.isnan(obs).any())
        has_inf = bool(np.isinf(obs).any())
        has_nan_or_inf = has_nan or has_inf
        action_taken = info.get("action_taken", "unknown")

        print(
            f"Step {step_num}: "
            f"obs_shape={obs.shape}, "
            f"has_nan_or_inf={has_nan_or_inf}, "
            f"reward={reward:.8f}, "
            f"action_taken={action_taken}"
        )

        assert obs.shape == env.observation_space.shape
        assert not has_nan, f"NaN detected in observation at step {step_num}"
        assert not has_inf, f"Inf detected in observation at step {step_num}"
        assert isinstance(reward, float)
        assert isinstance(action_taken, str)

        if terminated or truncated:
            print(
                f"Episode ended early at step {step_num} "
                f"(terminated={terminated}, truncated={truncated})"
            )
            break

    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
