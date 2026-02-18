"""Basic smoke entrypoint for the KrakenLiveEnv environment."""

from env import KrakenLiveEnv


if __name__ == "__main__":
    env = KrakenLiveEnv()
    print(f"Loaded environment: {env.__class__.__name__}")
