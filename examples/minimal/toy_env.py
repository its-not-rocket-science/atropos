from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToyWordLengthEnv:
    """Tiny deterministic environment for first-run onboarding.

    - state: current step index
    - action: text string
    - reward: number of characters in that string
    """

    max_steps: int = 3
    step_index: int = 0

    def reset(self) -> dict[str, int | bool]:
        self.step_index = 0
        return {"step": self.step_index, "done": False}

    def step(self, action_text: str) -> dict[str, int | bool | str]:
        if self.step_index >= self.max_steps:
            return {
                "step": self.step_index,
                "action": action_text,
                "reward": 0,
                "done": True,
                "reason": "episode_complete",
            }

        self.step_index += 1
        done = self.step_index >= self.max_steps
        return {
            "step": self.step_index,
            "action": action_text,
            "reward": len(action_text),
            "done": done,
        }


def run_demo_episode() -> list[dict[str, int | bool | str]]:
    env = ToyWordLengthEnv(max_steps=3)
    _ = env.reset()

    actions = ["hi", "atropos", "toy"]
    transitions = [env.step(action) for action in actions]
    return transitions


if __name__ == "__main__":
    episode = run_demo_episode()
    total_reward = sum(int(transition["reward"]) for transition in episode)

    print("Toy environment demo")
    for transition in episode:
        print(transition)
    print(f"total_reward={total_reward}")
