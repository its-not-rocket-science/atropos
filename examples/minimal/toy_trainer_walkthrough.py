from __future__ import annotations

from dataclasses import dataclass

from toy_env import ToyWordLengthEnv


@dataclass
class ToyLengthTrainer:
    """Minimal trainer loop over ToyWordLengthEnv.

    The policy is intentionally simple: among candidate actions, pick the
    shortest one to maximize normalized efficiency (reward / length budget).
    """

    max_action_length: int = 10

    def choose_action(self, candidates: list[str]) -> str:
        if not candidates:
            raise ValueError("candidates must contain at least one action")
        return min(candidates, key=len)

    def run(self) -> dict[str, float | int | list[dict[str, int | bool | str]]]:
        env = ToyWordLengthEnv(max_steps=3)
        env.reset()

        candidate_batches = [
            ["hello", "hi", "atropos"],
            ["trainer", "toy", "path"],
            ["minimal", "docs", "go"],
        ]

        trajectory: list[dict[str, int | bool | str]] = []
        for candidates in candidate_batches:
            action = self.choose_action(candidates)
            transition = env.step(action)
            trajectory.append(transition)
            if bool(transition["done"]):
                break

        total_reward = sum(int(step["reward"]) for step in trajectory)
        max_possible = self.max_action_length * len(trajectory)
        normalized_score = round(total_reward / max_possible, 3) if max_possible else 0.0

        return {
            "steps": len(trajectory),
            "total_reward": total_reward,
            "normalized_score": normalized_score,
            "trajectory": trajectory,
        }


if __name__ == "__main__":
    report = ToyLengthTrainer().run()

    print("Toy trainer walkthrough")
    print(f"steps={report['steps']}")
    print(f"total_reward={report['total_reward']}")
    print(f"normalized_score={report['normalized_score']}")
    print("trajectory:")
    for row in report["trajectory"]:
        print(f"  {row}")
