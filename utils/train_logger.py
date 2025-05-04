import os
from typing import List, Callable
import matplotlib.pyplot as plt
import seaborn as sns

class TrainLogger:
    def __init__(self, save_dir: str = "results", printer: Callable[[str], None] = print, double_flag: bool = False) -> None:
        self.scores: List[float] = []
        self.successes: List[int] = []
        self.save_dir = save_dir
        self.print = printer
        self.double_flag: bool = double_flag
        os.makedirs(save_dir, exist_ok=True)

    def log(self, episode: int, score: float, success: bool) -> None:
        """Log score and success for an episode."""
        self.scores.append(score)
        self.successes.append(1 if success else 0)
        self.print(f"Episode {episode:10} | Score: {score:10.2f} | Success: {success}")

    def save_plots(self) -> None:
        """Create and save reward + success plots."""
        sns.set(style="darkgrid")
        dqn_type: str = "double" if self.double_flag else "single"

        # Reward plot
        plt.figure()
        plt.plot(self.scores, label="Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"{'Double' if self.double_flag else 'Single'} DQN Reward Over Time")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f"reward_plot_{dqn_type}.png"))
        plt.close()

        # Success rate (moving average)
        success_rate = self._moving_average(self.successes, window=20)
        plt.figure()
        plt.plot(success_rate, label="Success Rate (rolling)")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.title("LunarLander Success Rate")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f"success_plot_{dqn_type}.png"))
        plt.close()

    def _moving_average(self, data: List[int], window: int = 20) -> List[float]:
        return [sum(data[max(0, i - window):(i + 1)]) / min(i + 1, window) for i in range(len(data))]

    def save_metrics(self) -> None:
        """Save final scores to file."""
        with open(os.path.join(self.save_dir, "scores.txt"), "w") as f:
            for score in self.scores:
                f.write(f"{score}\n")
