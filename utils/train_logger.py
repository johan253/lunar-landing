import os
import matplotlib.pyplot as plt
import seaborn as sns

class TrainLogger:
    def __init__(self, save_dir="results"):
        self.scores = []
        self.successes = []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def log(self, episode: int, score: float, success: bool):
        """Log score and success for an episode."""
        self.scores.append(score)
        self.successes.append(1 if success else 0)
        print(f"Episode {episode} | Score: {score:.2f} | Success: {success}")

    def save_plots(self):
        """Create and save reward + success plots."""
        sns.set(style="darkgrid")

        # Reward plot
        plt.figure()
        plt.plot(self.scores, label="Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("DQN Reward Over Time")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, "reward_plot.png"))
        plt.close()

        # Success rate (moving average)
        success_rate = self._moving_average(self.successes, window=20)
        plt.figure()
        plt.plot(success_rate, label="Success Rate (rolling)")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.title("LunarLander Success Rate")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, "success_plot.png"))
        plt.close()

    def _moving_average(self, data, window=20):
        return [sum(data[max(0, i - window):(i + 1)]) / min(i + 1, window) for i in range(len(data))]

    def save_metrics(self):
        """Save final scores to file."""
        with open(os.path.join(self.save_dir, "scores.txt"), "w") as f:
            for score in self.scores:
                f.write(f"{score}\n")
