"""
Authors: Charankamal Brar, Corey Young, Brittney Jones, Johan Hernandez, Lucas Perry
Date: 05/07/2025
"""

import os
from typing import List, Callable
import matplotlib.pyplot as plt
import seaborn as sns

class TrainLogger:
    """TrainLogger class for logging training progress and saving results.

    This class helps in tracking the progress of a reinforcement learning agent during 
    training by storing episode scores, success rates, and generating plots of the 
    agent's performance over time. It also saves the final scores to a file for later 
    analysis.
    """

    def __init__(self, save_dir: str = "results", printer: Callable[[str], None] = print) -> None:
        """
        Initialize the TrainLogger with optional parameters for the save directory 
        and printing function.
        
        - save_dir: Directory where the results (plots and scores) will be saved. 
                    Defaults to "results".
        - printer: A callable function for printing logs to the console. Defaults to the 
                   built-in `print` function.
        """
        
        # Lists to hold the scores and success flags for each episode
        self.scores: List[float] = []
        self.successes: List[int] = []

        # Directory to save results such as plots and metrics
        self.save_dir = save_dir

        # Callable function for printing logs, default is print
        self.print = printer
        
        # Create the save directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)

    def log(self, episode: int, score: float, success: bool) -> None:
        """
        Log the score and success of a specific episode.
        
        This method adds the score and success flag (1 for success, 0 for failure) to 
        the internal lists and prints a log message to the console.

        - episode: The episode number.
        - score: The score achieved during this episode.
        - success: A boolean indicating whether the episode was successful (True) or not (False).
        """
        
        # Add the score and success flag to the respective lists
        self.scores.append(score)
        self.successes.append(1 if success else 0)
        
        # Print the log message with episode details
        self.print(f"Episode {episode:10} | Score: {score:10.2f} | Success: {success}")

    def save_plots(self) -> None:
        """
        Create and save plots for the training progress.

        This method generates and saves two plots:
        - A plot showing the reward per episode over time.
        - A plot showing the success rate (moving average) over time.

        Both plots are saved in the specified save directory.
        """
        
        # Set the plot style using Seaborn for a clean and professional look
        sns.set(style="darkgrid")

        # Reward plot: Plot the scores over episodes
        plt.figure()
        plt.plot(self.scores, label="Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("DQN Reward Over Time")
        plt.legend()
        
        # Save the reward plot as an image file
        plt.savefig(os.path.join(self.save_dir, "reward_plot.png"))
        plt.close()

        # Success rate plot: Calculate the moving average of successes for smooth representation
        success_rate = self._moving_average(self.successes, window=20)
        
        # Plot the success rate over episodes
        plt.figure()
        plt.plot(success_rate, label="Success Rate (rolling)")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.title("LunarLander Success Rate")
        plt.legend()
        
        # Save the success rate plot as an image file
        plt.savefig(os.path.join(self.save_dir, "success_plot.png"))
        plt.close()

    def _moving_average(self, data: List[int], window: int = 20) -> List[float]:
        """
        Compute the moving average of a given list of data with a specified window size.
        
        - data: The list of numerical data to compute the moving average for.
        - window: The size of the rolling window used to calculate the moving average. Defaults to 20.
        
        Returns A list containing the moving average values.
        """
        
        # Compute the moving average using a sliding window
        return [sum(data[max(0, i - window):(i + 1)]) / min(i + 1, window) for i in range(len(data))]

    def save_metrics(self) -> None:
        """
        Save the final episode scores to a text file.
        
        This method saves the list of scores to a text file in the specified save directory.
        Each score is written to a new line in the file.

        The file is named "scores.txt".
        """
        
        # Open the file in write mode and save each score in a new line
        with open(os.path.join(self.save_dir, "scores.txt"), "w") as f:
            for score in self.scores:
                f.write(f"{score}\n")
