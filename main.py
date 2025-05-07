import gymnasium as gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import Namespace
from agents.dqn_agent import DQNAgent
from utils.train_logger import TrainLogger
from typing import SupportsFloat, Tuple, Any
from tqdm import tqdm


def train(env: gym.Env, agent: DQNAgent, n_episodes: int = 1000, max_t: int = 1000,
          eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.995,
          render_last: bool = False, double_flag: bool = False) -> Tuple[list[float], list[int], list[float]]:
    logger = TrainLogger(printer=tqdm.write, double_flag=double_flag)
    eps = eps_start

    discounted_returns = []
    gamma = 0.99

    for i_episode in tqdm(range(1, n_episodes + 1)):
        if render_last and i_episode == n_episodes:
            env.close()
            env = gym.make("LunarLander-v3", render_mode="human")
        
        state: np.ndarray = env.reset()[0]
        score = 0.0
        rewards = []

        for t in range(max_t):
            action = agent.act(state, eps)
            result: Tuple[np.ndarray, SupportsFloat, bool, bool, Any] = env.step(action)
            next_state, reward, terminated, truncated, _ = result
            done = terminated or truncated

            agent.step(state, action, float(reward), next_state, done)
            state = next_state
            score += float(reward)
            rewards.append(float(reward))

            if done:
                break

        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
        discounted_returns.append(G)

        success = score >= 200  # Define success threshold
        logger.log(i_episode, score, success)
        eps = max(eps_end, eps_decay * eps)

    logger.save_plots()
    logger.save_metrics()
    logger.save_summary()
    return logger.scores, logger.successes, discounted_returns

# Parses arguments
def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="TCSS 435 Lunar Lander")
    parser.add_argument("--run_one", action="store_true",
                        help="Run only double or single DQN. Defaults to both.")
    parser.add_argument("--double", action="store_true",
                        help="Applies if --run_one is used. Use double DQN instead of single DQN. Defaults to single.")
    return parser.parse_args()

# Main method
def main() -> None:
    """Main program"""
    args: Namespace = parse_args()

    def run_training(double_flag: bool):
        env: gym.Env = gym.make("LunarLander-v3")
        state_size: int = env.observation_space.shape[0] # type: ignore
        action_size: int = env.action_space.n # type: ignore
        agent: DQNAgent = DQNAgent(state_size, action_size, double_flag=double_flag)
        print(f"{'Double DQN' if double_flag else 'Single DQN'} is being used.")
        scores, successes, returns = train(env, agent, render_last=True, double_flag=double_flag)
        env.close()
        return scores, successes, returns

    if args.run_one:
        #Only runs the specified model
        run_training(double_flag=args.double)
    else:
        single_scores, single_successes, single_returns = run_training(double_flag=False) #Single DQN
        double_scores, double_successes, double_returns = run_training(double_flag=True) #Double DQN

        # Combined Reward plot
        sns.set(style="darkgrid")
        plt.figure()
        plt.plot(single_scores, label="Single DQN", color='blue')
        plt.plot(double_scores, label="Double DQN", color='red')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Single vs Double DQN Reward Over Time")
        plt.legend()
        plt.savefig("results/reward_plot_combined.png")
        plt.close()

        # Compute moving average of success
        logger_util = TrainLogger()  
        single_success_rate = logger_util._moving_average(single_successes, window=20)
        double_success_rate = logger_util._moving_average(double_successes, window=20)

        # Combined success plot
        sns.set(style="darkgrid")
        plt.figure()
        plt.plot(single_success_rate, label="Single DQN", color='blue')
        plt.plot(double_success_rate, label="Double DQN", color='red')
        plt.xlabel("Episode")
        plt.ylabel("Success Rate (Rolling Average)")
        plt.title("Single vs Double DQN: Success Rate Over Time")
        plt.legend()
        plt.savefig("results/success_plot_combined.png")
        plt.close()

        # Combined Return plot
        sns.set(style="darkgrid")
        plt.figure()
        plt.plot(single_returns, label="Single DQN", color='blue')
        plt.plot(double_returns, label="Double DQN", color='red')
        plt.xlabel("Episode")
        plt.ylabel("Episodic Return (Discounted)")
        plt.title("Single vs Double DQN: Discounted Episodic Return Over Time")
        plt.legend()
        plt.savefig("results/return_discounted_plot_combined.png")
        plt.close()


if __name__ == "__main__":
    main()
