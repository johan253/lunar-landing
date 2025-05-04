import gymnasium as gym
import numpy as np
from agents.dqn_agent import DQNAgent
from utils.train_logger import TrainLogger
from typing import SupportsFloat, Tuple, Any
from tqdm import tqdm


def train(env: gym.Env, agent: DQNAgent, n_episodes: int = 1000, max_t: int = 1000,
          eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.995,
          render_last: bool = False) -> list[float]:
    logger = TrainLogger(printer=tqdm.write)
    eps = eps_start

    for i_episode in tqdm(range(1, n_episodes + 1)):
        if render_last and i_episode == n_episodes:
            env.close()
            env = gym.make("LunarLander-v3", render_mode="human")
        
        state: np.ndarray = env.reset()[0]
        score = 0.0

        for t in range(max_t):
            action = agent.act(state, eps)
            result: Tuple[np.ndarray, SupportsFloat, bool, bool, Any] = env.step(action)
            next_state, reward, terminated, truncated, _ = result
            done = terminated or truncated

            agent.step(state, action, float(reward), next_state, done)
            state = next_state
            score += float(reward)

            if done:
                break

        success = score >= 200  # Define success threshold
        logger.log(i_episode, score, success)
        eps = max(eps_end, eps_decay * eps)

    logger.save_plots()
    logger.save_metrics()
    return logger.scores


if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0] # type: ignore
    action_size = env.action_space.n # type: ignore

    agent = DQNAgent(state_size, action_size, use_double_dqn=True)
    scores = train(env, agent, render_last=True)

    env.close()
