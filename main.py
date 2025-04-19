import gymnasium as gym
import numpy as np
from agents.dqn_agent import DQNAgent
from utils.train_logger import TrainLogger


def train(env, agent, n_episodes=1000, max_t=1000,
          eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    logger = TrainLogger()
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

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
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    scores = train(env, agent)

    env.close()
