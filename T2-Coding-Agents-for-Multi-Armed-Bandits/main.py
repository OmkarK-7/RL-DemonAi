# Write code which will run all the different bandit agents together and:
# 1. Plot a common cumulative regret curves graph
# 2. Plot a common graph of average reward curves

import numpy as np
import matplotlib.pyplot as plt
from base import *
from klucb import *
from ucb import *
from thompson import *
from epsilon_greedy import *


# Assuming the classes ThompsonSamplingAgent, UCBAgent, EpsilonGreedyAgent, and MultiArmedBandit are already defined

def run_experiment(agent_class, bandit, time_horizon, label, **kwargs):
    agent = agent_class(time_horizon, bandit, **kwargs)
    rewards = np.zeros(time_horizon)
    for t in range(time_horizon):
        rewards[t] = agent.give_pull()
    cumulative_rewards = np.cumsum(rewards)
    average_rewards = cumulative_rewards / (np.arange(time_horizon) + 1)
    cumulative_regret = np.arange(time_horizon) * max(bandit.probabilities) - cumulative_rewards
    return average_rewards, cumulative_regret

if __name__ == "__main__":
    TIME_HORIZON = 10_000
    bandit_probabilities = np.array([0.23,0.55,0.76,0.44])
    bandit = MultiArmedBandit(bandit_probabilities)

    agents = [
        (ThompsonSamplingAgent, {}),
        (UCBAgent, {}),
        (EpsilonGreedyAgent, {'epsilon': 0.1})
    ]

    plt.figure(figsize=(12, 6))

    # Plot common graph of average reward curves
    plt.subplot(1, 2, 1)
    for agent_class, kwargs in agents:
        average_rewards, _ = run_experiment(agent_class, bandit, TIME_HORIZON, agent_class.__name__, **kwargs)
        plt.plot(average_rewards, label=agent_class.__name__)
    plt.title('Average Reward Curves')
    plt.xlabel('Time')
    plt.ylabel('Average Reward')
    plt.legend()

    # Plot common cumulative regret curves graph
    plt.subplot(1, 2, 2)
    for agent_class, kwargs in agents:
        _, cumulative_regret = run_experiment(agent_class, bandit, TIME_HORIZON, agent_class.__name__, **kwargs)
        plt.plot(cumulative_regret, label=agent_class.__name__)
    plt.title('Cumulative Regret Curves')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.legend()

    plt.tight_layout()
    plt.show()