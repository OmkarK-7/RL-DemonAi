import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt


class UCBAgent(Agent):
    # Add fields 

    def __init__(self, time_horizon, bandit:MultiArmedBandit,): 
        # Add fields
        super().__init__(time_horizon, bandit)
        self.counts = np.zeros(self.bandit.n_arms)  # Number of times each arm was pulled
        self.values = np.zeros(self.bandit.n_arms)  # Estimated value of each arm
        self.time = 0

    def give_pull(self):
        chosen_arm = np.argmax([self.ucb_value(arm) for arm in range(self.bandit.n_arms)])
        reward = self.bandit.pull(chosen_arm)
        self.reinforce(reward, chosen_arm)
        self.time += 1

    def reinforce(self, reward, arm):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[arm] = new_value
 
    def plot_arm_graph(self):
        plt.bar(range(self.bandit.n_arms), self.values)
        plt.xlabel('Arms')
        plt.ylabel('Estimated Value')
        plt.title('Estimated Value of Each Arm')
        plt.show()


# Code to test
if __name__ == "__main__":
    # Init Bandit
    TIME_HORIZON = 10_000
    bandit = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    agent = UCBAgent(TIME_HORIZON, bandit) ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()
