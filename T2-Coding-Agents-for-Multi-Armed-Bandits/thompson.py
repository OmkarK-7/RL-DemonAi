import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt


class ThompsonSamplingAgent(Agent):
    # Add fields 

    def __init__(self, time_horizon, bandit:MultiArmedBandit,): 
        # Add fields
        super().__init__(time_horizon, bandit)
        self.successes = np.zeros(self.bandit.arms)
        self.failures = np.zeros(self.bandit.arms)

    def give_pull(self):
        samples = [beta(a=1+self.successes[i], b=1+self.failures[i]).rvs()
                   for i in range(self.bandit.arms)]
        chosen_arm = np.argmax(samples)
        reward = self.bandit.pull(chosen_arm)
        self.reinforce(reward, chosen_arm) 

    def reinforce(self, reward, arm):
        if reward == 1:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1
 
    def plot_arm_graph(self):
        arms = range(self.bandit.arms)
        plt.bar(arms, self.successes / (self.successes + self.failures + 1), label='Success rate')
        plt.xlabel('Arms')
        plt.ylabel('Success Rate')
        plt.title('Success Rate of Each Arm')
        plt.legend()
        plt.show()

# Code to test
if __name__ == "__main__":
    # Init Bandit
    TIME_HORIZON = 10_000
    bandit = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    agent = ThompsonSamplingAgent(TIME_HORIZON, bandit) ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()
