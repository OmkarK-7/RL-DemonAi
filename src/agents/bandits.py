import numpy as np
import random
import matplotlib.pyplot as plt

class MultiArmedBandit:
    def __init__(self, arms: list):
        self.arms = arms 
        self.best_arm = np.max(arms) 
        self.cumulative_regret_array = [0]

    def pull(self, arm: int) -> int:
        assert arm in np.arange(0, len(self.arms)), "Action undefined for bandit"
        reward = 1 if np.random.random() < self.arms[arm] else 0
        self.cumulative_regret_array.append(self.cumulative_regret_array[-1] + self.best_arm - reward)
        return reward
    
    def plot_cumulative_regret(self):
        timesteps = np.arange(1, len(self.cumulative_regret_array) + 1)
        plt.figure(figsize=(8,4))
        plt.plot(timesteps, self.cumulative_regret_array, linestyle='-', color='r', label='Cumulative Regret')
        plt.title('Cumulative Regret Over Time', fontsize=16)
        plt.xlabel('Timesteps', fontsize=14)
        plt.ylabel('Cumulative Regret', fontsize=14)
        plt.grid(True, which='both', linestyle='-', linewidth=0.5)
        plt.legend(loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.show()

class Agent:
    def __init__(self, time_to_run: int, bandit: MultiArmedBandit):
        self.time_to_run = time_to_run
        self.rewards = []
        self.bandit = bandit
        self.arms = len(bandit.arms)
    
    def plot_reward_vs_time_curve(self):
        timesteps = np.arange(1, len(self.rewards) + 1)
        avg_rewards = [np.mean(self.rewards[0:T+1]) for T in range(self.time_to_run)]
        plt.figure(figsize=(8,4))
        plt.plot(timesteps, avg_rewards, linestyle='-', color='g', label='Rewards')
        plt.title('Average Reward Over Time', fontsize=16)
        plt.xlabel('Timesteps', fontsize=14)
        plt.ylabel('Mean Reward Value upto timestep t', fontsize=14)
        plt.grid(True, which='both', linestyle='-', linewidth=0.5)
        plt.legend(loc='upper left', fontsize=12)
        plt.tight_layout()
        plt.show()

class EpsilonGreedyAgent(Agent):
    def __init__(self, time_to_run: int, bandit: MultiArmedBandit, epsilon: float = 0.1):
        super().__init__(time_to_run, bandit)
        self.epsilon = epsilon
        self.estimated_rewards = np.zeros(self.arms)
        self.number_of_pulls = np.zeros(self.arms)
    
    def give_pull(self) -> int:
        best_action = np.argmax(self.estimated_rewards)
        probabilities = np.ones(self.arms) * (self.epsilon / self.arms)
        probabilities[best_action] = (1 - self.epsilon) + (self.epsilon / self.arms)
        action_pulled = np.random.choice(np.arange(0, self.arms), p=probabilities)
        reward_observed = self.bandit.pull(action_pulled)
        
        self.number_of_pulls[action_pulled] += 1
        self.estimated_rewards[action_pulled] += (reward_observed - self.estimated_rewards[action_pulled]) / self.number_of_pulls[action_pulled]
        self.rewards.append(reward_observed)
        return reward_observed

class UCBAgent(Agent):
    def __init__(self, time_to_run: int, bandit: MultiArmedBandit):
        super().__init__(time_to_run, bandit)
        self.estimated_rewards = np.zeros(self.arms)
        self.number_of_pulls = np.zeros(self.arms)
    
    def give_pull(self) -> int:
        if len(self.rewards) < self.arms:
            action_pulled = len(self.rewards)
        else:
            ucb_values = self.estimated_rewards + np.sqrt(2 * np.log(len(self.rewards)) / self.number_of_pulls)
            action_pulled = np.argmax(ucb_values)
            
        reward_observed = self.bandit.pull(action_pulled)
        self.number_of_pulls[action_pulled] += 1
        self.estimated_rewards[action_pulled] += (reward_observed - self.estimated_rewards[action_pulled]) / self.number_of_pulls[action_pulled]
        self.rewards.append(reward_observed)
        return reward_observed

class ThompsonSamplingAgent(Agent):
    def __init__(self, time_to_run: int, bandit: MultiArmedBandit):
        super().__init__(time_to_run, bandit)
        self.successes = np.ones(self.arms)
        self.failures = np.ones(self.arms)
    
    def give_pull(self) -> int:
        samples = [np.random.beta(self.successes[arm], self.failures[arm]) for arm in range(self.arms)]
        action_pulled = np.argmax(samples)
        reward_observed = self.bandit.pull(action_pulled)
        
        if reward_observed == 1:
            self.successes[action_pulled] += 1
        else:
            self.failures[action_pulled] += 1
            
        self.rewards.append(reward_observed)
        return reward_observed
