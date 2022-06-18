"""Contains Code for classic bandits test bed"""

import numpy as np
from . import utils as u
import matplotlib.pyplot as plt
import seaborn as sns
import dask
import dask.bag as db

dask.config.set(scheduler="processes")


class ClassicBanditsTestBed:
    """
    This class is made for testing out the classical bandits test bed.
    Reward distributions have mean between (-10,+10) and variance = 1.
    """

    def __init__(
        self,
        arms=10,
        epsilon=0.1,
        samples=1000,
        iterations=10,
        reward_size=10,
        partitions=128,
    ):
        """
        Args:
            arms (int, optional): How many arms to consider for teastbed. Defaults to 10. \n

            epsilon (float/list, optional): This the exploration factor,\n
                                            how often do the we want to explore alternate solutions.\n
                                            Can be a list or a float. Defaults to 0.1. \n

            samples (int, optional): How many samples to run each bandits for? Defaults to 1000.\n

            iterations (int, optional): how many bandits to train? Defaults to 10. \n

            reward_size (int, optional): How big should be the reward distribution? Defaults to 10.\n

            partitions (int, optional): Partitions to create for multiprocessing,\n
                                        greatly reduces compute time .Defaults to 128.\n
        """
        self.arms = arms
        self.epsilon = epsilon
        self.sample = samples
        self.iterations = iterations
        self.reward_size = reward_size
        self.arms_dist = u.generate_arm_distributions(self.reward_size, self.arms)
        self.rewards_per_iteration = []
        self.overall_avg_reward = None
        self.is_data_avail = False
        self.epsilon_reward_histories = None
        self.workers = partitions

    def run_testbed(self):
        """Runs the test bed with initiated configuration"""
        if isinstance(self.epsilon, list):
            bag = db.from_sequence([i for i in self.epsilon], npartitions=self.workers)
            self.epsilon_reward_histories = bag.map(self.run_epsilon).compute()
        else:
            self.epsilon_reward_histories = self.run_epsilon(self.epsilon)
        self.is_data_avail = True

    def run_epsilon(self, epsilon):
        """Run bandits with specific epsilon provided

        Args:
            epsilon (float): exploration factor

        Returns:
            average_reward_history: _description_
        """
        bag = db.from_sequence(
            [epsilon for i in range(self.iterations)], npartitions=self.workers
        )
        reward_histories = bag.map(self.run_iteration).compute()
        self.rewards_per_iteration = reward_histories
        epsilon_avg_reward = np.mean(np.array(self.rewards_per_iteration), axis=0)
        return epsilon_avg_reward

    def run_iteration(self, epsilon):
        """Run an individual bandit with specific epsilon

        Args:
            epsilon (float): exploration factor

        Returns:
            reward_history: reward history of current iteration
        """
        average_reward = 0
        reward_history = np.zeros(self.sample)
        current_reward_per_arm = np.zeros(shape=self.arms)
        current_selection_per_arm = np.zeros(shape=self.arms)
        for step in range(1, self.sample + 1):
            if u.greedy(epsilon):
                greedy_arm = u.select_greedy_arm(current_reward_per_arm)
                reward = np.random.choice(self.arms_dist[greedy_arm])
                reward_history[step - 1] = reward
                average_reward = u.update_avg_reward(reward, average_reward, step + 1)
                current_reward_per_arm[greedy_arm] = u.update_avg_reward(
                    reward,
                    current_reward_per_arm[greedy_arm],
                    current_selection_per_arm[greedy_arm],
                )
                current_selection_per_arm[greedy_arm] += 1

            else:
                arm = u.select_non_greedy_arm(current_reward_per_arm.copy())
                reward = np.random.choice(self.arms_dist[arm])
                reward_history[step - 1] = reward

                average_reward = u.update_avg_reward(reward, average_reward, step + 1)
                current_reward_per_arm[arm] = u.update_avg_reward(
                    reward,
                    current_reward_per_arm[arm],
                    current_selection_per_arm[arm],
                )
                current_selection_per_arm[arm] += 1
        return reward_history

    def plot_arms(self):
        """Plots the reward distribution of the arms"""
        ones = np.ones(self.arms_dist.shape)
        indices = np.array([j * i for i, j in enumerate(ones)], dtype="int")
        indices = indices.flatten(order="C")
        rewards = self.arms_dist.flatten(order="C")
        plt.figure(figsize=(14, 6), dpi=120)
        sns.violinplot(x=indices, y=rewards)
        plt.xlabel("Arm ID")
        plt.ylabel("Reward Distribution")
        plt.title("Reward Distribution of Arms")
        plt.show()

    def plot_avg_rewards(self):
        """Plots the average reward history of bandits with each epsilon"""
        if self.is_data_avail is False:
            self.run_testbed()
        xlab = np.array(list(range(1, self.sample + 1)))
        plt.figure(figsize=(14, 6), dpi=120)
        for j, i in enumerate(self.epsilon):
            plt.plot(
                xlab, self.epsilon_reward_histories[j], label="epsilon = {}".format(i)
            )
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.title("Average Reward of {} Testbed".format(self.iterations))
        plt.legend()
        plt.show()
