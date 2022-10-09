"""Contains Code for classic bandits test bed"""

import numpy as np
from . import utils as u
import matplotlib.pyplot as plt
import seaborn as sns
import dask
import dask.bag as db

dask.config.set(scheduler="processes")


class UCBBanditsTestBed:
    """
    This class is made for testing out the classical bandits test bed.
    Reward distributions have mean between (-10,+10) and variance = 1.
    """

    def __init__(
        self,
        arms=10,
        confidence_factor=0.1,
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
        self.confidence = confidence_factor
        self.sample = samples
        self.iterations = iterations
        self.reward_size = reward_size
        self.arms_dist = u.generate_arm_distributions(self.reward_size, self.arms)
        self.rewards_per_iteration = []
        self.overall_avg_reward = None
        self.is_data_avail = False
        self.epsilon_reward_histories = None
        self.workers = partitions
        self.rolling_sum = None

    def run_testbed(self):
        """Runs the test bed with initiated configuration"""
        if isinstance(self.confidence, list):
            for confidence in self.confidence:
                bag = db.from_sequence(
                    [confidence for _ in range(self.iterations)],
                    npartitions=self.workers,
                )
                reward_histories = bag.map(self.run_iteration).compute()
                self.rewards_per_iteration = np.mean(np.array(reward_histories), axis=0)
        else:
            bag = db.from_sequence(
                [self.confidence for _ in range(self.iterations)],
                npartitions=self.workers,
            )
            reward_histories = bag.map(self.run_iteration).compute()
            self.rewards_per_iteration = np.mean(np.array(reward_histories), axis=0)
            self.is_data_avail = True

    def run_iteration(self, confidence):
        """Run an individual bandit with specific epsilon

        Args:
            epsilon (float): exploration factor

        Returns:
            reward_history: reward history of current iteration
        """
        reward_history = np.zeros(self.sample)
        current_reward_per_arm = np.zeros(shape=self.arms)
        current_selection_per_arm = np.zeros(shape=self.arms)
        current_confidence_bound = u.update_confidence_bound(
            0, current_selection_per_arm, confidence
        )

        for step in range(1, self.sample + 1):
            best_arm = u.select_highest_ucb(
                current_reward_per_arm, current_confidence_bound
            )
            reward = np.random.choice(self.arms_dist[best_arm])
            reward_history[step - 1] = reward
            current_selection_per_arm[best_arm] += 1
            current_reward_per_arm[best_arm] = u.update_avg_reward(
                reward,
                current_reward_per_arm[best_arm],
                current_selection_per_arm[best_arm],
            )
            current_confidence_bound = u.update_confidence_bound(
                step, current_selection_per_arm, confidence
            )
            current_confidence_bound[best_arm] = u.update_ucb_arm(
                confidence, current_selection_per_arm[best_arm], step
            )
            cumulative_reward = np.cumsum(reward_history)
        return cumulative_reward

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
        if not isinstance(self.confidence, list):
            tmp = []
            tmp.append(self.confidence)
            self.confidence = tmp
            self.rewards_per_iteration = self.rewards_per_iteration.reshape(1, -1)
        for j, i in enumerate(self.confidence):
            plt.plot(
                xlab, self.rewards_per_iteration[j], label="confidence_factor = {}".format(i)
            )
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.title("Average Reward of {} Testbed".format(self.iterations))
        plt.legend()
        plt.show()
