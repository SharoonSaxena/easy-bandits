"""Contains Code for Optimistic Start Bandits test bed"""

import itertools
import dask
import dask.bag as db
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from . import utils as u


dask.config.set(scheduler="processes")


class OptimisticBanditsTestBed:
    """
    This class is made for testing out the classical bandits test bed.
    Reward distributions have mean between (-10,+10) and variance = 1.
    """

    def __init__(
        self,
        arms=10,
        epsilon=0.1,
        optimism_factor=1,
        samples=1000,
        iterations=10,
        reward_size=10,
        partitions=128,
    ):
        """
        Args:
            arms (int, optional): How many arms to consider for teastbed. Defaults to 10.

            epsilon (float/list, optional): This the exploration factor,
                                            how often do the we want to explore alternate solutions.
                                            Can be a list or a float. Defaults to 0.1

            optimism_factor (float/list, optional): what should be the optimistic inital reward to promote exploration

            samples (int, optional): How many samples to run each bandits for? Defaults to 1000.

            iterations (int, optional): how many bandits to train? Defaults to 10.

            reward_size (int, optional): How big should be the reward distribution? Defaults to 10.

            partitions (int, optional): Partitions to create for multiprocessing,
                                        greatly reduces compute time .Defaults to 128.
        """
        self.arms = arms
        self.epsilon = epsilon
        self.sample = samples
        self.optimism = optimism_factor
        self.iterations = iterations
        self.reward_size = reward_size
        self.arms_dist = u.generate_arm_distributions(self.reward_size, self.arms)
        self.rewards_per_iteration = []
        self.overall_avg_reward = None
        self.is_data_avail = False
        self.epsilon_reward_histories = None
        self.optimism_epsilon_avg_reward_histories = None
        self.workers = partitions

    def run_testbed(self):
        """Runs the test bed with initiated configuration"""
        if (not isinstance(self.epsilon, list)) and (
            not isinstance(self.optimism, list)
        ):
            params = (self.optimism, self.epsilon)
            bag = db.from_sequence(params for i in self.iterations)
            self.rewards_per_iteration = bag.map(self.run_iteration).compute()
            self.optimism_epsilon_avg_reward_histories = np.mean(
                np.array(self.rewards_per_iteration), axis=0
            )
            self.optimism_epsilon_avg_reward_histories = {
                params: {
                    "values": self.optimism_epsilon_avg_reward_histories,
                    "optimism": params[0],
                    "epsilon": params[1],
                }
            }
            self.is_data_avail = True
        else:
            tmp1, tmp2 = self.optimism, self.epsilon
            if not isinstance(self.optimism, list):
                tmp1 = []
                tmp1.append(self.optimism)
            if not isinstance(self.epsilon, list):
                tmp2 = []
                tmp2.append(self.epsilon)
            self.optimism, self.epsilon = tmp1, tmp2
            self.optimism_epsilon_avg_reward_histories = {}
            cross_product = tuple(itertools.product(tmp1, tmp2))
            for combo in cross_product:
                bag = db.from_sequence(combo for i in range(self.iterations))
                combo_iteration_history = bag.map(self.run_iteration).compute()
                combo_history_average = np.mean(
                    np.array(combo_iteration_history), axis=0
                )
                self.optimism_epsilon_avg_reward_histories[combo] = {
                    "values": combo_history_average,
                    "optimism": combo[0],
                    "epsilon": combo[1],
                }
                self.is_data_avail = True

    def run_iteration(self, params):
        """Run an individual bandit with specific epsilon

        Args:
            epsilon (float): exploration factor

        Returns:
            reward_history: reward history of current iteration

            params(tuple): contains optimistic factor and epsilon values

        Returns:
            reward_history: reward history of current iteration with param combo
        """
        optimism, epsilon = params
        average_reward = 0
        reward_history = np.zeros(self.sample)
        current_reward_per_arm = np.array(
            [np.max(self.arms_dist) * optimism for i in range(self.arms)]
        )
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
        for combo in tuple(itertools.product(self.optimism, self.epsilon)):
            plt.plot(
                xlab,
                self.optimism_epsilon_avg_reward_histories[combo]["values"],
                label="optimism {}; epsilon {}".format(*combo),
            )
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.title("Average Reward of {} Testbed".format(self.iterations))
        plt.legend()
        plt.show()
