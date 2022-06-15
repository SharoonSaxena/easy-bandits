import numpy as np
import utils as u
import matplotlib.pyplot as plt
import dask
import dask.bag as db

dask.config.set(scheduler="processes")


class BanditsTestBed:
    """
    input: k_arms=10, epsilon=0.1, iter=10, reward_distribution_size=10, algo_iter=10, n_jobs=1

    Algo:
    set n_jobs
    reward_per_algo_loop_tracker
    Generate reward distributions
    algo_iter loop : #distributed#
        average reward tracker = 0
        reward_per_arm_tracker = [0s]
        loop for iterations:
            epsilon probalility
            if greedy
                sample reward from max util arm
                update arm
                update average reward
            if explore
                randomise from non-greedy arms
                update picked arm
                update average reward
        update reward per algo loop
    calculate average reward per iteration across algo loop

    plotting function for arms
    plotting function for average rewards
    """

    def __init__(
        self,
        arms=10,
        epsilon=0.1,
        samples=1000,
        iterations=10,
        reward_size=10,
        n_jobs=4,
    ):
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
        self.workers = n_jobs

    def run_testbed(self):
        if isinstance(self.epsilon, list):
            bag = db.from_sequence([i for i in self.epsilon], npartitions=self.workers)
            self.epsilon_reward_histories = bag.map(self.run_epsilon).compute()
        else:
            self.epsilon_reward_histories = self.run_epsilon(self.epsilon)
        self.is_data_avail = True

    def run_epsilon(self, epsilon):
        bag = db.from_sequence(
            [epsilon for i in range(self.iterations)], npartitions=self.workers
        )
        reward_histories = bag.map(self.run_iteration).compute()
        self.rewards_per_iteration = reward_histories
        epsilon_avg_reward = np.mean(np.array(self.rewards_per_iteration), axis=0)
        return epsilon_avg_reward

    def run_iteration(self, epsilon):
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
        u.prepare_and_plot_data(self.arms_dist)

    def plot_avg_rewards(self):
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
