"""Contain utils for classic bandits"""

import numpy as np


def generate_arm_distributions(size, arms):
    """Generates random reward distribution for the bandit arms

    Args:
        size (int): size of distributions
        arms (int): number of arms

    Returns:
        np.array(arms,size): reward_distribution of each srm
    """
    dist_array = np.array(
        [np.random.normal(np.random.randint(-5, +5), 1, size) for i in range(arms)]
    )
    return dist_array


# print(generate_arm_distributions(5, 5)[0])


def greedy(epsilon):
    """probability sampler

    Args:
        epsilon (float): probability [0,1)

    Returns:
        bool: greedy or not
    """
    return np.random.ranf() >= epsilon


# print(greedy(0.1))


def select_greedy_arm(arms):
    """Selects the best rewarding arm, breaks tie at random

    Args:
        arms (np.array): arm rewards

    Returns:
        int: index of best arm
    """
    contenders = np.where(arms == arms.max())
    if contenders is int:
        return contenders
    index = np.random.choice(a=contenders[0])
    return index


# print(select_greedy_arm(np.array([1, 5, 2, 3, 4, 5, 5, 5])))


def update_avg_reward(reward, old_reward, n_samples):
    """Updates reward vales using incremental formula

    Args:
        reward (float): new_reward
        old_reward (float): current reward
        n_samples (_type_): steps already taken

    Returns:
        float: updated reward
    """
    if n_samples == 0:
        return reward
    return old_reward + (1 / n_samples) * (reward - old_reward)


# print(update_avg_reward(5, 3, 3))


def select_non_greedy_arm(arms):
    """Selects the random rewarding arm

    Args:
        arms (np.array): arm rewards

    Returns:
        int: index of best arm
    """
    contenders = np.where(arms != arms.max())
    try:
        index = np.random.choice(a=contenders[0])
    except ValueError:
        index = np.random.choice(a=np.arange(len(arms)))
    return index


def update_confidence_bound(step, selection_array, confidence):
    """updates the confidense bound after every arm iteration

    Args:
        step (current step): current step
        selection_array (np.array): array of arms chosen
        confidence (float): confidence factor

    Returns:
        np.array: updates UCB
    """
    confidence_bound = confidence * np.log(step / selection_array)
    if step == 0:
        confidence_bound = np.inf
    return confidence_bound


def update_ucb_arm(confidence, n_samples, step):
    """updates UCB of selected arm

    Args:
        confidence (float): confidence factor
        n_samples (int): times arm is selected
        step (int): current step

    Returns:
        float: updated UCB
    """
    confidence_bound = confidence * np.log(step / n_samples)
    return confidence_bound


def select_highest_ucb(rewards, ucb):
    """selects arm with highest ucb

    Args:
        rewards (_type_): _description_
        ucb (_type_): _description_

    Returns:
        _type_: _description_
    """
    tmp = rewards + ucb
    contenders = np.where(tmp == tmp.max())
    if contenders is int:
        return contenders
    index = np.random.choice(a=contenders[0])
    return index
