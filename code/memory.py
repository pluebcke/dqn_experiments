from collections import deque, namedtuple

import random
import numpy as np
import torch
import typing

Experience = namedtuple('Experience', 'last_state last_action reward discount state action done')


class ReplayMemory:
    """
    ReplayMemory stores experience in form of tuples  (last_state last_action reward discount state action done)
    in a deque of maximum length buffer_size.

    Methods:
        add(self, sample): add a sample to the buffer
        sample_batch(self, batch_size): return an experience batch of size batch_size
        update_priorities(self, indices, weights): not implemented, needed for prioritizied replay buffer
        number_samples(self): returns the number of samples currently stored.
    """

    def __init__(self,
                 device: torch.device,
                 buffer_size: int,
                 gamma: float, number_steps: typing.Optional[torch.Tensor]) -> None:
        """
        Initializes the memory buffer

        Args:
            device(str): "gpu" or "cpu"
            buffer_size(int): maximum number of elements in the ReplayMemory
            gamma(float): decay factor
            number_steps(torch.Tensor): not used yet
        """
        self.data = deque(maxlen=buffer_size)
        self.device = device
        self.range = np.arange(0, buffer_size, 1)
        self.gamma = gamma
        self.number_steps = number_steps
        return

    def add(self, sample: Experience) -> None:
        """
        Adds experience to the buffer.
        Args:
            sample(tuple): tuple of 'last_state last_action reward discount state action done'
        Returns:
            None
        """
        self.data.append(sample)
        return

    def sample_batch(self, batch_size: int) -> tuple:
        """
        Samples a batch of size batch_size and returns a tuple of PyTorch tensors.
        Args:
            batch_size(int):  number of elements for the batch

        Returns:
            tuple of tensors
        """
        states, actions = [], []
        last_states, last_actions = [], []
        rewards, discounts, dones = [], [], []

        experiences = random.sample(self.data, k=batch_size)
        for experience in experiences:
            last_states.append(experience.last_state)
            last_actions.append(experience.last_action)
            rewards.append(experience.reward)
            discounts.append(experience.discount)
            states.append(experience.state)
            actions.append(experience.action)
            dones.append(experience.done)

        last_states = torch.from_numpy(np.array(last_states)).float().to(self.device)
        last_actions = torch.from_numpy(np.array(last_actions)).float().unsqueeze(-1).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(-1).to(self.device)
        discounts = torch.from_numpy(np.array(discounts).astype(np.float32)).float().unsqueeze(-1).to(self.device)
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).float().unsqueeze(-1).to(self.device)
        dones = torch.from_numpy(np.array(dones)).bool().unsqueeze(-1).to(self.device)
        return tuple((last_states, last_actions, rewards, discounts, states, actions, dones, None, None))

    def update_priorities(self, indices: typing.Optional[np.array], priorities: typing.Optional[np.array]):
        """
        This method later needs to be implemented for prioritized experience replay.
        Args:
            indices(list(int)): list of integers with the indices of the experience tuples in the batch
            priorities(list(float)): priorities of the samples in the batch

        Returns:
            None
        """
        return

    def number_samples(self):
        """
        Returns:
              Number of elements in the Replay Memory
        """
        return len(self.data)
