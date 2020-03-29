from collections import deque, namedtuple

import random
import numpy as np
import torch
import typing

Experience = namedtuple('Experience', 'state action reward discount next_state next_action done')


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
                 memory_size: int,
                 gamma: float,
                 number_steps: typing.Optional[torch.Tensor]) -> None:
        """
        Initializes the memory buffer

        Args:
            device(str): "gpu" or "cpu"
            memory_size(int): maximum number of elements in the ReplayMemory
            gamma(float): decay factor
            number_steps(torch.Tensor): not used yet
        """
        self.gamma = gamma
        self.number_steps = number_steps
        self.device = device
        self.range = np.arange(0, memory_size, 1)

        self.data = deque(maxlen=memory_size)
        self.buffer = deque(maxlen=number_steps)
        return

    def add(self, sample: Experience) -> None:
        """
        Adds experience to a buffer. Once the buffer is at full capacity or when the episode
        is over, elements are added to the ReplayMemory.
        Args:
            sample(Experience): tuple of 'last_state last_action reward discount state action done'
        Returns:
            None
        """
        self.buffer.appendleft(sample)

        if sample.done:
            while self.buffer:
                self.add_to_memory()
                self.buffer.pop()

        if len(self.buffer) == self.number_steps:
            self.add_to_memory()
        return

    def add_to_memory(self) -> None:
        """
            Adds experience to the memory after calculating the n-step return.
        Args:

        Returns:

        """
        buffer = self.buffer
        if len(buffer) == 0:
            return

        reward = 0.0
        for element in buffer:
            reward = element.reward + self.gamma * reward

        first_element = self.buffer[-1]
        last_element = self.buffer[0]

        exp = Experience(first_element.state,
                         first_element.action,
                         reward,
                         last_element.discount,
                         last_element.state,
                         0,
                         last_element.done
                         )
        self.data.append(exp)
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
        next_states, next_actions = [], []
        rewards, discounts, dones = [], [], []

        experiences = random.sample(self.data, k=batch_size)
        for experience in experiences:
            states.append(experience.state)
            actions.append(experience.action)
            rewards.append(experience.reward)
            discounts.append(experience.discount)
            next_states.append(experience.next_state)
            next_actions.append(experience.next_action)
            dones.append(experience.done)

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).float().unsqueeze(-1).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(-1).to(self.device)
        discounts = torch.from_numpy(np.array(discounts).astype(np.float32)).float().unsqueeze(-1).to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        next_actions = torch.from_numpy(np.array(next_actions)).float().unsqueeze(-1).to(self.device)
        dones = torch.from_numpy(np.array(dones)).bool().unsqueeze(-1).to(self.device)
        return tuple((states, actions, rewards, discounts, next_states, next_actions, dones, None, None))

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
