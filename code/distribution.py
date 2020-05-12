import numpy as np
import torch
import torch.nn as nn
from noisy_linear import NoisyLinear
from typing import Callable

"""
This file contains different classes that implement DistributionalDQN.
These classes are inspired by the Deep Reinforcement Learning: Hands-On book [1]
and the original manuscript that described Distributional DQN [2].

[1] Lapan, Maxim. Deep Reinforcement Learning Hands-On, Packt Publishing Ltd, 2018.
[2] Bellemare, Marc G., Will Dabney, and Rémi Munos. "A distributional perspective on reinforcement learning."
    Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.
"""


class DistributionUpdater:
    def __init__(self, vmin: int, vmax: int, number_atoms: int) -> None:
        """
        Initializes the DistributionUpdater.
        This class is used to perform the distributional Bellman update.
        Args:
            vmin: min value of the support
            vmax: maximum value of support
            number_atoms: number of support atoms
        """
        self.vmin = vmin
        self.vmax = vmax
        self.number_atoms = number_atoms
        self.delta_z = (vmax - vmin) / (number_atoms - 1)
        self.support = np.linspace(vmin, vmax, num=number_atoms)

    def update_distribution(self, previous_distribution: np.ndarray, rewards: np.ndarray, dones: np.ndarray,
                            gamma: np.ndarray = 0.9) -> np.ndarray:
        """
        Calculates the distributional Bellman update.
        Args:
            previous_distribution(np.ndarray):  previous distributions, size: batch_size x number_atoms
            rewards(np.ndarray):  rewards, size: batch_size x 1
            dones(np.ndarray): dones, boolean flag, size: batch_size x 1
            gamma(np.ndarray): gamma, size batch_size x 1

        Returns:
            np.ndarray: Bellman updated distributions
        """
        batch_size = previous_distribution.shape[0]
        next_distribution = np.zeros_like(previous_distribution)
        ones = np.ones(batch_size, dtype=np.int)

        for j in np.arange(self.number_atoms):
            # Reward + discounting and clipping
            tzj = rewards + gamma * self.support[j]
            tzj = np.maximum(np.minimum(tzj, self.vmax), self.vmin)
            # project it to the atom space
            bj = ((tzj - self.vmin) / self.delta_z)
            # distribute the values to the neighboring atoms
            low = np.floor(bj).astype(np.int)
            up = np.ceil(bj).astype(np.int)
            upper_weight = bj - low
            lower_weight = 1. - upper_weight
            next_distribution[range(batch_size), low.squeeze()] += previous_distribution[range(
                batch_size), j * ones] * lower_weight.squeeze()
            next_distribution[range(batch_size), up.squeeze()] += previous_distribution[range(
                batch_size), j * ones] * upper_weight.squeeze()

        if dones.any():
            # Set distributions to zero
            next_distribution[dones.squeeze(), :] = 0.0
            # Clip reward
            tzj = np.maximum(np.minimum(rewards[dones], self.vmax), self.vmin)
            # project it to the atom space
            bj = ((tzj - self.vmin) / self.delta_z).squeeze()
            # distribute the values to the neighboring atoms
            low = np.floor(bj).astype(np.int)
            up = np.ceil(bj).astype(np.int)
            upper_weight = bj - low
            lower_weight = 1. - upper_weight
            next_distribution[dones.squeeze(), low] += lower_weight
            next_distribution[dones.squeeze(), up] += upper_weight
        return next_distribution


class DistributionalNetHelper:
    def __init__(self, settings: dict, neural_network_call: Callable, device: torch.device) -> None:
        """
        Args:
            settings: dictionary with settings
            neural_network_call(Callable): function call to the neural network
            device(torch.device): "gpu" or "cpu"
        """
        self.number_atoms = settings["number_atoms"]
        vmax = settings["vmax"]
        vmin = settings["vmin"]
        delta = (vmax - vmin) / (self.number_atoms - 1)
        self.supports = torch.arange(vmin, vmax + delta, delta).to(device)
        self.forward = neural_network_call

    def calc(self, x: torch.Tensor) -> tuple:
        """
        Gets the distributions from the neural network and calculates the q-values
        Args:
            x(torch.Tensor): observation or a batch of observations
        Returns:
            tuple(torch.Tensor, torch.Tensor): distributions, q-values
        """
        batch_size = 1
        if x.ndimension() > 1:
            batch_size = x.size()[0]
        distributions = self.forward(x).view(batch_size, -1, self.number_atoms)
        probabilities = self.softmax(distributions)
        qvals = (probabilities * self.supports).sum(dim=2)
        return distributions, qvals

    def get_max_action(self, x: torch.Tensor) -> np.ndarray:
        """
        Args:
            x(torch.Tensor): batch of observations
        Returns:
            np.ndarray: action with largest q-value for each sample
        """
        _, qvals = self.calc(x)
        action = torch.argmax(qvals, dim=-1).cpu().detach().numpy()
        return action

    def softmax(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the softmax to each individual distribution (one distribution per sample and action)
        Args:
            x(torch.Tensor): distributions of size: batch_size x action_size x number_atoms
        Returns:
            torch.Tensor of same size as input
        """
        return nn.functional.softmax(x.view(-1, self.number_atoms), dim=1).view(x.shape)


class DistributionalDuelDQN(nn.Module, DistributionalNetHelper):
    def __init__(self, states_size: int, action_size: int, settings: dict, device: torch.device) -> None:
        """
        Initializes the DistributionalDuelDqn
        Args:
            states_size (int): Size of the input space.
            action_size (int):Size of the action space.
            settings (dict): dictionary with settings
            device( torch.device): "gpu" or "cpu"
        """
        super(DistributionalDuelDQN, self).__init__()
        DistributionalNetHelper.__init__(self, settings, neural_network_call=self.forward, device=device)
        self.batch_size = settings["batch_size"]
        self.number_atoms = settings["number_atoms"]
        layers_size = settings["layers_sizes"][0]
        self.noisy_net = settings['noisy_nets']
        if not self.noisy_net:
            self.FC1 = nn.Linear(int(states_size), layers_size)
            self.FC2 = nn.Linear(layers_size, layers_size)
            self.FC3v = nn.Linear(layers_size, self.number_atoms)
            self.FC3a = nn.Linear(layers_size, int(action_size * self.number_atoms))
        else:
            self.FC1 = NoisyLinear(int(states_size), layers_size)
            self.FC2 = NoisyLinear(layers_size, layers_size)
            self.FC3v = NoisyLinear(layers_size, self.number_atoms)
            self.FC3a = NoisyLinear(layers_size, int(action_size) * self.number_atoms)
        self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the distributional neural networ
        Args:
            x(torch.Tensor): a batch of observations
        Returns:
            torch.Tensor: distributions for each sample and action, size: batch_size x action_size x number_atoms
        """
        if x.ndimension() == 1:
            batch_size = 1
        else:
            batch_size = x.size()[0]
        x = nn.functional.relu(self.FC1(x))
        x = nn.functional.relu(self.FC2(x))
        a = self.FC3a(x)
        v = self.FC3v(x)

        a = a.view([batch_size, -1, self.number_atoms])
        average = a.mean(1).unsqueeze(1)
        a_scaled = a - average
        if batch_size > 1:
            v = v.unsqueeze(1)
        return_vals = v + a_scaled
        return return_vals

    def reset(self) -> None:
        """
        Resets the weights of the neural network layers and the noise of the noisy layers.
        Returns:
            None
        """
        torch.nn.init.xavier_uniform_(self.FC1.weight.data)
        torch.nn.init.xavier_uniform_(self.FC2.weight.data)
        torch.nn.init.xavier_uniform_(self.FC3a.weight.data)
        torch.nn.init.xavier_uniform_(self.FC3v.weight.data)
        if self.noisy_net:
            self.reset_noise()

    def reset_noise(self) -> None:
        """
        Samples noise for the noisy layers.
        """
        self.FC1.reset_noise()
        self.FC2.reset_noise()
        self.FC3a.reset_noise()
        self.FC3v.reset_noise()
