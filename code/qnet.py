import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class Qnet(nn.Module):
    def __init__(self) -> None:
        super(Qnet, self).__init__()
        return

    def get_max_action(self, observation: torch.Tensor) -> int:
        """
        Get the action with the maximum q-value for an observation.
        Args:
            observation(torch.Tensor): an observation
        Returns:
            int: action with the maximum q-value for the current state
        """
        qvals = self.forward(observation)
        return int(torch.argmax(qvals, dim=-1).cpu().detach().numpy())


class Dqn(Qnet):
    def __init__(self, states_size: np.ndarray, action_size: np.ndarray, settings: dict) -> None:
        """
        Initializes the neural network.
        Args:
            states_size: Size of the input space.
            action_size:Size of the action space.
            settings: dictionary with settings, currently not used.
        """
        super(Dqn, self).__init__()
        self.batch_size = settings["batch_size"]
        layers_size = settings["layers_sizes"][0]
        self.FC1 = nn.Linear(int(states_size), layers_size)
        self.FC2 = nn.Linear(layers_size, layers_size)
        self.FC3 = nn.Linear(layers_size, int(action_size))
        self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward step of the neural network
        Args:
            x(torch.Tensor): observation or a batch of observations

        Returns:
            torch.Tensor: q-values for all  observations and actions
        """
        x = functional.relu(self.FC1(x))
        x = functional.relu(self.FC2(x))
        return self.FC3(x)

    def reset(self) -> None:
        """
        Resets the weights of the neural network layers.
        Returns:
            None
        """
        torch.nn.init.xavier_uniform_(self.FC1.weight.data)
        torch.nn.init.xavier_uniform_(self.FC2.weight.data)
        torch.nn.init.xavier_uniform_(self.FC3.weight.data)


class DuelDQN(Qnet):
    def __init__(self, states_size: np.ndarray, action_size: np.ndarray, settings: dict) -> None:
        """
        Initializes the neural network.
        Args:
            states_size: Size of the input space.
            action_size:Size of the action space.
            settings: dictionary with settings, currently not used.
        """
        super(DuelDQN, self).__init__()
        self.batch_size = settings["batch_size"]
        layers_size = settings["layers_sizes"][0]
        self.FC1 = nn.Linear(int(states_size), layers_size)
        self.FC2 = nn.Linear(layers_size, layers_size)
        self.FC3v = nn.Linear(layers_size, 1)
        self.FC3a = nn.Linear(layers_size, int(action_size))

        self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward step of the duelling q-network
        Args:
            x(torch.Tensor): observation or a batch of observations

        Returns:
            torch.Tensor: q-values for all  observations and actions
        """
        x = functional.relu(self.FC1(x))
        x = functional.relu(self.FC2(x))
        v = self.FC3v(x)
        a = self.FC3a(x)
        if x.ndimension() == 1:
            qvals = v + (a - torch.mean(a))
        else:
            qvals = v + (a - torch.mean(a, 1, True))
        return qvals

    def reset(self) -> None:
        """
        Resets the weights of the neural network layers.
        Returns:
            None
        """
        torch.nn.init.xavier_uniform_(self.FC1.weight.data)
        torch.nn.init.xavier_uniform_(self.FC2.weight.data)
        torch.nn.init.xavier_uniform_(self.FC3a.weight.data)
        torch.nn.init.xavier_uniform_(self.FC3v.weight.data)
