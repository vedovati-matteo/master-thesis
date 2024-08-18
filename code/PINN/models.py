import torch.nn as nn
import torch.optim as optim
from torch import vmap
from torch.func import jacrev
import torch

class LinearNN(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_layers,
        num_neurons,
        num_outputs,
        act: nn.Module = nn.Tanh(),
    ) -> None:
        """Basic neural network architecture with linear layers
        
        Args:
            num_inputs (int, optional): the dimensionality of the input tensor
            num_layers (int, optional): the number of hidden layers
            num_neurons (int, optional): the number of neurons for each hidden layer
            act (nn.Module, optional): the non-linear activation function to use for stitching
                linear layers togeter
        """
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.num_layers = num_layers

        layers = []

        # input layer
        layers.append(nn.Linear(self.num_inputs, num_neurons))

        # hidden layers with linear layer and activation
        for _ in range(num_layers):
            layers.extend([nn.Linear(num_neurons, num_neurons), act])

        # output layer
        layers.append(nn.Linear(num_neurons, num_outputs))

        # build the network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.reshape(-1, 1)).squeeze()
    

class NewArchNN(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_layers,
        num_neurons,
        num_outputs,
        act: nn.Module = nn.Tanh(),
    ) -> None:
        """Basic neural network architecture with linear layers
        
        Args:
            num_inputs (int, optional): the dimensionality of the input tensor
            num_layers (int, optional): the number of hidden layers
            num_neurons (int, optional): the number of neurons for each hidden layer
            act (nn.Module, optional): the non-linear activation function to use for stitching
                linear layers togeter
        """
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        
        self.act = act

        self.encoder_1 = nn.Linear(num_inputs, num_neurons)
        self.encoder_2 = nn.Linear(num_inputs, num_neurons)

        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(nn.Linear(self.num_inputs, num_neurons))

        # hidden layers with linear layer and activation
        for _ in range(num_layers):
            self.layers.append(nn.Linear(num_neurons, num_neurons))

        # output layer
        self.out_layer = nn.Linear(num_neurons, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        H = x.reshape(-1, 1)
        
        U = self.act(self.encoder_1(H))
        V = self.act(self.encoder_1(H))
        
        for layer in self.layers:
            Z = self.act(layer(H))
            H = torch.mul((1 - Z), U) + torch.mul(Z, V)
        
        f = self.out_layer(H)

        
        return f.squeeze()