"""PyTorch model definition for house price prediction."""

import torch
import torch.nn as nn


def get_activation_fn(activation_name):
    """
    Get activation function by name.

    Args:
        activation_name (str): Name of activation function

    Returns:
        nn.Module: Activation function module
    """
    activation_map = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(negative_slope=0.01),
        'ELU': nn.ELU(),
        'GELU': nn.GELU(),
        'SELU': nn.SELU(),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        'None': nn.Identity()
    }

    return activation_map.get(activation_name, nn.ReLU())


class HousingMLP(nn.Module):
    """
    Multi-Layer Perceptron for house price regression.

    Args:
        input_size (int): Number of input features
        hidden_layers (list): List of hidden layer sizes
        dropout_rate (float): Dropout probability
        use_batch_norm (bool): Whether to use batch normalization
        activation_fn (str): Activation function name
    """

    def __init__(self, input_size=8, hidden_layers=[128, 64], dropout_rate=0.2, use_batch_norm=True, activation_fn='ReLU'):
        super(HousingMLP, self).__init__()

        layers = []
        prev_size = input_size
        activation = get_activation_fn(activation_fn)

        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size)) ### <-- conexion densa entre las capas de inicio y la primera capa oculta

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer (single neuron for regression)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers) #### <-- creamos la red secuencial pasandole de manera plana toda la arquitectura y orden de la red

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


def initialize_model(config):
    """
    Initialize model from configuration.

    Args:
        config (dict): Full configuration dictionary

    Returns:
        HousingMLP: Initialized model
    """
    model_config = config.get("model_config", {})
    arch_config = model_config.get('architecture', {})



    # Actualizar input_size basado en si usamos polynomial features
    use_polynomial = config.get('use_polynomial_features', False) ## <- sacamos de config por defecto falso
    input_size = 22 if use_polynomial else 12

    model = HousingMLP(
        input_size=input_size,
        hidden_layers=arch_config.get('hidden_layers', [128, 64]),
        dropout_rate=arch_config.get('dropout_rate', 0.2),
        use_batch_norm=arch_config.get('use_batch_norm', True),
        activation_fn=arch_config.get('activation_fn', 'ReLU')
    )

    return model
