from typing import List, Optional

from dneg_ml_toolkit.src.Networks.BASE_Network.BASE_Network_component import BASE_Network
import dneg_ml_toolkit.src.Networks.layers as dneg_ml
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from src.Networks.ExtendedSimpleCNN.ExtendedSimpleCNN_config import ExtendedSimpleCNNConfig

import torch.nn as nn


class ExtendedSimpleCNN(BASE_Network):
    """
    The same network architecture as the SimpleCNN Network, but extended with additional configuration parameters
    that allow features of the network to be controlled from JSON configuration.

    Args:
        config: The ExtendedSimpleCNN's configuration object
        input_shape: Data shape in [H,W,C] format
    """

    def __init__(self, config: ExtendedSimpleCNNConfig, input_shape: List[int]):
        super().__init__(config, input_shape)

        # Inform the type checker that the config is of type ExampleCNNConfig
        self.config: ExtendedSimpleCNNConfig = config

        assert self.config.NumOutputs is not None, "NumOutputs must have a valid positive value"

        # Select the activation to use based on the configuration parameter
        activation = dneg_ml.get_activation(self.config.Activation)
        activation_params = {}
        if self.config.Activation == dneg_ml.ActivationType.LeakyReLU:
            activation_params["negative_slope"] = self.config.ActivationNegativeSlope

        # Configure the creation of each convolution with the selected activation, and the parameter to enable
        # batch normalization
        conv_1 = dneg_ml.Convolution2D(in_channels=input_shape[2], out_channels=8, kernel_size=3, stride=1, padding=1,
                                       batch_norm=self.config.BatchNorm, activation=activation, **activation_params)
        # DNEG ML Toolkit's Networks track each layer by name, accessible through the layers dictionary
        self.add_layer("conv_1", conv_1)
        # Track the output shape of each layer to inform the next layer of its input shape
        output_shape = conv_1.get_output_shape(input_shape=self.input_shape)

        conv_2 = dneg_ml.Convolution2D(in_channels=output_shape[2], out_channels=8, kernel_size=3, stride=1, padding=1,
                                       batch_norm=self.config.BatchNorm, activation=activation, **activation_params)
        self.add_layer("conv_2", conv_2)
        output_shape = conv_2.get_output_shape(input_shape=output_shape)

        pool_1 = dneg_ml.MaxPooling2D(2)
        self.add_layer("pool_1", pool_1)
        output_shape = pool_1.get_output_shape(input_shape=output_shape)

        conv_3 = dneg_ml.Convolution2D(in_channels=output_shape[2], out_channels=8, kernel_size=3, stride=1, padding=1,
                                       batch_norm=self.config.BatchNorm, activation=activation, **activation_params)
        self.add_layer("conv_3", conv_3)
        output_shape = conv_3.get_output_shape(input_shape=output_shape)

        pool_2 = dneg_ml.MaxPooling2D(2)
        self.add_layer("pool_2", pool_2)
        output_shape = pool_2.get_output_shape(input_shape=output_shape)

        conv_4 = dneg_ml.Convolution2D(in_channels=output_shape[2], out_channels=16, kernel_size=3, stride=1, padding=1,
                                       batch_norm=self.config.BatchNorm, activation=activation, **activation_params)
        self.add_layer("conv_4", conv_4)
        output_shape = conv_4.get_output_shape(input_shape=output_shape)

        flatten = dneg_ml.Flatten()
        self.add_layer("flatten", flatten)
        output_shape = flatten.get_output_shape(output_shape)

        self.add_layer("linear", nn.Linear(output_shape[0], self.config.NumOutputs))

        # Since there are no branching paths in this network, can just add all layers to a Sequential and have
        # a simple forward pass
        self.network = nn.Sequential(*self.layers.values())

        # Call this after self.network is created, as it applies to the submodules of this class
        self.init_layer_weights()

    def forward(self, train_dict: MLToolkitDictionary, step: Optional[int] = -1) -> MLToolkitDictionary:
        """
        Perform the forward pass on the network.

        Args:
            train_dict: All data is transported through ML Toolkit systems in ML Toolkit dictionaries (a custom
                dictionary for holding Tensors). This provides flexibility for training, as multiple tensors can
                be passed into the forward pass at the same time. The ML Toolkit standard is for the Dataset (see
                CIFAR10 or FashionMNIST) to store the core tensor, such as the image in this case, under the "data"
                keyword, and the ground truth under the "target" keyword.
            step: Allow the trainer to inform the Network of the current step

        Returns:
            The input ML Toolkit dictionary, with the "data" field updated with the Network outputs
        """

        # TODO DEBUG - Networks must support tensor-only inputs to export to onnx
        if isinstance(train_dict, dict):
            x = train_dict["data"]
        else:
            x = train_dict

        # x = train_dict["data"]
        output = self.network(x)  # Replace the network input in-place with the network output

        if isinstance(x, dict):
            # Update the "data" entry with the network output, preserving any other metadata in the dictionary
            train_dict["data"] = output
        else:
            train_dict = output

        return train_dict
