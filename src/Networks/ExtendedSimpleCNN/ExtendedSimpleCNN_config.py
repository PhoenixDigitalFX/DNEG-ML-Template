from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from dneg_ml_toolkit.src.Networks.BASE_Network import BASE_NetworkConfig
from dneg_ml_toolkit.src.Networks.layers import ActivationType


@dataclass
class ExtendedSimpleCNNConfig(BASE_NetworkConfig):
    """
    Configuration dataclass for the Extended Simple CNN example Network, which demonstrates how you can customize
    an ML Toolkit Component by exposing fields to JSON configuration.

    Attributes:
        NumOutputs (int): Number of outputs returned by the final layer of the Network. If using the
            ClassificationTrainModule, this field will be set automatically by reading the number of classes
            from the dataset.
        BatchNorm (bool, default: false), Whether to enable batch normalization on the Network's convolution layers.
        Activation (enum:[RelU, PReLU, LeakyReLU], default: ReLU), What type of activation function to use for the
            convolution layers.
        ActivationNegativeSlope (Optional float), Required if Activation is LeakyReLU. Set the negative slope of the
            activation function.
    """

    # Normally, this type of field would not be flagged as optional, as it is a required value to initialize the
    # Network, and we would want the JSON validation to raise an error if it was not configured.
    # However, if using the ClassificationTrainModule, this field will be set automatically after the JSON
    # validation is performed, so flagging it as optional allows the validation to succeed in the case where we don't
    # give it a value in the JSON. If we are not using the ClassificationTrainModule
    # (i.e. using the StandardTrainModule from the core Toolkit), failing to configure
    # this field in the JSON will not raise a validation error as we would normally want. To solve this, the
    # Network component asserts that this value is not None at the start of its constructor.
    NumOutputs: Optional[int] = None

    BatchNorm: bool = False
    Activation: ActivationType = ActivationType.ReLU
    ActivationNegativeSlope: Optional[float] = None  # Only needed for LeakyReLU

    # JSON validation rule to ensure that ActivationNegativeSlope is configured when using
    # LeakyReLU activation.
    # For any field, you can create a field with an "_" prefix and "_conditional" suffix to define
    # custom JSON schema rules
    _ActivationNegativeSlope_conditional: Dict[str, Any] = field(
        default_factory=lambda: {
            "if": {
                "properties": {"Activation": {"const": "LeakyReLU"}}
            },
            "then": {
                "required": ["ActivationNegativeSlope"],
                # The JSON validation system accepts the custom "error_message" property which can be used to display
                # more helpful error messages.
                "properties": {"error_message": "ActivationNegativeSlope is required when using LeakyReLU Activation."}
            }

        })
