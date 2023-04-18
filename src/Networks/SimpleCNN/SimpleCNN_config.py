from dataclasses import dataclass
from typing import Optional

from dneg_ml_toolkit.src.Networks.BASE_Network import BASE_NetworkConfig


@dataclass
class SimpleCNNConfig(BASE_NetworkConfig):
    """
    Configuration dataclass for the Simple CNN example Network

    Attributes:
        NumOutputs (int): Number of outputs returned by the final layer of the Network. If using the
            ClassificationTrainModule, this field will be set automatically by reading the number of classes
            from the dataset.
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
