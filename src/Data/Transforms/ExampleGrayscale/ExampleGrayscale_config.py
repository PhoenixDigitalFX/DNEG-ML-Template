from dataclasses import dataclass

from dneg_ml_toolkit.src.Data.Transforms.BASE_Transform import BASE_TransformConfig


@dataclass
class ExampleGrayscaleConfig(BASE_TransformConfig):
    """
    Config dataclass for the Example Greyscale Transform

    """

    SplitChannels: bool = True
