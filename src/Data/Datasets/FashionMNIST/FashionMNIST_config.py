from dataclasses import dataclass

from dneg_ml_toolkit.src.Data.Datasets.BASE_Dataset import BASE_DatasetConfig

# All Component Config classes must have the @dataclass decorator. A Python dataclass defines fields (type-annotated
# variables). DNEG ML Toolkit uses this structure to expose the fields to the JSON system, mapping and type-validating
# JSON values into Component Config objects.
# Note that all components must have a correctly implemented Config class, even if it does not define any fields. The
# configuration system uses the structure and naming conventions of the Config class and corresponding Component
# class to create the Components when the system is being initialized.
# Fields are inherited from the parent Config class, so a Component may have more configuration and fields with
# default values that can be customized.
@dataclass
class FashionMNISTConfig(BASE_DatasetConfig):
    """
    Configuration dataclass for the FashionMNIST Dataset. Any dataclass fields defined here are configurable from JSON,
    and this object is used to create the FashionMNIST Dataset object, initialized based on the values of the fields.

    Attributes:
        TrainingSet (boolean, default: true); Whether to use the training or testing subset of FashionMNIST

    """
    TrainingSet: bool = True
