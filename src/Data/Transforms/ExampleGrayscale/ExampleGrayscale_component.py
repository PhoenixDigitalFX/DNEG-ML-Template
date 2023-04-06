import copy
from typing import cast, Any, Optional, Dict, Tuple
from dneg_ml_toolkit.src.Data.Transforms.BASE_Transform.BASE_Transform_component import BASE_Transform
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from dneg_ml_toolkit.src.Data.image_tools import image_dtype_utils

from src.Data.Transforms.ExampleGrayscale.ExampleGrayscale_config import ExampleGrayscaleConfig

from torchvision.transforms import Grayscale
import numpy as np


class ExampleGrayscale(BASE_Transform):
    """
    Transforms are used to apply data augmentations as part of the data pipeline. A Dateset loads a
    data item along with its metadata (and potentially other data elements) and packages it into a sample,
    which is passed through the Transforms in the order they were configured. Each Transform requests items
    from the sample and transforms them, adding them back into the sample either in-place, or as a new data entry.

    This Transform is just example showcasing the main features that are available when applying a data Transform.
    """

    def __init__(self, config: ExampleGrayscaleConfig):
        # The Transform will apply the data augmentation to any data specified in its configuration's
        # ApplyTo field.
        # It may specify an additional list of keys and pass it to the parent constructor as the
        # required_transform_metadata parameter. These data items will always be retrieved and are used
        # for calculating the augmentation to apply to the ApplyTo data. These metadata items will not
        # be altered by the Transform
        # This example just requests the dataset index for the piece of data
        required_transform_metadata = ["index"]

        super().__init__(config, required_transform_metadata=required_transform_metadata)
        self.config: ExampleGrayscaleConfig = cast(ExampleGrayscaleConfig, config)

        # This Transform wraps around PyTorch's Grayscale transform
        self.to_grayscale = Grayscale()

    def apply_transform(self, data_key: str, data_to_transform: Any, transform_metadata: Optional[Dict[str, Any]]) \
            -> Tuple[Any, Optional[MLToolkitDictionary[Any]]]:
        """
        Apply the Transform to the data

        Args:
             data_key: The key that identifies the data
            data_to_transform: The piece of data to apply the transformation to
            transform_metadata: An optional dictionary of additional data gathered using the keys passed as
                required_transform_metadata in the constructor. This additional data does not get
                transformed, but can be used for the calculation and application of the transform
                to the data_to_transform.

        Returns:
            The transformed piece of data; an optional dictionary of additional data that will be added to the same
                dictionary that the transformed data is stored in, for cases where the Transform generates new data from
                the input data.
        """

        # 1. Get the data type of the input data (and the device if it is a tensor). These are tracked so that
        # the transformed data can be returned with the matching data type
        input_datatype, device = image_dtype_utils.get_image_datatype(data_to_transform)

        # 2. Torchvision's Grayscale transform expects the data to be in PIL image format, so ensure it is in PIL image
        # format using the ML Toolkit's helper function
        data_to_transform, _ = image_dtype_utils.transform_data_type(data_to_transform,
                                                                     to_type=image_dtype_utils.ImageDataType.PILImage)

        # 4. Perform the transformation of the data
        grayscale_image = self.to_grayscale(data_to_transform)

        # 5. Convert the transformed image back to the original Data type.
        transformed_data, _ = image_dtype_utils.transform_data_type(grayscale_image, to_type=input_datatype,
                                                                    device=device)

        # 6. This just shows that any data requested with required_transform_metadata in the constructor will
        # be available in transform_metadata. This can be used where a Transform requires additional data to perform
        # the transformation (for example, applying a mask to an image).
        assert "index" in transform_metadata, "index was not found in the transform metadata, even though it was " \
                                              "requested by the Transform."

        # 7. Example of random number generation for Transforms
        # The core Transform logic generates a new random seed each step, and this seed is available to this function
        # when it is called for each piece of data in a step. Using this seed to initialize a random number generator
        # in this function ensures that there is consistency between the Transformation being applied to multiple pieces
        # of data in a step
        rng = np.random.default_rng(seed=self.random_seed)
        # For example, if a Transform randomly rotates an image, and the Transform is configured to
        # ApplyTo: ["data", "ground_truth"], using this approach ensures that the same random rotation is generated
        # for both the data and ground_truth images.
        # This rotation value will be the same for each data item in the ApplyTo list for the current step
        rotation = rng.uniform(0, 180)

        additional_data = None

        # 8. Example showing how a Transform can generate additional data items that will be added back
        # into the data dictionary as new entries
        if self.config.SplitChannels:
            data_to_transform_numpy, _ = image_dtype_utils.transform_data_type(data_to_transform,
                                                                               to_type=image_dtype_utils.ImageDataType.NPArray)

            red_channel = data_to_transform_numpy[:, :, 0]  # Single channel [HxW]
            red_channel = red_channel.reshape((*red_channel.shape, 1))  # Image in [HxWxC]
            red_channel, _ = image_dtype_utils.transform_data_type(red_channel,
                                                                   to_type=image_dtype_utils.ImageDataType.NPArray)

            green_channel = data_to_transform_numpy[:, :, 1]  # Single channel [HxW]
            green_channel = green_channel.reshape((*green_channel.shape, 1))  # Image in [HxWxC]
            green_channel, _ = image_dtype_utils.transform_data_type(green_channel,
                                                                     to_type=image_dtype_utils.ImageDataType.NPArray)

            blue_channel = data_to_transform_numpy[:, :, 2]  # Single channel [HxW]
            blue_channel = blue_channel.reshape((*blue_channel.shape, 1))  # Image in [HxWxC]
            blue_channel, _ = image_dtype_utils.transform_data_type(blue_channel,
                                                                    to_type=image_dtype_utils.ImageDataType.NPArray)

            additional_data = MLToolkitDictionary(
                {"red_channel": red_channel, "green_channel": green_channel, "blue_channel": blue_channel})

        return transformed_data, additional_data
