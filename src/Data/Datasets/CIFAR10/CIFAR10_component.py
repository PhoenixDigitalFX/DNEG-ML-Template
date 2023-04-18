from typing import List, Optional, cast, Tuple

from dneg_ml_toolkit.src.Data.Datasets.BASE_Dataset import BASE_Dataset
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from dneg_ml_toolkit.src.Data.Collate.BASE_Collate import BASE_Collate
from dneg_ml_toolkit.src.Data.Transforms.ToTensor import ToTensorConfig, ToTensor

# Import the Component's corresponding Config with a relative import
from .CIFAR10_config import CIFAR10Config

from torchvision.datasets import CIFAR10 as pytorch_CIFAR10
import numpy as np


class CIFAR10(BASE_Dataset):
    """
    Dataset Component to connect to the PyTorch CIFAR10 dataset and serve its images to the Dataloader

    Args:
        config: Dataset config object
        collate_component: Reference to the Collate Component. This can be used to access the Collate's Transform
            pipeline, and apply Transforms in the Dataset's getitem if the Collate is configured to not apply
            the Transforms (Transforming in the Dataset is more efficient as it uses the Dataloader's workers,
            but the Transforms cannot be applied with an awareness of other batch elements.)
    """

    def __init__(self, config: CIFAR10Config, collate_component: BASE_Collate):

        # CIFAR10 returns PIL images, so append a ToTensor Transform to the end of the
        # configured Transforms so that the data is always returned in tensor format.
        collate_component.append_transform(ToTensor(ToTensorConfig(Type="ToTensor", ApplyTo=["data"])))

        super().__init__(config, allow_multiple_sources=False, check_source_exists=False,
                         collate_component=collate_component)

        # This just informs the type checker that the config is of type CIFAR10Config,
        # since self.config is set in the parent's constructor, which is not aware of CIFAR10Config,
        # so it types self.config as the parent BASE_DatasetConfig.
        # It doesn't affect the code execution, but provides robustness through type checking, and
        # helps with code completion and access to documentation when using an IDE.
        self.config: CIFAR10Config = cast(CIFAR10Config, self.config)

        # Create a connection to the data - for folder-type datasets, this can just be a list of the files in the
        # folder. DNEG ML Toolkit provides some standard data connections,
        # such as ImageFileConnection, JSONFileConnection
        self.data_source = pytorch_CIFAR10(root=config.Source, train=self.config.TrainingSet, download=True)

        # Private variables that are lazy-loaded by reading the dataset metadata in some of theDataset's functions
        self._image_resolution: Optional[List[int]] = None
        self._classes: Optional[List[int]] = None

    def __len__(self) -> int:
        return len(self.data_source)

    def get_item_data(self, index: int) -> Tuple[MLToolkitDictionary, Optional[MLToolkitDictionary]]:
        """
        Get the data for the provided index. The Dataloader calls this function through the Dataset's __getitem__
        if DataReadMode is DataOnly or DataAndMetadata

        Args:
            index: Valid dataset index

        Returns:
            An MLToolkitDictionary of the data associated with the index; An optional dictionary of additional
            target data generated from the data. The data dictionary is routed into the network, the target dict
            is routed into the losses. The target dict is combined with the output of get_item_metadata if the Dataset's
            DataReadMode is DataAndMetadata (i.e. its default mode).
        """

        # All data is transported through ML Toolkit systems in ML Toolkit dictionaries
        # (a custom dictionary for holding Tensors). This provides flexibility for training,
        # as multiple tensors can be passed into the forward pass of the Network and Loss at the same time.
        # The ML Toolkit standard is for the Dataset to store the core tensor, such as the image in this case,
        # under the "data" keyword, and the ground truth under the "target" keyword.
        image, _ = self.data_source[index]
        train_dict = MLToolkitDictionary({"data": image, "index": index})

        # CIFAR10 has no target data generated from the image, so don't return a target dict
        return train_dict, None

    def get_item_metadata(self, index: int) -> MLToolkitDictionary:
        """
        Get the metadata for the provided index. CIFAR10 stores the data and the metadata together,
        but the standard for ML Toolkit datasets is to store the data and metadata separately, so that
        analysis on the metadata can be performed without having to load all the data into memory.

        Args:
            index: Valid dataset index

        Returns:
            An MLToolkitDictionary of the metadata associated with the index
        """

        _, target = self.data_source[index]

        metadata = MLToolkitDictionary({"target": target, "index": index})

        return metadata

    def get_data_shape(self) -> List[int]:
        """
        Get the image resolution of the data in the dataset, assuming that all images have the same resolution. This
        is passed to the Network and can be used to drive the creation of the Network architecture

        Returns:
            Image resolution of the image data after all Transforms have been applied, in [H,W,C] format
        """

        if self._image_resolution is None:
            # If the Collate normally applies transforms, call its apply transforms function manually so that the data
            # shape is the correct shape after transforming. Otherwise, just call getitem, which applies the transforms
            # automatically when Collate doesn't
            if self._collate_component.apply_transforms_in_collate:
                data, _ = self._collate_component.apply_transforms_to_sample(*self.__getitem__(0))
            else:
                data, _ = self.__getitem__(0)

            shape = np.array(data['data'].shape)  # Shape in C, H, W

            # Store the image resolution once it has been retrieved, so that future calls to this function
            # do not need to do any data processing
            self._image_resolution = np.roll(shape, -1).tolist()  # Swap dimensions to H, W, C

        return self._image_resolution

    def get_classes(self) -> List[int]:
        """
        Gets a list of all the unique classes in the dataset. See the ClassificationTrainModule where this is used
        to dynamically configure the Network's output shapes based on the number of classes in the dataset.

        Returns:
            Unique class list
        """

        if self._classes is None:
            # Only loop through the full dataset once to get the classes, then store the result
            # and use it for all future calls to this function
            self._classes = []
            for i in range(len(self)):
                self._classes.append(self.get_item_metadata(i)["target"])

        unique_classes = list(set(self._classes))

        return unique_classes
