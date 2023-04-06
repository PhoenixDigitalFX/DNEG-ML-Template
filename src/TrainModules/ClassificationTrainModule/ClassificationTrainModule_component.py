from typing import List, Dict, Any, cast, Optional
import os

from dneg_ml_toolkit.src.Component.component_store import ComponentStore
from dneg_ml_toolkit.src.TrainModules.BASE_TrainModule.BASE_TrainModule_component import BASE_TrainModule
from dneg_ml_toolkit.src.Losses.BASE_Loss.BASE_Loss_component import BASE_Loss
from dneg_ml_toolkit.src.Networks.BASE_Network.BASE_Network_component import BASE_Network
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from dneg_ml_toolkit.src.TrainModules.BASE_TrainModule.BASE_TrainModule_config import ExecutionModeEnum
from dneg_ml_toolkit.src.globals import Globals
from dneg_ml_toolkit.src.Exporters.BASE_Exporter.BASE_Exporter_component import BASE_Exporter
from src.TrainModules.ClassificationTrainModule.ClassificationTrainModule_config import ClassificationTrainModuleConfig

import torch
from torchmetrics import Accuracy, MeanMetric


class ClassificationTrainModule(BASE_TrainModule):
    """
    The Classification Train Module performs training on a single Network, calculates loss on the Network output
    using 1 or more loss functions, and optimizes with the specified Optimizer, for image classification models.

    Since it is for image classification, the number of classes are read from the training dataset and this value is
    set as the number of network outputs.

    Args:
        config: ComponentConfig object for the TrainModule
        experiment_name: Name of the experiment
        experiment_folder: Save folder for the experiment
    """

    def __init__(self, config: ClassificationTrainModuleConfig, experiment_name: str, experiment_folder: str):

        super().__init__(config=config, experiment_name=experiment_name, experiment_folder=experiment_folder)
        self.config: ClassificationTrainModuleConfig = config

        assert hasattr(self.config.Network, "NumOutputs"), "Cannot use {} Network for image classification. " \
                                                           "It does not have a" \
                                                           "NumOutputs config field".format(
            self.config.Network.get_component_name())

        # Set the number of Network outputs from the data's classes
        self.config.Network.NumOutputs = self.config.NumClasses

        # build_component_from_config accepts kwargs that it passes to the constructor of whatever Component it is
        # creating. All Network constructors take input_shape as a parameter
        self.Network: BASE_Network = cast(BASE_Network,
                                          ComponentStore().build_component_from_config(self.config.Network,
                                                                                       input_shape=self.config.InputShape))

        # Only initialize the Loss and optimizer during training
        if self.config.Mode == ExecutionModeEnum.TRAIN:
            self.Losses: List[BASE_Loss]
            if isinstance(self.config.Loss, list):
                # Support multiple losses
                self.Losses = [cast(BASE_Loss, ComponentStore().build_component_from_config(loss)) for loss in
                               self.config.Loss]
            else:
                # If only a single loss is configured, create a list with one element, since the training loop
                # will loop over a list of losses
                self.Losses = [cast(BASE_Loss, ComponentStore().build_component_from_config(self.config.Loss))]

            # build_component_from_config accepts kwargs that it passes to the constructor of whatever Component it is
            # creating. All Optimizer constructors take optimizer_params as a parameter
            self.Optimizer = ComponentStore().build_component_from_config(self.config.Optimizer,
                                                                          optimizer_params=self.get_training_parameters())

            self.Scheduler = None
            if self.config.Scheduler is not None:
                # If the optional Learning Rate scheduler is configured, create the Scheduler Component
                self.Scheduler = ComponentStore().build_component_from_config(self.config.Scheduler)

            self.train_metrics = {
                'accuracy': Accuracy(),
                'loss': MeanMetric()
            }

            self.val_metrics = {
                'accuracy': Accuracy(),
                'loss': MeanMetric()
            }

        elif self.config.Mode == ExecutionModeEnum.TEST:
            # During testing, save images to file using the Toolkit's Image File Connection
            from dneg_ml_toolkit.src.Data.data_connections.image_file_connection import ImageFileConnection
            self._test_image_writers = {}
            # Create a file connection to write images into a separate folder for each class
            for i in range(self.config.NumClasses):
                self._test_image_writers[i] = ImageFileConnection(
                    source=os.path.join(experiment_folder, Globals().REPORTS_FOLDER, "class_{}".format(i)),
                    readonly=False)

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        """
        Lightning function to inform Lightning what optimizers and LR schedulers are being used

        Returns:
            List of configuration for each optimizer
        """
        optimizers = []
        optimizer_configuration = {"optimizer": self.Optimizer}

        # If a scheduler has been built, add it to the configuration for Lightning
        if self.Scheduler is not None:
            optimizer_configuration["lr_scheduler"] = {"scheduler": self.Scheduler,
                                                       "interval": self.config.Scheduler.Interval.value,
                                                       # Get the interval from the config's enum
                                                       "name": "lr"
                                                       }

        optimizers.append(optimizer_configuration)

        return optimizers

    def get_training_parameters(self) -> List[torch.nn.parameter.Parameter]:
        """
        Get the trainable parameters for the Network

        Returns:
            List of training parameters
        """

        parameters = []
        parameters.extend(list(self.Network.parameters()))

        return parameters

    def forward(self, batch_data: MLToolkitDictionary, step: Optional[int] = -1) -> MLToolkitDictionary:
        """
        Perform a forward pass on the Network

        Args:
            batch_data: The Data for the batch, which gets passed into the Network
            step: When training, pass the number of the current training iteration. -1 implies that the model
                is not training.

        Returns:
            The network output

        """

        return self.Network.forward(batch_data, step=step)

    def forward_testing(self, batch_data: MLToolkitDictionary, batch_metadata: MLToolkitDictionary, step: int) -> None:
        """
        Perform a testing forward pass on the model, saving the classified images into a folder corresponding to
        the predicted class

        This function is called by the base test_step for each testing batch when trainer.test is run
        on a Lightning trainer

        Args:
            batch_data: Batch of data to test
            batch_metadata: Any metadata to include with the batch
            step: the current testing step
        Returns:
            None
        """

        # Store a copy of the input images in a separate variable, since the network replaces
        # the batch "data" field in-place with the network output
        images = batch_data["data"].clone().detach()

        outputs = self.forward(batch_data, step=-1)

        # Get the max class value across the class predictions for each batch sample
        _, predicted = torch.max(outputs["data"], dim=1)

        # Save each image to the corresponding calss folder
        for i, image_class in enumerate(predicted):
            image = images[i]
            image_name = "step_{}_image_{}.png".format(step, i)

            self._test_image_writers[image_class.item()].write_data(
                {"image": image, "filename": image_name})

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Implements the training loop for the model.

        Args:
            batch: Current data batch
            batch_idx: index of the current batch

        Returns:
            Loss tensor
        """

        step_metrics = {}

        # The batch is a tuple of ML Toolkit Dictionaries, so separate it out into the
        # data to bass to the network, and the metadata (containing the targets) to pass to the Loss
        data, metadata = batch
        # Ensure that the created tensor is on the same device as the data
        device = data['data'].device
        total_loss = torch.zeros(1)
        total_loss = total_loss.to(device)

        # Perform the forward pass
        network_outputs = self.forward(data, step=self._train_step)

        batch_accuracy = self.train_metrics['accuracy'](
            network_outputs["data"].detach().cpu().softmax(-1), metadata["target"].cpu())

        step_metrics["batch_accuracy"] = batch_accuracy

        # Calculate loss for 1 or more losses
        for loss_function in self.Losses:
            loss = loss_function(network_outputs, metadata)
            total_loss += loss

            loss_name = loss_function.Name()
            step_metrics["step/{}".format(loss_name)] = loss.item()

        self.train_metrics['loss'].update(total_loss.detach().cpu())

        step_metrics["batch_loss"] = total_loss.item()

        # Log the step metrics
        self.log_dict(step_metrics, on_step=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Called during training to perform validation
        Args:
            batch:
            batch_idx:

        Returns:

        """

        data, targets = batch

        network_outputs = self.forward(data)

        self.val_metrics['accuracy'](
            network_outputs["data"].detach().cpu().softmax(-1), targets["target"].cpu())

        device = data['data'].device
        total_loss = torch.zeros(1)
        total_loss = total_loss.to(device)

        for loss_function in self.Losses:
            loss = loss_function(network_outputs, targets)
            total_loss += loss

            loss_name = loss_function.Name()

        self.val_metrics['loss'].update(total_loss.detach().cpu())

    def validation_epoch_end(self, outputs) -> None:
        """
        Log the validation metrics at the end of each validation epoch
        Args:
            outputs:

        Returns:

        """
        self._log_epoch_metrics(self.val_metrics, prefix='val/')

    def training_epoch_end(self, outputs):
        """
        Log the epoch metrics at the end of each epoch
        Args:
            outputs:

        Returns:

        """

        self._log_epoch_metrics(self.train_metrics, prefix='train/')

    def _log_epoch_metrics(self, metrics, prefix='train/'):
        logs = {}
        for key, metric in metrics.items():
            logs["{}{}".format(prefix, key)] = metric.compute()
            metric.reset()

        self.log_dict(logs, on_epoch=True, logger=True)

    def export(self, exporters: List[BASE_Exporter], sample_input_data: torch.Tensor,
               validation_data: torch.Tensor) -> None:
        """
        Use the specified exporters to export the Module's Network(s)

        Args:
            exporters: List of Exporter Components
            sample_input_data: Sample data item used by the exporters to build the Network graphs for export
            validation_data: A different piece of data to use for validating that the network exported correctly
        Returns:
            None
        """

        # Export the Module's Network using each of the specified Exporters
        for exporter in exporters:
            exporter.export(save_filename=self.Network.Name(),
                            network=self.Network,
                            sample_input_data=sample_input_data,
                            validation_data=validation_data)
