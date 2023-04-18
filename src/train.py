import os
from typing import cast, Optional

from dneg_ml_toolkit.src.AppConfigs.Train_config import TrainConfig
from dneg_ml_toolkit.src.Component.component_store import ComponentStore
from dneg_ml_toolkit.src.Data.DataModules.DataModule import DataModule
from dneg_ml_toolkit.src.utils import device_utils
from dneg_ml_toolkit.src.globals import Globals
from dneg_ml_toolkit.src.checkpoints import checkpoint_utils
from dneg_ml_toolkit.src.utils.logger import Logger
from dneg_ml_toolkit.src.Data.Dataloaders.Dataloader import Dataloader

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


def train(training_config: TrainConfig, resume: bool, resume_checkpoint: Optional[str] = None) -> None:
    """
    Run the Pytorch Lightning Training for the project using the provided configuration object
    Args:
        training_config: A config object created by loading the configuration file for the experiment
        resume: Flag, if true will attempt to load the latest saved checkpoint for the experiment
        resume_checkpoint: Path to a specific checkpoint to resume from. Ignored if resume is set to True.
    Returns:
        None
    """

    # 1. Build the Lightning DataModule from the configuration
    Logger().Log("--------------------Building Data Module----------")
    data_module = cast(DataModule, ComponentStore().build_component_from_config(training_config.DataModule))
    Logger().Log("--------------------Data Module Built----------")

    # 2. Build the Lightning Training Module from the configuration
    Logger().Log("--------------------Building Train Module----------")

    # 2.1 Initialize private Train Module fields by generating the values from the Data Module
    if hasattr(training_config.TrainModule, "_InputShape"):
        image_shape = data_module.train_dataloader().get_shared_dataset_property(property_name="data_shape")
        training_config.TrainModule._InputShape = image_shape

    if hasattr(training_config.TrainModule, "_NumClasses"):
        num_classes = len(cast(Dataloader, data_module.train_dataloader()).get_shared_dataset_property(
            property_name="classes"))
        training_config.TrainModule._NumClasses = num_classes

    # 3. Resolve the train_module config to the corresponding Component class registered in the component store
    train_module = cast(LightningModule,
                        ComponentStore().build_component_from_config(training_config.TrainModule,
                                                                     experiment_name=training_config.Name,
                                                                     experiment_folder=training_config.Experiment_Folder))
    Logger().Log("--------------------Train Module Built----------")

    # 4. Configure the device, lr monitor, checkpointer, logger
    device_config = device_utils.get_lightning_device_configuration(device=training_config.Device)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_folder = os.path.join(training_config.Experiment_Folder, Globals().CHECKPOINTS_FOLDER)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_folder,
        every_n_epochs=1,
        save_top_k=-1,  # Save all checkpoints
        filename="{}".format(training_config.Name) + "_{epoch:03d}",
        save_on_train_epoch_end=True  # Separate checkpoint saving from the validation step, since if this is false it
        # only performs checkpointing as part of the validation step
    )

    report_folder = os.path.join(training_config.Experiment_Folder, Globals().REPORTS_FOLDER)
    tensorboard_logger = TensorBoardLogger(save_dir=report_folder, name=training_config.Name)

    if resume:
        resume_checkpoint = checkpoint_utils.get_latest_checkpoint(checkpoint_folder)

    Logger().Log("--------------------Building Lightning Trainer----------")
    # 5. Build the Trainer
    trainer = Trainer(max_epochs=training_config.Epochs,
                      callbacks=[checkpoint_callback, lr_monitor],
                      resume_from_checkpoint=resume_checkpoint,
                      logger=tensorboard_logger,
                      log_every_n_steps=training_config.LogInterval,
                      check_val_every_n_epoch=1, # Run validation every epoch
                      limit_val_batches=1,  # Run validation on a single batch
                      num_sanity_val_steps=0,  # Don't run sanity checking
                      **device_config)
    Logger().Log("--------------------Lightning Trainer Built----------")

    Logger().Log("--------------------Starting Training----------")
    # 6. Run the training
    trainer.fit(train_module, data_module)
    Logger().Log("--------------------Training Complete----------")
