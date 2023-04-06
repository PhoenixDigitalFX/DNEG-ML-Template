import os
from typing import cast, Optional

from dneg_ml_toolkit.src.AppConfigs.Train_config import TrainConfig
from dneg_ml_toolkit.src.AppConfigs.Test_config import TestConfig
from dneg_ml_toolkit.src.Component.component_store import ComponentStore
from dneg_ml_toolkit.src.Data.DataModules.DataModule.DataModule_component import DataModule
from dneg_ml_toolkit.src.utils import device_utils
from dneg_ml_toolkit.src.globals import Globals
from dneg_ml_toolkit.src.checkpoints import checkpoint_utils
from dneg_ml_toolkit.src.utils.logger import Logger
from dneg_ml_toolkit.src.TrainModules.BASE_TrainModule.BASE_TrainModule_config import ExecutionModeEnum

from pytorch_lightning import Trainer, LightningModule


def run_testing(testing_config: TestConfig, resume_checkpoint: Optional[str] = None) -> None:
    """
    Run testing on a trained model. This is configured with a Test Config, which specifies a
    Test Dataloader for loading test data. It loads a checkpoint and initializes the model
    using the configuration saved with the checkpoint

    Args:
        testing_config: Testing configuration
        resume_checkpoint: Name of the checkpoint in the experiment's Checkpoint folder to load. If not specified,
            will load the latest checkpoint.

    Returns:
        None
    """

    Logger().Log("--------------------Building Data Module----------")
    # TestConfig just defines a Data Module to use for testing. Its Data Module configuration must have a
    # Test Dataloader defined.
    # Load the data module with the test configuration
    data_module = cast(DataModule, ComponentStore().build_component_from_config(testing_config.DataModule))

    assert data_module.test_dataloader() is not None, "The Data Module has no Test Dataloader. A Test Dataloader " \
                                                      "must be configured to run testing. "

    Logger().Log("--------------------Data Module Built----------")

    checkpoint_folder = os.path.join(testing_config.Experiment_Folder, Globals().CHECKPOINTS_FOLDER)

    if resume_checkpoint is None:
        # Load the latest checkpoint
        resume_checkpoint = checkpoint_utils.get_latest_checkpoint(checkpoint_folder)
    else:
        # Ensure the specified checkpoint is a valid file in the experiment's checkpoint folder
        resume_checkpoint = os.path.join(checkpoint_folder, resume_checkpoint)
        assert os.path.isfile(resume_checkpoint), "Cannot load checkpoint. {} does not exist".format(resume_checkpoint)

    # The configuration used for training is saved alongside every checkpoint. Load this configuration and initialize
    # the model using it
    checkpoint_configuration: TrainConfig = cast(TrainConfig,
                                                 checkpoint_utils.load_checkpoint_configuration(resume_checkpoint))

    # Ensure the Train Module is in Test mode
    checkpoint_configuration.TrainModule.Mode = ExecutionModeEnum.TEST

    Logger().Log("--------------------Building Train Module----------")
    # Resolve the train_module config to the corresponding Component class registered in the component store.
    # Use the TrainModule config saved with the checkpoint to create the Module
    train_module = cast(LightningModule,
                        ComponentStore().build_component_from_config(checkpoint_configuration.TrainModule,
                                                                     experiment_name=testing_config.Name,
                                                                     experiment_folder=testing_config.Experiment_Folder
                                                                     ))
    Logger().Log("--------------------Train Module Built----------")
    device_config = device_utils.get_lightning_device_configuration(device=testing_config.Device)

    Logger().Log("--------------------Building Lightning Trainer----------")

    trainer = Trainer(**device_config)
    Logger().Log("--------------------Lightning Trainer Built----------")

    Logger().Log("--------------------Starting Testing----------")
    trainer.test(train_module, data_module, ckpt_path=resume_checkpoint)
    Logger().Log("--------------------Testing Complete----------")
