import os
from typing import cast, Optional, List

from dneg_ml_toolkit.src.AppConfigs.Train_config import TrainConfig
from dneg_ml_toolkit.src.AppConfigs.Export_config import ExportConfig
from dneg_ml_toolkit.src.Component.component_store import ComponentStore
from dneg_ml_toolkit.src.Data.DataModules.DataModule import DataModule
from dneg_ml_toolkit.src.globals import Globals
from dneg_ml_toolkit.src.checkpoints import checkpoint_utils
from dneg_ml_toolkit.src.utils.logger import Logger
from dneg_ml_toolkit.src.TrainModules.BASE_TrainModule import BASE_TrainModule
from dneg_ml_toolkit.src.Exporters.BASE_Exporter import BASE_Exporter


def export(export_config: ExportConfig, resume_checkpoint: Optional[str] = None) -> None:
    """

    Args:
        export_config: Export configuration
        resume_checkpoint: Name of the checkpoint in the experiment's Checkpoint folder to load. If not specified,
            will load the latest checkpoint.

    Returns:
        None
    """

    Logger().Log("--------------------Building Data Module----------")
    # The ExportConfig's Data Module configuration must have a Train Dataloader defined.
    data_module = cast(DataModule, ComponentStore().build_component_from_config(export_config.DataModule))

    assert data_module.train_dataloader() is not None, "The Data Module has no Train Dataloader. A Train Dataloader " \
                                                       "must be configured to load sample data for exporting. "

    Logger().Log("--------------------Data Module Built----------")

    checkpoint_folder = os.path.join(export_config.Experiment_Folder, Globals().CHECKPOINTS_FOLDER)

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

    Logger().Log("--------------------Building Train Module----------")
    # Resolve the train_module config to the corresponding Component class registered in the component store.
    # Use the TrainModule config saved with the checkpoint to create the Module
    train_module = cast(BASE_TrainModule,
                        ComponentStore().build_component_from_config(checkpoint_configuration.TrainModule,
                                                                     experiment_name=export_config.Name,
                                                                     experiment_folder=export_config.Experiment_Folder
                                                                     ))
    Logger().Log("--------------------Train Module Built----------")

    train_module = train_module.load_from_checkpoint(checkpoint_path=resume_checkpoint,
                                                     config=checkpoint_configuration.TrainModule,
                                                     experiment_name=export_config.Name,
                                                     experiment_folder=export_config.Experiment_Folder)

    # Build the exporters
    exporters: List[BASE_Exporter] = []
    for exporter_config in export_config.Exporters:
        exporter = cast(BASE_Exporter, ComponentStore().build_component_from_config(exporter_config,
                                                                                    experiment_name=export_config.Name,
                                                                                    experiment_run_folder=export_config.Experiment_Folder))
        exporters.append(exporter)

    # Get a single piece of data from the Dataloader
    dataloader_iterator = iter(data_module.train_dataloader())
    data, metadata = next(dataloader_iterator)
    input_sample = data["data"]

    data, metadata = next(dataloader_iterator)
    validation_sample = data["data"]

    Logger().Log("--------------------Starting Exporting----------")
    train_module.export(exporters=exporters, sample_input_data=input_sample, validation_data=validation_sample)
    Logger().Log("--------------------Exporting Complete----------")
