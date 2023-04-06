from typing import Optional, Union
import click
import pathlib

from dneg_ml_toolkit.src.globals import Globals
from dneg_ml_toolkit import run_experiment_utils
from dneg_ml_toolkit.src.register_components import register_toolkit_components
from src.register_components import register_project_components
from src.export import export

current_folder = pathlib.Path(__file__).parent.resolve()


@click.command()
@click.option('--experiment', help='Name of the experiment to run testing on.', required=True)
@click.option('--run', help='Number of the experiment run to test.', required=True)
@click.option('--checkpoint', help="Name of the checkpoint in the experiment's Checkpoint folder to load."
                                   "If not specified, will load the latest checkpoint.")
def run_test(experiment: str, run: Union[str, int], checkpoint: Optional[str] = None) -> None:
    # 1. Register the core toolkit components, then the components for the template project
    register_toolkit_components()
    register_project_components()

    # 2. The export config must be built after all the components have been registered,
    # so that it has access to the registered components
    config = run_experiment_utils.build_experiment_config(project_root_folder=current_folder,
                                                          experiment=experiment, run=run, device=None,
                                                          config_file_suffix=Globals().EXPORT_CONFIG_SUFFIX)

    export(config, resume_checkpoint=checkpoint)


if __name__ == "__main__":
    run_test()
