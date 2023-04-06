# DNEG-ML-Template
Template for projects using DNEG ML Toolkit 

## Installation
DNEG-ML-Template extends the DNEG-ML-Toolkit, following its structure and extending its functionality.
DNEG-ML-Toolkit currently needs to be run from source, which can be found at:
https://github.com/dneg/DNEG-ML-Toolkit/tree/v1.1. Follow the instructions there to set up the environment and install
the Toolkit.

Once this is done, install the following in the same conda environment to run DNEG-ML-Template:
1. Navigate to the root folder of DNEG-ML-Template and run *pip install -r requirements.txt*
2. Install the correct version of PyTorch 1.12 from https://pytorch.org/get-started/previous-versions/#v1121, using the version with the correct cudatoolkit (i.e. 11.3)

## Usage

### Make Experiment
Make Experiment is a helper tool to speed up creating a new experiment. It allows you to choose a configuration template
to start from, and builds the experiment's folder using this template.

Example usage: 

*python run_experiment.py make-experiment --name {experiment_name} --template Simple*

- *name* Name of the experiment to create.
- *template* is the name of a json configuration file found in the config_templates folder. The tool's command line help will list all available templates.
- Once the experiment has been created, edit the experiment's json configuration file to provision the system and set the hyperparameters for the experiment.

### Make Run
Experiments are divided up into runs. A folder for a new run can be created in the experiment's folder (with the naming convention run_#),
or this tool can be used to automatically create the sub-folder with the correctly incremented run number, and a copy of the experiment's
base JSON configuration.

*python run_experiment.py make-run --name {experiment_name}*

- *name* Name of an existing experiment to create a new run for.

### Run Experiment
Run Experiment runs the training of the ML system, using the json configuration for the specified configuration. The experiment
folder and json configuration can be created manually, or you can use the Make Experiment tool to assist with this.

Example usage: 

*python run_experiment.py train --experiment {experiment_name} --run 0 --device 1*

- *experiment* is the name of the experiment within the experiments folder
- *run* Number of the experiment run to train.
- *device* allows the experiment to be run on gpu. 1 will run on 1 gpu, [0] will run on gpu with index 0, cpu will run the training on cpu.
- *resume* Attempt to resume the training from the latest checkpoint. If no checkpoint is found, will start from the beginning.
- *resume_from_checkpoint* Path to a specific checkpoint to resume the training from. Cannot be use if --resume is enabled.

### Run Testing
To run testing on a trained model, configure a test configuration with the name {experiment_name}_test.json in the experiment's run folder. See Example_test.json config template for reference.
run_testing will load a checkpoint for the experiment and run inference on the testing data.

Example usage: 

*python run_testing.py --experiment {experiment_name} --run 0 --device 1*

- *experiment* is the name of the experiment within the experiments folder
- *run* Number of the experiment run to test.
- *device* allows the testing to be run on gpu. 1 will run on 1 gpu, [0] will run on gpu with index 0, cpu will run the training on cpu.
- *checkpoint* Name of the checkpoint in the experiment's Checkpoint folder to load. If not specified, will load the latest checkpoint.

### Exporting Networks
The DNEG ML Toolkit supports exporting trained networks to common formats, such as ONNX and TorchScript. To export a 
trained model, configure an export configuration with the name {experiment_name}_export.json in the experiment's run folder. 
See Example_export.json config template for reference.

*python run_export.py --experiment {experiment_name} --run 0*

- *experiment* is the name of the experiment within the experiments folder
- *run* Number of the experiment run to export.
- *checkpoint* Name of the checkpoint in the experiment's Checkpoint folder to load. If not specified, will load the latest checkpoint.


### Running Tensorboard
The system generates Tensorboard reports as it is running. To view these reports, run tensorboard from commandline using:

*tensorboard --logdir {Experiment Dir}*

- The *logdir* is the experiment's folder. This will allow reports from all experiment runs to be viewed.
- With tensorboard running, follow its instructions to connect to it from a web browser.

### Make Component Tool
The Make Component Tool aims to speed up development by creating the scaffolding for new Components. See the Development
section below for details on ML Toolkit's Components - this tool essentially automates the implementation guidelines
described there.

*python run_make_component.py --name EXAMPLETransform --target_folder Data/Transforms --parent_component BASE_Transform*

- *name* is the name of the new Component to create.
- *target_folder* is the folder relative to the project's src folder to create the Component in.
- *parent_component* Name of the Component to inherit from. E.g. if creating a new Network, inherit from Base_Network. If not provided, will inherit from ML Toolkit's base Component."
- *is_base_component* Optional flag used when creating a new base Component type.




## Tutorial
The DNEG ML Template project is an example project that shows how to build ML systems using DNEG ML Toolkit. It uses image classification
as the ML task, and includes a variety of Components and options to showcase the ML Toolkit's features.

### Configuration
The Simple.json configuration template gives an example of how to quickly start training experiments using DNEG ML Toolkit.
- All Train configurations require a TrainModule and DataModule.
- TrainModule configuration:
  - The TrainModule has been configured to use the StandardTrainModule from the core ML Toolkit, so no development work is necessary.
  - The StandardTrainModule requires a Network, Loss, Optimizer (and optional LR Scheduler). It performs a forward pass on the Network, calculates loss from the outputs, and optimizes.
  - The Network is configured to use SimpleCNN, which is defined in DNEG ML Template. This showcases how a project's Components can be used alongside core Toolkit Components.
  - The Loss and Optimizer are configured to use CrossEntropyLoss and SGD, both from the core Toolkit.
- DataModule configuration:
  - The Data Module uses the standard DataModule from the core Toolkit.
  - The DataModule is configured with a TrainDataloader, which uses the core Toolkit's Dataloader Component.
  - The TrainDataloader is configured to load data from the FashionMNIST Dataset, which is defined in DNEG ML Template. 
In this example, only the SimpleCNN Network and FashionMNIST Dataset were developed for the DNEG ML Template project, all other components are generic core Toolkit Components.

The Extend.json template shows how modifying the configuration of Simple.json can give further control over how the experiment is provisioned.  
- The TrainModule has been switched to ClassificationTrainModule, which is a TrainModule Component defined in DNEG ML Template. 
This TrainModule is similar to the StandardTrainModule, but implements functionality specific for image classification, 
including configuring the Network's number of outputs from the classes read from the dataset, running validation during training,
and defining a testing forward pass. 
- Note that the Network no longer needs NumOutputs to be configured, since this is set automatically by the TrainModule.
- The Network has been changed to ExtendedSimpleCNN, which exposes fields that allow batch normalization and the activation
function of the convolution layers to be configured.
- The DataModule has had its ValDataloader configured, so that validation can be run during training.

The TransformTutorial.json provides an example of how the data pipeline allows data to be augmented as it is loaded.
- Each item of data that the CIFAR10 Dataset loads and prepares (this includes the image and target class in this example, 
but could contain more complex data) is packaged into a data dictionary and passed through all Transforms configured for the Dataloader.
- The Mirror Transform from the core Toolkit is configured to apply to "data" (the Toolkit's standard is for a Dataset 
to name the input to the model "data"). For CIFAR10, "data" is the image. Mirror will horizontally flip the image, with 
a probability of 0.4 of doing the flip.
- Next, the data dictionary is passed into the first ExampleGrayscale Transform.
- ExampleGrayscale is defined in DNEG ML Template and showcases the features available for performing data augmentations.
- It is configured with InPlace: false, meaning that the Transform does not override the input data in the data dictionary. 
Instead, it will create a new entry, using the configured OutputSuffix to create a new item named "data_grayscale". Note that
if a Transform applies to multiple items of data, the OutputSuffix will be appended to each one to create new data entries.
- Since SplitChannels is true, this Transform will also generate additional data, returning each colour channel as a 
separate data entry: "red_channel", "blue_channel", "green_channel", and add them to the data dictionary. 
Any Transform further down the stack will have access to these items, as will the training Components that the TrainModule
passes the data to (i.e. the Network, Loss).
- To showcase the flexibility of the Transform system, a second ExampleGrayscale is configured.
- Since the previous ExampleGrayscale did not override "data" in-place, the original rgb image is still available. Additionally,
this Transform is configured to apply to "data_grayscale" created by the previous Transform.
- InPlace is still set to false, but this time OutputNames is used instead of OutputSuffix. This allows the full output name for each
item of input data to be separately defined.
- SplitChannels is set to false so the additional channel data is not returned. This is done to show that the data passed through the Transform
system must all have unique identifiers. Setting SplitChannels to true would generate a second set of "red_channel", "blue_channel", "green_channel"
entries, but since the previous Transform added items with those names to the data dictionary it would cause a conflict.
- Finally, all the data entries created by the example Transforms are passed into the Toolkit's ToTensor Transform.
- Note that the CIFAR10 Dataset automatically adds a ToTensor Transform that applies to "data" to the end of this list, 
as CIFAR10 data is loaded as PIL images and the Network requires it to be a tensor.

### Development
The DNEG ML Template project is documented and commented throughout to guide the development of projects using DNEG ML 
Toolkit. Some general points on development:
- DNEG ML Toolkit implements a Component-based environment. It defines generic base Components for most ML system objects
  (Network, Loss, Optimizer, Dataset, etc.), which can be inherited from to create specific Components. Any object 
structured as a Component is exposed to the ML Toolkit's configuration system.
- The project should follow the same folder structure as the core Toolkit for types of Components. ML Template includes most 
of the Component types available in the Toolkit, but not all.
- To add a new Component for an existing type, create a new {ComponentName} sub-folder in the Component type's folder. This sub-folder
requires 2 python files: {ComponentName}_config.py and {ComponentName}_component.py.
- The class implemented in {ComponentName}_config.py is called {ComponentName}Config and inherits from the base component
config for the Component type (i.e. a new Network's config class would inherit from BASE_NetworkConfig).
- The ComponentConfig class defines the fields that are exposed to the JSON configuration system, and is used to build
the corresponding Component object when the system is being initialized from config.
- The class implemented in {ComponentName}_component.py is called {ComponentName} and inherits from the base component 
for the Component type (i.e. a new Network's class would inherit from BASE_Network).
- The Component always takes its corresponding ComponentConfig as input to its constructor.
- It is possible for a Component to be a sub-class of another Component, just inherit from the parent Component's
object and Config class instead of the base class.
- It is also possible to define new base Component types. This requires the component_group function to be defined in
the base Component's Config, as this tells the configuration system to group all the Components that inherit from this
type together.

#### Git Repositories
The DNEG ML Toolkit standard for managing Git repositories is to use a Main->Develop->Feature branching strategy.
- All commits on the Main branch should be stable, releasable systems, tagged with a version number. This will be used
to manage releasing models as part of ML solutions.
- The Develop branch is for current development work, and allows multiple researchers collaborate on the same system. Each
researcher will create a Feature branch for the current feature they are working on, and merge it back to Develop when it is complete.
This ensures that Develop remains stable while each new feature is being researched and developed.
- Merging to Develop should be done with a Pull Request reviewed by at least one other project member. Merges to Main should
be reviewed by all project members.
