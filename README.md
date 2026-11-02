# Mantid VESUVIO

[![Nightly Build Status](https://github.com/mantidproject/vesuvio/actions/workflows/deploy_package_nightly.yml/badge.svg)](https://github.com/mantidproject/vesuvio/actions/workflows/deploy_package_nightly.yml)
[![Coverage Status](https://coveralls.io/repos/github/mantidproject/vesuvio/badge.svg?branch=main)](https://coveralls.io/github/mantidproject/vesuvio?branch=main)
[![Anaconda-Server Badge](https://anaconda.org/mantid/mvesuvio/badges/latest_release_relative_date.svg)](https://anaconda.org/mantid/mvesuvio)
[![Anaconda-Server Badge](https://anaconda.org/mantid/mvesuvio/badges/version.svg)](https://anaconda.org/mantid/mvesuvio)
[![Anaconda-Server Badge](https://anaconda.org/mantid/mvesuvio/badges/downloads.svg)](https://anaconda.org/mantid/mvesuvio)

This repository contains:
- `mvesuvio` package containing Neutron Compton Profile (NCP) analysis procedures for Vesuvio, published nightly.
- Vesuvio calibration scripts, under the `tools` folder

Currently only the NCP analysis is usable, the calibration scripts are not yet ready. 

## Installing mvesuvio package (Try this option first)
The `mvesuvio` package is meant to be used inside the [Mantid software](https://www.mantidproject.org/index.html), so you'll need to install Mantid first if you haven't.

Once you have a working version of Mantid, go to the `IPython` tab on the bottom center of the window and inside the tab type the following command(s):

**If you're on ADA or Linux:**

Run `mamba install mantid/label/nightly::mvesuvio` followed by `!mvesuvio version` in the `IPython` tab:

![linux ipython install](https://raw.githubusercontent.com/mantidproject/vesuvio/main/images/linux_ipython_install.png)

**WARNING: This command alone may not create the .mvesuvio folder in the home directory. This needs to be forced by running the command `!mvesuvio config` in the `IPython` editor. The command will output the configured *experiment directory* and *ip directory*.**
![config output](https://raw.githubusercontent.com/mantidproject/vesuvio/main/images/config_output.png)

**If you're on Windows:**

Run `pip install mvesuvio` followed by `pip show mvesuvio` in the `IPython` tab:

![windows ipython install](https://raw.githubusercontent.com/mantidproject/vesuvio/main/images/windows_ipython_install.png)

To do all the necessary setup, type:

`!mvesuvio config`

If you see some output then the package is successfully installed and setup!

**WARNING: This way of installing the package is not officially supported by Mantid, it just so happens to work. As you can see above, a folder *C:\<User Path>\Lib\site-packages* is created after the installation. However, this folder does not contain the analysis_inputs.py script. This script is contained in the .mvesuvio folder that is usually created by the installation under *C:\Users\<fedid>\.mvesuvio*. If after the pip installation the folder .mvesuvio has not been created under *C:\Users\<fedid>\.mvesuvio* you can try to copy it from another computer on which installation was successful. Try to run analysis_inputs.py from the newly copied .mvesuvio folder and check the inputs and outputs created.**

### Installing mantid and mvesuvio using conda/mamba (If the first option failed)

If the previous installation attempt was unsuccessful, then you'll have to use Mantid inside a *conda/mamba* environment. This is the official recommended way of using the mantid with the `mvesuvio` package.

**If you're on your personal machine (i.e. not on ADA):**

To install `mamba`, follow the steps at:
https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html

**If you're on ADA:**

`mamba` is already installed on ADA.
Open a terminal (Applications → System → Terminal) on **ADA**:

![ada menu](https://raw.githubusercontent.com/mantidproject/vesuvio/main/images/ada_menu.png)

Verify mamba is installed by running:

`mamba --version`

You should see some output with the versions available in your system.

Now create a new *conda* environment in the terminal, for example I'll call it `mantid-mvesuvio`:

`mamba create -n mantid-mvesuvio mantidworkbench mantid/label/nightly::mvesuvio`

![ada mamba environment creation](https://raw.githubusercontent.com/mantidproject/vesuvio/main/images/ada_mamba_create.png)

And activate the environment you created, by typing the following in the terminal:

`mamba activate mantid-mvesuvio`

You can now start Mantid with `mvesuvio` already installed by typing the following in the terminal:

`workbench`

![ada mamba environment activation](https://raw.githubusercontent.com/mantidproject/vesuvio/main/images/ada_mamba_activate.png)

## Updating versions

If you want to check the current version of the package, go to the `IPython` editor in Mantid and type:

`!mvesuvio version`

The easiest way to update the `mvesuvio` package is to uninstall the current package in your environment and install it again.
You'll need to use `pip`, `conda` or `mamba` depending on which one you used to install the package.

So for example if you did `mamba install mvesuvio` then the **uninstall** command is:

`mamba uninstall mvesuvio`

Otherwise if you used pip:

`pip uninstall mvesuvio`

If you do not remember which command you used during the installation, then just run both `pip` and `mamba` uninstall commands, one of them will fail but the other one will succeed.

Once you have uninstalled the package, you can install it again to get the latest version.
If you're looking for a specific version, you can do:

`mamba install mvesuvio=1.0`

Which will install version *1.0* of the package.

### Quickstart (Running your first analysis)

To run your very first analysis (and check that everything is working), go to your home folder and find the folder `mvesuvio`.

If the folder does not exist, go the the `IPython` tab in Mantid and type:

`!mvesuvio config`

You should see some output of default inputs, which will lead you to the `mvesuvio` folder.

Once you have located the `mvesuvio` folder, open Mantid workbench and inside it open the script `mvesuvio/emperiment_template/run_reduction.py`.

This script represents the basics for passing in the inputs of the reduction routine.
Click the run button on the workbench to start the execution of the script.
(Check that you have the archive search enabled, the facility is set to ISIS and the instrument set to VESUVIO, otherwise the Vesuvio runs might not be found).
This script is an example of a well-behaved sample and it should run without issues.

If the run was successful, you will notice that a new folder was created inside `mvesuvio/experiment_template` containing all sorts of outputs for this script.

You can also see that the folder `mvesuvio/experiment_template` also contains another script `run_fitting.py` to run the fitting routine after the reduction. When you run `run_reduction.py`, some of the outptus will be placed inside a folder `fitting_inputs` to be used by `run_fitting.py`.

**IMPORTANT: To run a new sample with different inputs, you should *copy* the example folder `experiment_template` and place it in *any* folder of your choice outside `mvesuvio`. For providing the instrument parameters files, place them inside `mvesuvio/ip_files/`.**
(You can change the directory of the instrument files too, consult next section).

For a more detailed explanation on what the inputs in the `run_reduction.py` mean, read [USERGUIDE.md](https://github.com/mantidproject/vesuvio/blob/main/USERGUIDE.md)

## Tips and useful commands

A very useful command is:

`mvesuvio version`

Which returns the version of `mvesuvio` that you have currently installed.
If you're running a Python script or have access to a Python interpreter (like the `IPython` tab in Mantid), you can do:

```
import mvesuvio
mvesuvio.version()
```
In a Python interpreter like the `IPython` tab in Mantid, you can also run terminal commands by starting the command with `!`:

`!mvesuvio version`

Or to see the current configuration:

`!mvesuvio config`


## Advanced Usage (CLI)
If you're using a *conda* environment and have installed Mantid and `mvesuvio` with *conda/mamba*, then you might be interested in the CLI options of `mvesuvio`.
With your environment activated you can type in the terminal:

`mvesuvio -h`

And this will list all available commands.
Currently, the commands that are stable are `config`, `run` and `version`. All other commands are available but are still in development.

#### mvesuvio config

The `config` command is used to display or set the experiment directory and the folder used to look for the instrument parameters.
You can do so by providing two optional arguments:
- `--experiment-dir` - Set the location of the experiment directory (default is the `experiment_template` directory inside `mvesuvio`).
- `--ip-dir` - Set the directory for the instrument parameter files (default is `ip_files` in `mvesuvio` folder).

If you run `mvesuvio config` with no arguments then the output will tell you the current experiment directory and instrument parameters directory.

Usage examples:
- `mvesuvio config --ip-dir C:\IPFolder` - Set instrument parameters folder.
- `mvesuvio config --experiment-dir C:\Vesuvio\experiment` - Set experiment directory.

#### mvesuvio run

The `run` command does not take any arguments and simply runs the routine based on the current configuration.

Usage example:
- `mvesuvio run`- Run NCP analysis.

This command now triggers the configured reduction script followed by the configured fitting script.

#### mvesuvio bootstrap

The `bootstrap` command does not take any arguments. It runs the bootstrap script using the configured bootstrap script.

Usage example:
- `mvesuvio bootstrap` - Run bootstrap reduction using `BOOTSTRAP_INPUTS_DIRECTORY`.

### Python API

The commands available in the CLI can be triggered from Python by calling the method with the same name.
So for example to set the configuration from a Python script (or from `IPython` tab):

```
import mvesuvio
mvesuvio.config(experiment_dir='C:\Vesuvio\experiment', ip_dir='C:\IPFolder')
```
In fact, this functionality is what you see at the end of the `analysis_inputs.py` file, which sets the experiment directory based on the currently opened script location:

```
import mvesuvio
from pathlib import Path

mvesuvio.config(experiment_dir=str(Path(__file__).parent), ip_dir='')
mvesuvio.run()
```

The Python `mvesuvio.run()` helper mirrors the CLI and no longer accepts workspace/output override arguments.
