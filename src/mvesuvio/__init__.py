"""
Vesuvio
=============

Vesuvio is an instrument that performs Neuton Compton Scattering, based at ISIS, RAL, UK. This code processes raw output data to determine
nuclear kinetic energies and moment distributions.
"""

from mvesuvio._version import __version__
from mvesuvio.main import main

__project_url__ = "https://github.com/mantidproject/vesuvio"

__all__ = ["__project_url__", "__version__", "config", "run", "set_config", "version"]


class ArgInputs:
    def __init__(self, command):
        self.__command = command

    @property
    def command(self):
        return self.__command


class ConfigArgInputs(ArgInputs):
    def __init__(self, experiment_dir, ip_dir):
        super().__init__("config")
        self.__set_experiment_dir = experiment_dir
        self.__set_ip_dir = ip_dir

    @property
    def experiment_dir(self):
        return self.__set_experiment_dir

    @property
    def ip_dir(self):
        return self.__set_ip_dir


class RunArgInputs(ArgInputs):
    def __init__(self):
        super().__init__("run")


def config(experiment_dir="", ip_dir=""):
    config_args = ConfigArgInputs(experiment_dir, ip_dir)
    main(config_args)


def run():
    run_args = RunArgInputs()
    main(run_args)


def version():
    """Returns the version of the mvesuvio package."""
    return __version__
