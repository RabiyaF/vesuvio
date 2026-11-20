"""Package defining entry points."""

import argparse
import runpy
from os import path
from pathlib import Path

from mvesuvio.util import handle_config
from mvesuvio.util.files_manager import FilesManager


def main(manual_args=None):
    args = manual_args if manual_args else _setup_and_parse_args()
    if args.command == "version":
        _print_version()

    if args.command == "config":
        _setup_config(args)

    if args.command == "run":
        _setup_config(None)
        _run_analysis(args)

    if args.command == "bootstrap":
        _setup_config(None)
        _run_bootstrap(args)


def _setup_and_parse_args():
    parser = _set_up_parser()
    args = parser.parse_args()
    return args


def _set_up_parser():
    parser = argparse.ArgumentParser(description="Package to analyse Vesuvio instrument data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version", help="Display the version of mvesuvio")
    config_parser = subparsers.add_parser("config", help="Set mvesuvio configuration")
    config_parser.add_argument("--experiment-dir", "-e", help="Set the experiment directory", default="", type=str)
    config_parser.add_argument(
        "--ip-dir",
        "-i",
        help="Set the instrument parameters directory",
        default="",
        type=str,
    )

    subparsers.add_parser("run", help="Run mvesuvio analysis")
    subparsers.add_parser("bootstrap", help="Run bootstrap of vesuvio analysis (without y-space fitting)")
    return parser


def _setup_config(args):
    _set_logging_properties()

    handle_config.refresh_config_dir_and_contents()

    if not handle_config.is_cache_set():
        handle_config.set_default_config_vars()

    inputs = handle_config.read_cached_var("caching.inputs")
    ipfolder_dir = handle_config.read_cached_var("caching.ipfolder")

    if args and args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
        if experiment_dir.is_dir():
            inputs = str(experiment_dir.absolute())
        else:
            print(f"\nError setting directory: {args.experiment_dir}\nUsing default.")

    if args and args.ip_dir:
        ip_dir = Path(args.ip_dir)
        if ip_dir.is_dir():
            ipfolder_dir = str(ip_dir.absolute())
        else:
            print(f"\nError setting directory: {args.ip_dir}\nUsing default.")

    handle_config.set_config_vars(
        {
            "caching.inputs": inputs,
            "caching.ipfolder": ipfolder_dir,
        }
    )


def _set_logging_properties():
    from mantid.kernel import ConfigService

    # Good to have
    ConfigService.setString("default.facility", "ISIS")
    ConfigService.setString("default.instrument", "Vesuvio")
    ConfigService.setString("datasearch.searcharchive", "On")

    # Logging
    ConfigService.setString("logging.loggers.root.channel.class", "SplitterChannel")
    ConfigService.setString("logging.loggers.root.channel.channel1", "consoleChannel")
    ConfigService.setString("logging.loggers.root.channel.channel2", "fileChannel")
    ConfigService.setString("logging.channels.fileChannel.path", str(FilesManager.get_mantid_log_file()))
    ConfigService.setString("logging.channels.fileChannel.formatter.class", "PatternFormatter")
    ConfigService.setString("logging.channels.fileChannel.formatter.pattern", "%Y-%m-%d %H:%M:%S,%i [%I] %p %s - %t")
    ConfigService.setString("logging.channels.fileChannel.rotateOnOpen", "true")
    ConfigService.setString("logging.channels.fileChannel.purgeCount", "1")
    ConfigService.setString("logging.channels.fileChannel.class", "FileChannel")
    ConfigService.setString("logging.channels.consoleChannel.class", "ConsoleChannel")
    # Set properties on Mantid.user.properties not working due to Mantid bug
    # Need to set properties on file in Mantid installation
    mantid_properties_file = path.join(ConfigService.getPropertiesDir(), "Mantid.properties")
    ConfigService.saveConfig(mantid_properties_file)


def _run_analysis(args):
    config_dir = Path(__file__).resolve().parent.parent / "default_config"
    runpy.run_path(str(config_dir / "run_reduction.py"), run_name="__main__")
    runpy.run_path(str(config_dir / "run_fitting.py"), run_name="__main__")


def _run_bootstrap(args):
    config_dir = Path(__file__).resolve().parent.parent / "default_config"
    runpy.run_path(str(config_dir / "run_bootstrap.py"), run_name="__main__")


def _print_version():
    from mvesuvio import version

    print(version())


if __name__ == "__main__":
    main()
