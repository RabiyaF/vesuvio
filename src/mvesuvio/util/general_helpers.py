import re
from pathlib import Path
from typing import Any

import numpy as np
from mantid.kernel import logger
from mantid.simpleapi import AnalysisDataService, Load, SumSpectra

from mvesuvio.globals import Mode, ModeRunning, ScatteringType, Tags
from mvesuvio.util.files_manager import FilesManager


def _get_scattering_type(inputs_class: Any) -> ScatteringType:
    """Infer scattering type from user inputs."""
    name = str(getattr(inputs_class, "name", "")).lower()
    class_name = getattr(inputs_class, "__name__", "").lower()
    mode = str(getattr(inputs_class, "mode", "")).lower().replace(" ", "")

    if (
        name.startswith(Tags.BACKWARD.value.lower())
        or ModeRunning.BACKWARD.value.lower() in class_name
        or mode == Mode.DOUBLE_DIFFERENCE.value.lower()
    ):
        return ScatteringType.BACKWARD

    if (
        name.startswith(Tags.FORWARD.value.lower())
        or ModeRunning.FORWARD.value.lower() in class_name
        or mode == Mode.SINGLE_DIFFERENCE.value.lower()
    ):
        return ScatteringType.FORWARD

    raise ValueError(f"Could not infer scattering type for input class: {inputs_class}")


def pass_data_into_ws(dataX, dataY, dataE, ws):
    "Modifies ws data to input data"
    dataX = np.atleast_2d(dataX)
    dataY = np.atleast_2d(dataY)
    dataE = np.atleast_2d(dataE)
    for i in range(ws.getNumberHistograms()):
        ws.dataX(i)[:] = dataX[i, :]
        ws.dataY(i)[:] = dataY[i, :]
        ws.dataE(i)[:] = dataE[i, :]
    return ws


def extractWS(ws):
    """Directly extracts data from a workspace into arrays."""
    return ws.extractX(), ws.extractY(), ws.extractE()


def print_table_workspace(table, precision=3):
    table_dict = table.toDict()
    # Convert floats into strings
    for key, values in table_dict.items():
        new_column = [int(item) if (isinstance(item, float) and item.is_integer()) else item for item in values]
        table_dict[key] = [f"{item:.{precision}f}" if isinstance(item, float) else str(item) for item in new_column]

    max_spacing = [max([len(item) for item in values] + [len(key)]) for key, values in table_dict.items()]
    header = "|" + "|".join(f"{item}{' ' * (spacing - len(item))}" for item, spacing in zip(table_dict.keys(), max_spacing)) + "|"
    logger.notice(f"Table {table.name()}:")
    logger.notice(" " + "-" * (len(header) - 2) + " ")
    logger.notice(header)
    for i in range(table.rowCount()):
        table_row = "|".join(
            f"{values[i]}{' ' * (spacing - len(str(values[i])))}" for values, spacing in zip(table_dict.values(), max_spacing)
        )
        logger.notice("|" + table_row + "|")
    logger.notice(" " + "-" * (len(header) - 2) + " ")
    return


def make_summarised_log_file() -> None:
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}")
    try:
        with open(FilesManager.get_mantid_log_file(), "r") as infile, open(FilesManager.get_summarised_log_file(), "w") as outfile:
            for line in infile:
                if "VesuvioAnalysisRoutine" in line:
                    outfile.write(line)

                if "Notice Python" in line:  # For Fitting notices
                    outfile.write(line)

                if not pattern.match(line):
                    outfile.write(line)
    except OSError:
        logger.error("Mantid log file not available. Unable to produce a summarized log file for this routine.")
    return


def load_overwritten_workspace_if_specified(analysis_inputs: Any) -> str:
    """Load a workspace override from ADS or disk when an analysis input class specifies one."""

    override = str(getattr(analysis_inputs, "overwrite_analysis_input_workspace", "") or "")

    if not override:
        return ""

    if AnalysisDataService.doesExist(override):
        SumSpectra(InputWorkspace=override, OutputWorkspace=override + "_sum")
        return override

    if Path(override).exists():
        override_name = Path(override).stem
        Load(Filename=override, OutputWorkspace=override_name)
        SumSpectra(InputWorkspace=override_name, OutputWorkspace=override_name + "_sum")
        return override_name
    raise ValueError(f"Override workspace not in ADS and not a path: {override}")


def get_workspace_name_if_path(path_or_name: str) -> str:
    """Return the workspace name if the input is a path, otherwise return the input as is."""
    path_obj = Path(path_or_name)
    if path_obj.exists():
        return path_obj.stem
    return path_or_name
