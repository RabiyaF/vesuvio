import runpy
from pathlib import Path

import matplotlib.pyplot as plt
from mantid.api import AnalysisDataService
from mantid.kernel import logger

from mvesuvio.util import bootstrap_helpers
from mvesuvio.util.files_manager import FilesManager

# WARNING: Not ready to be used yet, contains basic functionality but lots of details still left to be sorted out.
# How to use this script:
# 1. Place your bootstrap workspaces under FilesManager.get_bootstrap_inputs_dir().
# 2. Inside that directory, provide a parent folder containing two subdirectories: one for backward files called "backward" and one for forward files called "forward".
# 3. Fill the "backward" and "forward" subdirectories with bootstrap workspaces, each workspace needs to end with a sample index!
# 3. Run this script from inside Manitd editor; it will load the matching workspaces for sample 1, 2, 3, ... by looking for filenames that end with the sample index.
# 4. The reduction step is then run once per sample using those loaded workspaces and writing results under a sibling "*_outputs" folder.
# This is intended as a bootstrap workflow for sample-by-sample reduction, not a fully polished user-facing CLI.

RUN_REDUCTION_PATH = Path(__file__).with_name("run_reduction.py")
RUN_FITTING_PATH = Path(__file__).with_name("run_fitting.py")


def _run_reduction_with_injected_workspaces(back_ws_path: str = "", front_ws_path: str = ""):
    reduction_namespace = runpy.run_path(str(RUN_REDUCTION_PATH), run_name="run_bootstrap_reduction")
    reduction_namespace["BackwardAnalysisInputs"].overwrite_analysis_input_workspace = back_ws_path
    reduction_namespace["BackwardAnalysisInputs"].minimal_output = True
    reduction_namespace["ForwardAnalysisInputs"].overwrite_analysis_input_workspace = front_ws_path
    reduction_namespace["ForwardAnalysisInputs"].minimal_output = True
    reduction_namespace["run_reduction"](configure_with_main=False)


def _run_fitting_with_injected_workspaces(back_ws_path: str = "", front_ws_path: str = ""):
    fitting_namespace = runpy.run_path(str(RUN_FITTING_PATH), run_name="run_bootstrap_fitting")
    fitting_namespace["BackwardFittingInputs"].overwrite_analysis_input_workspace = back_ws_path
    fitting_namespace["BackwardFittingInputs"].minimal_output = True
    fitting_namespace["ForwardFittingInputs"].overwrite_analysis_input_workspace = front_ws_path
    fitting_namespace["ForwardFittingInputs"].minimal_output = True
    fitting_namespace["run_fitting"]()


def _find_workspace_path_for_sample_index(directory: Path, sample_index: int) -> Path | None:
    sample_suffix = str(sample_index)
    return next(
        (
            path_obj
            for path_obj in sorted(directory.iterdir(), key=bootstrap_helpers.get_bootstrap_sample_sort_key)
            if path_obj.stem.endswith(sample_suffix)
        ),
        None,
    )


def run_bootstrap():
    bootstrap_inputs_directory = FilesManager.get_bootstrap_inputs_dir()
    input_dirs = bootstrap_helpers.get_bootstrap_input_directories(bootstrap_inputs_directory)
    if not input_dirs:
        return
    _inputs_parent_path, inputs_backward_path, inputs_forward_path = input_dirs

    bootstrap_outputs_directory = FilesManager.get_bootstrap_outputs_dir()

    sample_index = 1
    while True:
        back_ws_path = _find_workspace_path_for_sample_index(inputs_backward_path, sample_index)
        front_ws_path = _find_workspace_path_for_sample_index(inputs_forward_path, sample_index)

        if back_ws_path is None or front_ws_path is None:
            logger.warning(f"Could not find bootstrap workspaces ending in sample index {sample_index}. Stoping bootstrap procedure.")
            break

        # TODO: Replace "boot_" with sample name
        FilesManager.set_experiment_dir(bootstrap_outputs_directory / ("boot_" + str(sample_index)))
        AnalysisDataService.clear()
        _run_reduction_with_injected_workspaces(str(back_ws_path), str(front_ws_path))
        _run_fitting_with_injected_workspaces(str(back_ws_path), str(front_ws_path))
        plt.close("all")
        sample_index += 1
    return


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    run_bootstrap()
