from __future__ import annotations

from typing import TYPE_CHECKING

from mantid import AnalysisDataService
import matplotlib.pyplot as plt
from mantid.simpleapi import (
    mtd,
    Load,
    RenameWorkspace,
    SumSpectra,
    CropWorkspace,
    MaskDetectors,
    DeleteWorkspaces,
    CreateEmptyTableWorkspace,
    DeleteWorkspace,
    SaveNexus,
    LoadVesuvio,
    Divide,
    Rebin,
    ConvertToYSpace,
    Integration,
    CloneWorkspace,
    WeightedMean,
    AppendSpectra,
    SaveAscii,
)
from mantid.kernel import logger
import numpy as np
import math
import re
from pathlib import Path

from mvesuvio.globals import ModeRunning, ScatteringType, Tags, Masking
from mvesuvio.util.files_manager import FilesManager
from mvesuvio.util import general_helpers, handle_config
from mvesuvio.util.general_helpers import pass_data_into_ws, print_table_workspace, extractWS
from mvesuvio.util.constraints_transport import serialize_constraints
from mvesuvio.util import fitting_helpers
from mvesuvio.analysis_reduction import VesuvioAnalysisRoutine
from mantid.api import AlgorithmFactory, AlgorithmManager

if TYPE_CHECKING:
    from mvesuvio.default_config.experiment_template.run_reduction import BackwardAnalysisInputs, ForwardAnalysisInputs

import ntpath

try:
    plt.style.use(["ggplot", handle_config.get_plots_config_file()])
except OSError:
    pass


def run_estimate_h_ratio(back_alg, front_alg, back_masses, back_chosen_mass_index, rtol, max_iter):
    """
    Used when H is present and H to first mass ratio is not known.
    Preliminary forward scattering is run to get rough estimate of H to first mass ratio.
    Runs iterative procedure with alternating back and forward scattering.
    """
    assert back_alg.getPropertyValue("ModeRunning") == ModeRunning.BACKWARD.value
    assert front_alg.getPropertyValue("ModeRunning") == ModeRunning.FORWARD.value

    # Ask for explicit consent from user to start procedure
    try:
        userInput = input("\nHydrogen intensity ratio to lowest mass is not set. Press Enter to start estimate procedure.")
        if not userInput == "":
            raise EOFError

    except EOFError:
        logger.error("Estimation of Hydrogen intensity ratio interrupted.")
        return

    chosen_mass = back_masses[back_chosen_mass_index]

    table_h_ratios = CreateEmptyTableWorkspace(OutputWorkspace="hydrogen_intensity_ratios_estimates")
    table_h_ratios.addColumn(type="float", name="Hydrogen intensity ratio to chosen mass at each iteration")

    front_alg.execute()

    means_table = mtd[front_alg.getPropertyValue("OutputMeansTable")]
    current_ratio = calculate_h_ratio(means_table, chosen_mass)

    table_h_ratios.addRow([current_ratio])
    previous_ratio = np.nan

    for _ in range(max_iter):
        if (
            current_ratio is not None
            and previous_ratio is not None
            and math.isclose(current_ratio, previous_ratio, rel_tol=rtol, abs_tol=1e-12)
        ):
            break

        back_alg.setProperty("HRatioToChosenMass", current_ratio)
        execute_joint_algorithms(back_alg, front_alg)

        previous_ratio = current_ratio

        means_table = mtd[front_alg.getPropertyValue("OutputMeansTable")]
        current_ratio = calculate_h_ratio(means_table, chosen_mass)

        table_h_ratios.addRow([current_ratio])

    logger.notice("\nProcedute to estimate Hydrogen ratio finished.\n")
    print_table_workspace(table_h_ratios)
    return table_h_ratios


def calculate_h_ratio(means_table, chosen_mass):
    masses = means_table.column("mass")
    intensities = np.array(means_table.column("mean_intensity"))

    if not np.isclose(min(masses), 1, atol=0.1):  # Hydrogen not present
        return None

    # Hydrogen present, assumes its lowest mass
    return intensities[np.argmin(masses)] / intensities[np.argmax(np.isclose(masses, chosen_mass, atol=0.01))]


def h_ratio_is_zero_when_h_present(back_inputs: type[BackwardAnalysisInputs], front_inputs: type[ForwardAnalysisInputs]):
    if back_inputs.run_this_scattering_type:
        if is_hydrogen_present(front_inputs.masses):
            if back_inputs.intensity_ratio_of_hydrogen_to_chosen_mass == 0:
                return True
        else:
            logger.warning("Ignoring Hydrogen ratio because not detected in masses.")
            back_inputs.intensity_ratio_of_hydrogen_to_chosen_mass = 0
    return False


def is_hydrogen_present(masses) -> bool:
    Hmask = np.abs(np.array(masses) - 1) / 1 < 0.1  # H mass whithin 10% of 1 au

    if ~np.any(Hmask):  # H not present
        return False

    print("\nH mass detected.\n")
    assert len(Hmask) > 1, "When H is only mass present, run independent forward procedure, not joint."
    assert Hmask[0], "H mass needs to be the first mass in masses and initPars."
    assert sum(Hmask) == 1, "More than one mass very close to H were detected."
    return True


def execute_joint_algorithms(back_alg, front_alg):
    assert back_alg.getPropertyValue("ModeRunning") == ModeRunning.BACKWARD.value
    assert front_alg.getPropertyValue("ModeRunning") == ModeRunning.FORWARD.value

    back_alg.execute()

    incoming_means_table = mtd[back_alg.getPropertyValue("OutputMeansTable")]
    h_ratio = back_alg.getProperty("HRatioToChosenMass").value

    assert incoming_means_table is not None, "Means table from backward routine not correctly accessed."
    assert h_ratio is not None, "H ratio from backward routine not correctly accesssed."

    receiving_profiles_table = mtd[front_alg.getPropertyValue("InputProfiles")]

    fixed_profiles_table = fix_profile_parameters(incoming_means_table, receiving_profiles_table, h_ratio)

    # Update original profiles table
    RenameWorkspace(fixed_profiles_table, receiving_profiles_table.name())
    print_table_workspace(mtd[receiving_profiles_table.name()])
    # Even if the name is the same, need to trigger update
    front_alg.setPropertyValue("InputProfiles", receiving_profiles_table.name())

    front_alg.execute()
    return


def fix_profile_parameters(incoming_means_table, receiving_profiles_table, h_ratio):
    means_dict = _convert_table_to_dict(incoming_means_table)
    profiles_dict = _convert_table_to_dict(receiving_profiles_table)

    # Set intensities
    for p in profiles_dict.values():
        if np.isclose(p["mass"], 1, atol=0.1):  # Hydrogen present
            p["intensity"] = h_ratio * _get_lightest_profile(means_dict)["mean_intensity"]
            continue
        p["intensity"] = means_dict[p["label"]]["mean_intensity"]

    # Normalise intensities
    sum_intensities = sum([p["intensity"] for p in profiles_dict.values()])
    for p in profiles_dict.values():
        p["intensity"] /= sum_intensities

    # Set widths
    for p in profiles_dict.values():
        try:
            p["width"] = means_dict[p["label"]]["mean_width"]
        except KeyError:
            continue

    # Fix all widths except lightest mass
    for p in profiles_dict.values():
        if p == _get_lightest_profile(profiles_dict):
            continue
        p["width_lb"] = p["width"]
        p["width_ub"] = p["width"]

    result_profiles_table = _convert_dict_to_table(profiles_dict)
    return result_profiles_table


def _convert_table_to_dict(table):
    result_dict = {}
    for i in range(table.rowCount()):
        row_dict = table.row(i)
        result_dict[row_dict["label"]] = row_dict
    return result_dict


def _convert_dict_to_table(m_dict):
    table = CreateEmptyTableWorkspace()
    for p in m_dict.values():
        if table.columnCount() == 0:
            for key, value in p.items():
                value_type = "str" if isinstance(value, str) else "float"
                table.addColumn(value_type, key)

        table.addRow(p)
    return table


def _get_lightest_profile(p_dict):
    profiles = [p for p in p_dict.values()]
    masses = [p["mass"] for p in p_dict.values()]
    return profiles[np.argmin(masses)]


def init_analysis_algorithm(ws_name: str, inputs_class: type[BackwardAnalysisInputs] | type[ForwardAnalysisInputs]):
    # Skip if workspace not found in ADS
    if not AnalysisDataService.doesExist(ws_name):
        logger.warning(f"\n{ws_name} not found in ADS: Skipping cropping and masking ...\n")
        return

    profiles_table = create_profiles_table(ws_name + "_initial_parameters", inputs_class)
    instrument_parameters_file = str(FilesManager.get_instrument_parameters_dir() / inputs_class.instrument_parameters_file)

    scattering_type = general_helpers._get_scattering_type(inputs_class)

    if scattering_type == ScatteringType.FORWARD:
        inputs_class.chosen_mass_index = 0
        inputs_class.intensity_ratio_of_hydrogen_to_chosen_mass = 0

    kwargs = {
        "InputWorkspace": ws_name,
        "InputProfiles": profiles_table.name(),
        "InstrumentParametersFile": instrument_parameters_file,
        "ChosenMassIndex": inputs_class.chosen_mass_index,
        "HRatioToChosenMass": inputs_class.intensity_ratio_of_hydrogen_to_chosen_mass,
        "NumberOfIterations": int(inputs_class.number_of_iterations_for_corrections),
        "InvalidDetectors": convert_to_list_of_spectrum_numbers(inputs_class.mask_detectors),
        "MultipleScatteringCorrection": inputs_class.do_multiple_scattering_correction,
        "GammaCorrection": inputs_class.do_gamma_correction,
        "ModeRunning": ModeRunning.BACKWARD.value if scattering_type == ScatteringType.BACKWARD else ModeRunning.FORWARD.value,
        "TransmissionGuess": inputs_class.transmission_guess,
        "MultipleScatteringOrder": int(inputs_class.multiple_scattering_order),
        "NumberOfEvents": int(inputs_class.multiple_scattering_number_of_events),
        "Constraints": serialize_constraints(inputs_class.constraints),
        "OutputMeansTable": " Final_Means",
    }

    AlgorithmFactory.subscribe(VesuvioAnalysisRoutine)
    alg = AlgorithmManager.createUnmanaged("VesuvioAnalysisRoutine")
    alg.initialize()
    alg.setProperties(kwargs)
    return alg


def save_plots_from_outputs(output_ncp_group_name: str, output_directory: Path, minimal_output: bool = False):
    if minimal_output:
        return

    data_workspace_name = output_ncp_group_name.removesuffix("_ncp_group")
    if data_workspace_name == output_ncp_group_name:
        raise ValueError(f"Expected an NCP output group, got {output_ncp_group_name!r}")

    output_ncp_group = mtd[output_ncp_group_name]
    ws_data_sum = mtd[data_workspace_name + "_sum"]
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(subplot_kw={"projection": "mantid"})
    ax.errorbar(ws_data_sum, fmt="k.", label="Sum of spectra", elinewidth=1.5)

    label_list = ["Sum of spectra"]
    ws_fig_name = data_workspace_name + "_ncp_fits_sum"
    CloneWorkspace(ws_data_sum, OutputWorkspace=ws_fig_name)

    for ws_name in output_ncp_group.getNames():
        if not ws_name.startswith(data_workspace_name + "_") or not ws_name.endswith("_ncp"):
            continue

        label = ws_name[len(data_workspace_name) + 1 : -len("_ncp")]
        ws_sum = mtd[ws_name + "_sum"]
        ax.plot(ws_sum, label=f"Sum of {label} profile")
        label_list.append(f"Sum of {label} profile")
        AppendSpectra(ws_fig_name, ws_sum, OutputWorkspace=ws_fig_name)

    ax.set_xlabel("TOF")
    ax.set_ylabel("Counts")
    ax.set_title("Sum of NCP fits")
    ax.legend()

    plt.savefig(figure_directory / f"{data_workspace_name}_profiles_sum.pdf", bbox_inches="tight")
    plt.close(fig)
    SaveAscii(ws_fig_name, str(output_directory / f"{data_workspace_name}_profiles_sum.txt"), LogList=label_list)


def save_results_from_outputs(
    input_workspace_name: str,
    output_ncp_group_name: str,
    output_directory: Path,
    minimal_output: bool = False,
):
    final_iteration_workspace_name = output_ncp_group_name.removesuffix("_ncp_group")
    if final_iteration_workspace_name == output_ncp_group_name or not final_iteration_workspace_name.startswith(input_workspace_name + "_"):
        raise ValueError(f"Expected an NCP output group for input workspace {input_workspace_name!r}, got {output_ncp_group_name!r}")

    output_directory.mkdir(parents=True, exist_ok=True)
    all_workspaces = [name for name in mtd.getObjectNames() if name.startswith(input_workspace_name + "_")]

    if minimal_output:
        final_prefix = final_iteration_workspace_name + "_"
        for ws_name in all_workspaces:
            if ws_name in (final_iteration_workspace_name + "_total_ncp", final_iteration_workspace_name + "_total_fse"):
                SaveAscii(ws_name, str(output_directory / ws_name) + ".txt")
            elif ws_name.startswith(final_prefix) and ws_name.endswith(("_fit_results", "_means")):
                SaveAscii(ws_name, str(output_directory / ws_name) + ".txt")
        return

    for ws_name in all_workspaces:
        if ws_name.endswith(("ncp", "fse", "initial_parameters", "means", "fit_results")):
            SaveAscii(ws_name, str(output_directory / ws_name) + ".txt")
        if ws_name[-1].isdigit():
            SaveNexus(ws_name, str(output_directory / ws_name) + ".nxs")


def crop_and_mask_workspace(ws_name, inputs_class: type[BackwardAnalysisInputs] | type[ForwardAnalysisInputs]):
    """Returns cloned and cropped workspace with modified name"""

    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", inputs_class.detectors)
    if match is None:
        raise ValueError(f"Invalid detectors range: {inputs_class.detectors!r}")
    first_detector, last_detector = map(int, match.groups())

    CropWorkspace(
        InputWorkspace=ws_name,
        StartWorkspaceIndex=mtd[ws_name].getIndexFromSpectrumNumber(first_detector),
        EndWorkspaceIndex=mtd[ws_name].getIndexFromSpectrumNumber(last_detector),
        OutputWorkspace=ws_name,
    )
    MaskDetectors(Workspace=ws_name, SpectraList=inputs_class.mask_detectors)

    # NOTE: Did not use MaskBins or RemoveBins from Mantid because those
    # algorithms act on histogram workspaces. Converting the workspace to
    # histogram and then back to point data removes masked flag on the bins.
    # Better to just set y data to zeros directly.
    mask_time_of_flight_bins_with_zeros(ws_name, inputs_class)
    return


def load_input_ws(input_class: type[BackwardAnalysisInputs] | type[ForwardAnalysisInputs]):
    raw_path, empty_path = store_input_ws_if_not_on_path(input_class)

    raw_name = raw_path.stem
    empty_name = empty_path.stem

    Load(Filename=str(raw_path), OutputWorkspace=raw_name)
    Load(Filename=str(empty_path), OutputWorkspace=empty_name)

    SumSpectra(InputWorkspace=raw_name, OutputWorkspace=raw_name + "_sum")
    SumSpectra(InputWorkspace=empty_name, OutputWorkspace=empty_name + "_sum")

    return raw_name, empty_name


def store_input_ws_if_not_on_path(
    inputs_class: type[BackwardAnalysisInputs] | type[ForwardAnalysisInputs],
) -> tuple[Path, Path]:
    scattering_type = general_helpers._get_scattering_type(inputs_class)

    if scattering_type == ScatteringType.BACKWARD:
        raw_path = FilesManager.get_reduction_inputs_dir() / FilesManager.get_backward_raw_filename()
        empty_path = FilesManager.get_reduction_inputs_dir() / FilesManager.get_backward_empty_filename()

    elif scattering_type == ScatteringType.FORWARD:
        raw_path = FilesManager.get_reduction_inputs_dir() / FilesManager.get_forward_raw_filename()
        empty_path = FilesManager.get_reduction_inputs_dir() / FilesManager.get_forward_empty_filename()
    else:
        raise ValueError(f"Input class for workspace not valid: {inputs_class.__name__}")

    if not ws_history_matches_inputs(inputs_class.runs, inputs_class.mode, inputs_class.instrument_parameters_file, raw_path):
        save_ws_from_load_vesuvio(
            inputs_class.runs,
            inputs_class.mode,
            str(FilesManager.get_instrument_parameters_dir() / inputs_class.instrument_parameters_file),
            raw_path,
        )

    if not ws_history_matches_inputs(inputs_class.empty_runs, inputs_class.mode, inputs_class.instrument_parameters_file, empty_path):
        save_ws_from_load_vesuvio(
            inputs_class.empty_runs,
            inputs_class.mode,
            str(FilesManager.get_instrument_parameters_dir() / inputs_class.instrument_parameters_file),
            empty_path,
        )

    return raw_path, empty_path


def convert_to_list_of_spectrum_numbers(detectors):
    if isinstance(detectors, str):
        detector_ranges = [r.split("-") for r in detectors.replace(" ", "").split(",")]
        return [i for r in detector_ranges for i in range(int(r[0]), int(r[-1]) + 1)]

    if isinstance(detectors, list) or isinstance(detectors, np.ndarray):
        return [int(d) for d in detectors]

    raise ValueError("Type not recognized: Masked detectors should be string, list or array.")


def ws_history_matches_inputs(runs, mode, ipfile, ws_path):
    if not (ws_path.is_file()):
        logger.notice(f"Cached workspace not found at {ws_path}")
        return False

    ws = Load(Filename=str(ws_path))
    ws_history = ws.getHistory()
    metadata = ws_history.getAlgorithmHistory(0)

    saved_runs = metadata.getPropertyValue("Filename")
    if saved_runs != runs:
        logger.notice(f"Filename in saved workspace did not match: {saved_runs} and {runs}")
        return False

    saved_mode = metadata.getPropertyValue("Mode")
    if saved_mode != mode:
        logger.notice(f"Mode in saved workspace did not match: {saved_mode} and {mode}")
        return False

    saved_ipfile_name = ntpath.basename(metadata.getPropertyValue("InstrumentParFile"))
    if saved_ipfile_name != ipfile:
        logger.notice(f"IP files in saved workspace did not match: {saved_ipfile_name} and {ipfile}")
        return False

    logger.notice("\nLocally saved workspace metadata matched with analysis inputs.\n")
    DeleteWorkspace(ws)
    return True


def save_ws_from_load_vesuvio(runs, mode, ipfile, ws_path):
    if Tags.BACKWARD in ws_path.stem:
        spectra = "3-134"
    elif Tags.FORWARD in ws_path.stem:
        spectra = "135-198"
    else:
        raise ValueError(f"Invalid name to save workspace: {ws_path.name}")

    vesuvio_ws = LoadVesuvio(
        Filename=runs,
        SpectrumList=spectra,
        Mode=mode,
        InstrumentParFile=str(ipfile),
        OutputWorkspace=ws_path.name,
        LoadLogFiles=False,
    )

    SaveNexus(vesuvio_ws, Filename=str(ws_path.absolute()))
    print(f"Workspace saved locally at: {ws_path.absolute()}")
    return


def save_fitting_input_workspaces(
    analysis_inputs: type[BackwardAnalysisInputs] | type[ForwardAnalysisInputs] | None,
) -> None:
    fitting_inputs_dir = FilesManager.get_fitting_inputs_dir()
    fitting_inputs_dir.mkdir(parents=True, exist_ok=True)
    if analysis_inputs is None:
        logger.warning("Could not save derived fitting workspaces because the inputs class was not provided.")
        return
    if not analysis_inputs.run_this_scattering_type:
        return

    iteration = str(analysis_inputs.number_of_iterations_for_corrections)
    overwrite_workspace = general_helpers.get_workspace_name_if_path(analysis_inputs.overwrite_analysis_input_workspace)
    if overwrite_workspace and AnalysisDataService.doesExist(overwrite_workspace):
        base_workspace_name = overwrite_workspace
    else:
        base_workspace_name = str(analysis_inputs.name_of_subtracted_workspace)

    workspace_name = f"{base_workspace_name}_{iteration}"
    ncp_group_name = f"{workspace_name}_ncp_group"

    if not AnalysisDataService.doesExist(workspace_name) or not AnalysisDataService.doesExist(ncp_group_name):
        logger.warning(f"Could not save derived fitting workspaces because expected reduction outputs are missing: {workspace_name}.")
        return

    ws_to_fit = mtd[workspace_name]
    ws_to_fit_ncps = mtd[ncp_group_name]

    ws_resolution = fitting_helpers.calculate_resolution(
        min(analysis_inputs.masses),
        ws_to_fit,
        analysis_inputs.range_for_rebinning_in_y_space,
    )

    ws_lighest_data, ws_lighest_ncp = fitting_helpers.isolate_lighest_mass_data(
        ws_to_fit,
        ws_to_fit_ncps,
        analysis_inputs.subtract_calculated_fse_from_data,
    )

    ws_joy, ws_joy_avg = y_space_reduction(ws_lighest_data, ws_lighest_ncp, analysis_inputs)

    SaveNexus(ws_resolution, str(fitting_inputs_dir / f"{workspace_name}_ws_resolution.nxs"))
    SaveNexus(ws_joy, str(fitting_inputs_dir / f"{workspace_name}_joy.nxs"))
    SaveNexus(ws_joy_avg, str(fitting_inputs_dir / f"{workspace_name}_joy_avg.nxs"))


def mask_time_of_flight_bins_with_zeros(ws_name, inputs_class: type[BackwardAnalysisInputs] | type[ForwardAnalysisInputs]):
    """
    Masks a given TOF range on ws with zeros on dataY.
    Leaves errors dataE unchanged, as they are used by later treatments.
    """

    if inputs_class.mask_time_of_flight_range:
        ranges = []
        for mask_range in inputs_class.mask_time_of_flight_range.split(","):
            match = re.fullmatch(r"\s*(\d+(?:\.\d*)?)\s*(?:-\s*(\d+(?:\.\d*)?))?\s*", mask_range)
            if match is None:
                raise ValueError(f"Invalid mask_time_of_flight_range: {inputs_class.mask_time_of_flight_range!r}")
            x_min, x_max = match.groups()
            ranges.append((float(x_min), float(x_max) if x_max is not None else float(x_min)))

        ws = mtd[ws_name]
        for i in range(ws.getNumberHistograms()):
            x = ws.dataX(i)
            masked = np.zeros(x.shape, dtype=bool)
            for x_min, x_max in ranges:
                masked |= (x >= x_min) & (x <= x_max)
            ws.mutableY(i)[masked] = 0
    return


def create_profiles_table(name, ai):
    table = CreateEmptyTableWorkspace(OutputWorkspace=name)
    table.addColumn(type="str", name="label")
    table.addColumn(type="float", name="mass")
    table.addColumn(type="float", name="intensity")
    table.addColumn(type="float", name="intensity_lb")
    table.addColumn(type="float", name="intensity_ub")
    table.addColumn(type="float", name="width")
    table.addColumn(type="float", name="width_lb")
    table.addColumn(type="float", name="width_ub")
    table.addColumn(type="float", name="center")
    table.addColumn(type="float", name="center_lb")
    table.addColumn(type="float", name="center_ub")

    def wrapb(bound):
        # Literally to just account for NoneType
        if bound is None:
            return np.inf
        return bound

    for mass, intensity, width, center, intensity_bound, width_bound, center_bound in zip(
        ai.masses,
        ai.initial_fitting_parameters[::3],
        ai.initial_fitting_parameters[1::3],
        ai.initial_fitting_parameters[2::3],
        ai.fitting_bounds[::3],
        ai.fitting_bounds[1::3],
        ai.fitting_bounds[2::3],
    ):
        table.addRow(
            [
                str(float(mass)),
                float(mass),
                float(intensity),
                float(wrapb(intensity_bound[0])),
                float(wrapb(intensity_bound[1])),
                float(width),
                float(wrapb(width_bound[0])),
                float(wrapb(width_bound[1])),
                float(center),
                float(wrapb(center_bound[0])),
                float(wrapb(center_bound[1])),
            ]
        )

    print_table_workspace(table)
    return table


def y_space_reduction(wsTOF, ws_ncp, inputs: type[BackwardAnalysisInputs] | type[ForwardAnalysisInputs]):
    """Seperate procedures depending on masking specified."""
    mass0 = inputs.masses[0]
    ncp = ws_ncp.extractY()

    rebin_pars = inputs.range_for_rebinning_in_y_space

    if inputs.mask_time_of_flight_range and inputs.mask_zeros_with == Masking.NCP:
        wsTOF = replace_zeros_with_ncp(wsTOF, ncp)

    ws_joy = ConvertToYSpace(wsTOF, Mass=mass0, OutputWorkspace=wsTOF.name() + "_joy")

    if inputs.mask_time_of_flight_range and inputs.mask_zeros_with == Masking.NAN:
        # NOTE: Technically this rebin is general and gives the same result as Rebin when
        # no zeros are present in dataY, but I didn't make this the default path
        # because it can be confusing.
        ws_joy_rebin = rebin_masked_weighted(ws_joy, rebin_pars, "_rebin")
    else:
        ws_joy_rebin = Rebin(InputWorkspace=ws_joy, Params=rebin_pars, OutputWorkspace=ws_joy.name() + "_rebin")

    Integration(ws_joy_rebin, OutputWorkspace="temp_integral")
    ws_joy_rebin = Divide(LHSWorkspace=ws_joy_rebin, RHSWorkspace="temp_integral", OutputWorkspace=ws_joy_rebin.name() + "_norm")

    # Weighted average over spectra
    ws_joy_rebin_avg = SumSpectra(
        InputWorkspace=ws_joy_rebin, WeightedSum=True, MultiplyBySpectra=False, OutputWorkspace=ws_joy_rebin.name() + "_avg"
    )

    if inputs.do_symmetrisation:
        ws_joy_rebin_avg = symmetrize_weighted_mean(ws_joy_rebin_avg)

    DeleteWorkspace("temp_integral")
    return ws_joy_rebin, ws_joy_rebin_avg


def rebin_masked_weighted(ws_joy, rebin_pars, suffix):
    """
    Rebins y-space data with masked (zero) points using Mantid Rebin on weighted sums.
    With w=1/E^2 (w=0 for masked points), Rebin gives sum(f*w*Y) and sum(f*w), where f is the
    overlap width of the new and old bin divided by the old bin width. Masked points therefore
    do not contribute.
    Rebin propagates errors as E^2 = sum(f * E_old^2) (not f^2). Passing E_old=sqrt(w) with the
    w*Y values then gives var(w*Y sum) = sum(f * w), so that
    var(mean) = sum(f * w) / sum(f * w)^2 = 1 / sum(f * w).
    This is exact for f=1 and slightly conservative for partially overlapping bins, where f^2 < f.
    Returns the per spectrum workspace and the workspace averaged over spectra.
    """
    dataX, dataY, dataE = extractWS(ws_joy)

    # Masked points can be 0 or NaN, in either dataY, dataE or dataX
    valid = (dataY != 0) & np.isfinite(dataY) & np.isfinite(dataE) & np.isfinite(dataX)
    dataY = np.where(valid, dataY, 0)  # Avoids 0 * NaN
    dataE = np.where(valid, dataE, 0)

    valid &= dataE > 0
    weights = np.zeros(dataY.shape)
    weights[valid] = 1 / np.square(dataE[valid])

    CloneWorkspace(InputWorkspace=ws_joy, OutputWorkspace="temp_wy")
    pass_data_into_ws(dataX, weights * dataY, np.sqrt(weights), mtd["temp_wy"])
    Rebin(InputWorkspace="temp_wy", Params=rebin_pars, OutputWorkspace="temp_wy")

    CloneWorkspace(InputWorkspace=ws_joy, OutputWorkspace="temp_w")
    pass_data_into_ws(dataX, weights, np.zeros_like(weights), mtd["temp_w"])
    Rebin(InputWorkspace="temp_w", Params=rebin_pars, OutputWorkspace="temp_w")

    wy = mtd["temp_wy"].extractY()
    var = np.square(mtd["temp_wy"].extractE())
    w = mtd["temp_w"].extractY()
    dataX_rebin = mtd["temp_wy"].extractX()

    # Spectra in y space, one dataY point per bin of each spectrum
    non_empty = w > 0
    safe = np.where(non_empty, w, 1)
    specY = np.where(non_empty, wy / safe, 0)
    specE = np.where(non_empty, np.sqrt(var) / safe, 0)

    # Use CloneWokspace to copy correct metadata
    ws_rebin = CloneWorkspace(InputWorkspace="temp_wy", OutputWorkspace=ws_joy.name() + suffix)
    pass_data_into_ws(dataX_rebin, specY, specE, ws_rebin)

    DeleteWorkspaces(["temp_wy", "temp_w"])
    return ws_rebin


def replace_zeros_with_ncp(ws, ncp):
    """
    Replaces columns of bins with zeros on dataY with NCP provided.
    """
    dataX, dataY, dataE = extractWS(ws)
    mask = np.all(dataY == 0, axis=0)  # Masked Cols

    dataY[:, mask] = ncp[:, mask[: ncp.shape[1]]]  # mask of ncp adjusted for last col present or not

    ws_mask = CloneWorkspace(ws, OutputWorkspace=ws.name() + "_ncpmask")
    pass_data_into_ws(dataX, dataY, dataE, ws_mask)
    SumSpectra(ws_mask, OutputWorkspace=ws_mask.name() + "_sum")
    return ws_mask


def symmetrize_weighted_mean(ws):
    """
    Symmetrizes workspace using Mantid WeightedMean on two workspaces:
    one with the lhs of zero reflected onto the rhs and one with the rhs reflected onto the lhs.
    Needs to have symmetric binning.
    """
    dataX, dataY, dataE = extractWS(ws)
    np.testing.assert_allclose(dataX, -np.flip(dataX, axis=1), atol=1e-8, err_msg="symmetrisatino range is not symmetric around zero.")

    nbins = dataY.shape[1]
    half = nbins // 2
    right = slice(nbins - half, nbins)  # Excludes central point when odd
    left = slice(0, half)

    # lhs reflected onto rhs
    yL, eL = dataY.copy(), dataE.copy()
    yL[:, right] = np.flip(dataY, axis=1)[:, right]
    eL[:, right] = np.flip(dataE, axis=1)[:, right]

    # rhs reflected onto lhs
    yR, eR = dataY.copy(), dataE.copy()
    yR[:, left] = np.flip(dataY, axis=1)[:, left]
    eR[:, left] = np.flip(dataE, axis=1)[:, left]

    ws_lhs = CloneWorkspace(ws, OutputWorkspace="temp_lhs_reflected")
    pass_data_into_ws(dataX, yL, eL, ws_lhs)
    ws_rhs = CloneWorkspace(ws, OutputWorkspace="temp_rhs_reflected")
    pass_data_into_ws(dataX, yR, eR, ws_rhs)

    # Weighted mean of lhs and rhs
    ws_sym = WeightedMean(InputWorkspace1=ws_lhs, InputWorkspace2=ws_rhs, OutputWorkspace=ws.name() + "_sym")
    DeleteWorkspaces(["temp_lhs_reflected", "temp_rhs_reflected"])
    return ws_sym
