import mvesuvio
from mantid.simpleapi import Rebin, Scale, Minus, SetSample, SaveAscii, mtd
from pathlib import Path
import math
from typing import Dict
from mvesuvio import ConfigArgInputs
from mvesuvio.globals import Masking, Mode
from mvesuvio.util import general_helpers
from mvesuvio.util import reduction_helpers
from mvesuvio.util.files_manager import FilesManager


class SharedGeometry:
    """
    Sample physical dimensions, in cm.
    Cylinder volume is the same as slab volume.
    """

    slab_height_cm: float = 10.0
    slab_width_cm: float = 10.0
    slab_thickness_cm: float = 0.1
    cylinder_height_cm: float = 10.0

    @property
    def slab_volume_cm3(self) -> float:
        return self.slab_height_cm * self.slab_width_cm * self.slab_thickness_cm

    @property
    def cylinder_radius_cm(self) -> float:
        return math.sqrt(self.slab_volume_cm3 / (math.pi * self.cylinder_height_cm))

    @property
    def cylinder_volume_cm3(self) -> float:
        return math.pi * self.cylinder_radius_cm**2 * self.cylinder_height_cm

    def slab_dict(self) -> Dict[str, object]:
        """SetSample FlatPlate dictionary, dimensions in cm."""
        return {
            "Shape": "FlatPlate",
            "Width": self.slab_width_cm,
            "Height": self.slab_height_cm,
            "Thick": self.slab_thickness_cm,
            "Center": [0.0, 0.0, 0.0],
            "Angle": 0.0,
        }

    def cylinder_dict(self) -> Dict[str, object]:
        """SetSample Cylinder dictionary, dimensions in cm."""
        return {
            "Shape": "Cylinder",
            "Height": self.cylinder_height_cm,
            "Radius": self.cylinder_radius_cm,
            "Center": [0.0, 0.0, 0.0],
            "Axis": [0.0, 1.0, 0.0],
        }


# Pick Slab or Cylinder geometry
SAMPLE_SHAPE = SharedGeometry().slab_dict()

# Example stoichiometry, replace with actual values
STOICHIOMETRY = {"H": 4, "C": 1, "O": 2, "Al": 1}

BOUND_SCATTERING_XS_BARN = {"H": 82.0, "C": 5.55, "O": 4.23, "Al": 1.49}


# Bound-cross-section-weighted recoil intensities, in a common arbitrary scale
RECOIL_WEIGHT = {element: STOICHIOMETRY[element] * BOUND_SCATTERING_XS_BARN[element] for element in STOICHIOMETRY}


class BackwardAnalysisInputs:
    run_this_scattering_type = True
    minimal_output = False

    # Name of workspace to use as input of analysis, instead of using the runs below
    # If already loaded, picks up the name in Mantid, otherwise attempts to load with Load()
    # Leave empty to use the runs specified below
    overwrite_analysis_input_workspace = ""

    runs = "43066-43076"  # Runs of your sample dataset
    empty_runs = "41876-41923"  # Empty CCR
    mode = Mode.DOUBLE_DIFFERENCE
    instrument_parameters_file = "ip2019.par"
    detectors = "3-134"
    mask_detectors = [18, 34, 42, 43, 59, 60, 62, 118, 119, 133]  # Can also be a string "18, 34, 42-43, 59-60, 62, 118-119, 133"
    time_of_flight_binning = "275.,1.,420"
    mask_time_of_flight_range = None  # Can be string eg. "110-120, 200-210"
    # Scaling factors, leave at default of 1 for most cases
    scale_empty_workspace = 1
    scale_raw_workspace = 1
    name_of_subtracted_workspace = "back"

    # Atomic mass in a.m.u. of each element/isotope present in sample + cell EXCEPT HYDROGEN
    masses = [12, 16, 27]

    # fmt: off
    initial_fitting_parameters = [  # NCP intensities, NCP widths, NCP centers
        1, 12, 0.0,
        1, 12, 0.0,
        1, 12.5, 0.0,
    ]
    fitting_bounds = [
        [0, None], [8, 16], [-3, 1],
        [0, None], [8, 16], [-3, 1],
        [0, None], [11, 14], [-3, 1],
    ]
    # fmt: on
    constraints = {"type": "eq", "fun": lambda par: (RECOIL_WEIGHT["C"] * par[0] - RECOIL_WEIGHT["O"] * par[3])}

    number_of_iterations_for_corrections = 0  # 4
    # Y-space reduction is performed at the end of reduction and its outputs are used by run_fitting.py
    range_for_rebinning_in_y_space = "-25, 0.5, 25"  # Needs to be symetric, usually bounds = 10 x lowest mass (a.m.u.)
    subtract_calculated_fse_from_data = True
    do_symmetrisation = False
    # Type of masking
    # Masking.NAN: Zeros in workspace being fit are ignored
    # Masking.NCP: Zeros in workspace being fit are replaced by the fitted neutron compton profile
    mask_zeros_with = Masking.NAN
    do_multiple_scattering_correction = True
    # Exact recoil ratio used to estimate Hydrogen intensity
    chosen_mass_index = 0  # C is indexed 0 in backward masses list
    intensity_ratio_of_hydrogen_to_chosen_mass = (
        RECOIL_WEIGHT["H"] / RECOIL_WEIGHT["C"]
        # Set to zero to estimate
    )
    transmission_guess = 0.8  # [1 - 2(1-T)] --> Twice the absorption, T: Experimental value from VesuvioTransmission
    multiple_scattering_order = 2
    multiple_scattering_number_of_events = 1.0e5  # 1.0e6 for smoother correction, at the cost of higher execution time
    do_gamma_correction = False


class ForwardAnalysisInputs:
    run_this_scattering_type = True
    minimal_output = False

    # Name of workspace to use as input of analysis, instead of using the runs below
    # If already loaded, picks up the name in Mantid, otherwise attempts to load with Load()
    # Leave empty to use the runs specified below
    overwrite_analysis_input_workspace = ""

    runs = "43066-43076"
    empty_runs = "43868-43911"  # Empty CCR
    mode = Mode.SINGLE_DIFFERENCE
    instrument_parameters_file = "ip2018_3.par"
    detectors = "144-182"
    mask_detectors = [173, 174, 179]  # Can also be a string "173-174, 179"
    time_of_flight_binning = "110,1,430"
    mask_time_of_flight_range = None  # Can be string Eg. "110-120, 200-210"
    # Scaling factors, leave at default of 1 for most cases
    scale_empty_workspace = 1
    scale_raw_workspace = 1
    name_of_subtracted_workspace = "front"

    masses = [1.0079, 12, 16, 27]  # Atomic mass in a.m.u. of each element/isotope present in sample + cell
    # fmt: off
    initial_fitting_parameters = [  # Intensities, NCP widths, NCP centers
        1, 4.7, 0.0,
        1, 12.71, 0.0,
        1, 8.76, 0.0,
        1, 13.897, 0.0,
    ]
    fitting_bounds = [
        [0, None], [3, 6], [-3, 1],
        [0, None], [12.71, 12.71], [-3, 1],
        [0, None], [8.76, 8.76], [-3, 1],
        [0, None], [13.897, 13.897], [-3, 1],
    ]
    # fmt: on
    constraints = {"type": "eq", "fun": lambda par: (RECOIL_WEIGHT["C"] * par[3] - RECOIL_WEIGHT["O"] * par[6])}

    number_of_iterations_for_corrections = 1  # 4
    # Y-space reduction is performed at the end of reduction and its outputs are used by run_fitting.py
    range_for_rebinning_in_y_space = "-25, 0.5, 25"  # Needs to be symetric, usually bounds = 10 x lowest mass (a.m.u.)
    subtract_calculated_fse_from_data = True
    do_symmetrisation = False
    # Type of masking
    # Masking.NAN: Zeros in workspace being fit are ignored
    # Masking.NCP: Zeros in workspace being fit are replaced by the fitted neutron compton profile
    mask_zeros_with = Masking.NAN
    do_multiple_scattering_correction = True
    transmission_guess = 0.9  # T : Experimental value from VesuvioTransmission
    multiple_scattering_order = 2
    multiple_scattering_number_of_events = 1.0e5  # 1.0e6 for smoother correction, at the cost of higher execution time
    do_gamma_correction = True


########################
### END OF USER EDIT ###
########################


def run_reduction(configure_with_main: bool = True):
    if configure_with_main:
        mvesuvio.main(ConfigArgInputs(experiment_dir=str(Path(__file__).parent), ip_dir=""))

    back_input_ws = ""
    front_input_ws = ""

    if BackwardAnalysisInputs.run_this_scattering_type:
        if BackwardAnalysisInputs.overwrite_analysis_input_workspace:
            back_input_ws = general_helpers.load_overwritten_workspace_if_specified(BackwardAnalysisInputs)
        else:
            raw_name, empty_name = reduction_helpers.load_input_ws(BackwardAnalysisInputs)

            ### User Editing Section ###
            # For some edge cases, you can modify the raw and empty workspaces as needed
            # You can also apply any modifications to the subtracted workspace
            # The only requirement is that the final subtracted workspace is called BackwardAnalysisInputs.name_of_subtracted_workspace

            Rebin(InputWorkspace=raw_name, Params=BackwardAnalysisInputs.time_of_flight_binning, OutputWorkspace=raw_name)
            Rebin(InputWorkspace=empty_name, Params=BackwardAnalysisInputs.time_of_flight_binning, OutputWorkspace=empty_name)

            Scale(InputWorkspace=raw_name, Factor=BackwardAnalysisInputs.scale_raw_workspace, OutputWorkspace=raw_name)
            Scale(InputWorkspace=empty_name, Factor=BackwardAnalysisInputs.scale_empty_workspace, OutputWorkspace=empty_name)

            Minus(LHSWorkspace=raw_name, RHSWorkspace=empty_name, OutputWorkspace=BackwardAnalysisInputs.name_of_subtracted_workspace)

            ### End of User Editing Section ###
            back_input_ws = BackwardAnalysisInputs.name_of_subtracted_workspace

        # Set sample geometry, used in MS corrections
        SetSample(InputWorkspace=back_input_ws, Geometry=SAMPLE_SHAPE)
        reduction_helpers.crop_and_mask_workspace(back_input_ws, BackwardAnalysisInputs)

    if ForwardAnalysisInputs.run_this_scattering_type:
        if ForwardAnalysisInputs.overwrite_analysis_input_workspace:
            front_input_ws = general_helpers.load_overwritten_workspace_if_specified(ForwardAnalysisInputs)
        else:
            raw_name, empty_name = reduction_helpers.load_input_ws(ForwardAnalysisInputs)

            ### User Editing Section ###
            # For some edge cases, you can modify the raw and empty workspaces as needed
            # You can also apply any modifications to the subtracted workspace
            # The only requirement is that the final subtracted workspace is called ForwardAnalysisInputs.name_of_subtracted_workspace

            Rebin(InputWorkspace=raw_name, Params=ForwardAnalysisInputs.time_of_flight_binning, OutputWorkspace=raw_name)
            Rebin(InputWorkspace=empty_name, Params=ForwardAnalysisInputs.time_of_flight_binning, OutputWorkspace=empty_name)

            Scale(InputWorkspace=raw_name, Factor=ForwardAnalysisInputs.scale_raw_workspace, OutputWorkspace=raw_name)
            Scale(InputWorkspace=empty_name, Factor=ForwardAnalysisInputs.scale_empty_workspace, OutputWorkspace=empty_name)

            Minus(LHSWorkspace=raw_name, RHSWorkspace=empty_name, OutputWorkspace=ForwardAnalysisInputs.name_of_subtracted_workspace)

            ### End of User Editing Section ###
            front_input_ws = ForwardAnalysisInputs.name_of_subtracted_workspace

        ## Set sample geometry, used in MS corrections
        SetSample(InputWorkspace=front_input_ws, Geometry=SAMPLE_SHAPE)
        reduction_helpers.crop_and_mask_workspace(front_input_ws, ForwardAnalysisInputs)

    back_alg = reduction_helpers.init_analysis_algorithm(back_input_ws, BackwardAnalysisInputs)
    front_alg = reduction_helpers.init_analysis_algorithm(front_input_ws, ForwardAnalysisInputs)

    if reduction_helpers.h_ratio_is_zero_when_h_present(BackwardAnalysisInputs, ForwardAnalysisInputs):
        h_ratios_table = reduction_helpers.run_estimate_h_ratio(
            back_alg=back_alg,
            front_alg=front_alg,
            back_masses=BackwardAnalysisInputs.masses,
            back_chosen_mass_index=BackwardAnalysisInputs.chosen_mass_index,
            rtol=0.01,
            max_iter=3,
        )
        SaveAscii(InputWorkspace=h_ratios_table, Filename=str(FilesManager.get_experiment_dir() / h_ratios_table.name()))

    if (
        BackwardAnalysisInputs.run_this_scattering_type
        and ForwardAnalysisInputs.run_this_scattering_type
        and back_alg is not None
        and front_alg is not None
    ):
        reduction_helpers.execute_joint_algorithms(back_alg=back_alg, front_alg=front_alg)
    elif BackwardAnalysisInputs.run_this_scattering_type and back_alg is not None:
        back_alg.execute()
    elif ForwardAnalysisInputs.run_this_scattering_type and front_alg is not None:
        front_alg.execute()

    general_helpers.make_summarised_log_file()
    reduction_helpers.save_fitting_input_workspaces(BackwardAnalysisInputs)
    reduction_helpers.save_fitting_input_workspaces(ForwardAnalysisInputs)
    for analysis_alg, input_workspace_name, inputs_class in (
        (back_alg, back_input_ws, BackwardAnalysisInputs),
        (front_alg, front_input_ws, ForwardAnalysisInputs),
    ):
        if analysis_alg is not None:
            output_ncp_group_name = analysis_alg.getPropertyValue("OutputNCPGroup")
            output_directory = FilesManager.get_reduction_outputs_dir()
            reduction_helpers.save_results_from_outputs(
                input_workspace_name=input_workspace_name,
                output_ncp_group_name=output_ncp_group_name,
                output_directory=output_directory,
                minimal_output=inputs_class.minimal_output,
            )
            # Plots are saved for every iteration, not only the final one
            iteration_ncp_groups = sorted(
                name for name in mtd.getObjectNames() if name.startswith(input_workspace_name + "_") and name.endswith("_ncp_group")
            )
            for iteration_ncp_group in iteration_ncp_groups:
                reduction_helpers.save_plots_from_outputs(
                    output_ncp_group_name=iteration_ncp_group,
                    output_directory=output_directory,
                    minimal_output=inputs_class.minimal_output,
                )


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    run_reduction()
