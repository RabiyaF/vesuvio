import os
import sys
from contextlib import suppress
from functools import partial

import numpy as np
import scipy.constants
from mantid.api import (
    AnalysisDataService,
    FileAction,
    FileProperty,
    ITableWorkspaceProperty,
    Progress,
    PropertyMode,
    PythonAlgorithm,
    TextAxis,
    WorkspaceFactory,
    WorkspaceGroupProperty,
)
from mantid.kernel import (
    Direction,
    FloatArrayBoundedValidator,
    FloatArrayMandatoryValidator,
    FloatArrayProperty,
    IntArrayBoundedValidator,
    IntArrayProperty,
    StringListValidator,
    StringMandatoryValidator,
    logger,
)
from mantid.simpleapi import (
    AppendSpectra,
    CloneWorkspace,
    CreateEmptyTableWorkspace,
    CreateWorkspace,
    CropWorkspace,
    DeleteWorkspace,
    Divide,
    ExtractSingleSpectrum,
    ExtractUnmaskedSpectra,
    FindPeakBackground,
    FindPeaks,
    Fit,
    GroupWorkspaces,
    MaskDetectors,
    Plus,
    RebinToWorkspace,
    ReplaceSpecialValues,
    SumSpectra,
    mtd,
)

from mvesuvio.globals import FitFunction, Mode, PeakType, SharedParameterFitType
from mvesuvio.util.calibration_helpers import EVSGlobals, EVSMiscFunctions, InvalidDetectors


class EVSCalibrationFit(PythonAlgorithm):
    def summary(self):
        return "Fits peaks to a list of spectra using the mass or the d-spacings (for bragg peaks) of the sample."

    def category(self):
        return "VesuvioCalibration"

    def PyInit(self):
        self.declareProperty(WorkspaceGroupProperty("Samples", "", Direction.Input), doc="Sample workspaces to fit peaks to.")

        self.declareProperty(
            WorkspaceGroupProperty("Background", "", Direction.Input, PropertyMode.Optional),
            doc="Optional workspace group to use as a background.",
        )

        self.declareProperty(
            "Mode",
            Mode.FOIL_OUT.value,
            StringListValidator([mode.value for mode in Mode]),
            doc="Mode to load files with. This is passed to the LoadVesuvio algorithm. Default is FoilOut.",
        )

        self.declareProperty(
            "Function",
            FitFunction.GAUSSIAN.value,
            StringListValidator([func.value for func in FitFunction]),
            doc="Function to fit each of the spectra with. Default is Gaussian",
        )

        spectrum_validator = IntArrayBoundedValidator()
        spectrum_validator.setLower(EVSGlobals.DETECTOR_RANGE[0])
        spectrum_validator.setUpper(EVSGlobals.DETECTOR_RANGE[1])
        self.declareProperty(
            IntArrayProperty("SpectrumRange", EVSGlobals.DETECTOR_RANGE, spectrum_validator, Direction.Input),
            doc=f"Spectrum range to use. Default is the total range ({EVSGlobals.DETECTOR_RANGE[0]:d},{EVSGlobals.DETECTOR_RANGE[1]:d})",
        )

        self.declareProperty("Mass", 207.19, doc="Mass of the sample in amu to be used when calculating energy. Default is Pb: 207.19")

        greaterThanZero = FloatArrayBoundedValidator()
        greaterThanZero.setLower(0)
        self.declareProperty(
            FloatArrayProperty("DSpacings", [], greaterThanZero, Direction.Input),
            doc="List of d-spacings used to estimate the positions of bragg peaks in TOF.",
        )

        self.declareProperty(
            FloatArrayProperty("Energy", [EVSGlobals.ENERGY_ESTIMATE], FloatArrayMandatoryValidator(), Direction.Input),
            doc=f"List of estimated expected energies for peaks. Optional: the default is {EVSGlobals.ENERGY_ESTIMATE:f}",
        )

        self.declareProperty(
            FileProperty("InstrumentParameterFile", "", action=FileAction.OptionalLoad, extensions=["par"]),
            doc="Filename of the instrument parameter file.",
        )

        self.declareProperty(
            "PeakType",
            "",
            StringListValidator([peak.value for peak in PeakType]),
            doc="Choose the peak type that is being fitted.Note that supplying a set of dspacings overrides the setting here",
        )

        shared_fit_type_validator = StringListValidator([fit_type.value for fit_type in SharedParameterFitType])
        self.declareProperty(
            "SharedParameterFitType",
            SharedParameterFitType.INDIVIDUAL.value,
            doc="Calculate shared parameters using an individual and/orglobal fit.",
            validator=shared_fit_type_validator,
        )

        detector_validator = IntArrayBoundedValidator()
        detector_validator.setLower(EVSGlobals.DETECTOR_RANGE[0])
        detector_validator.setUpper(EVSGlobals.DETECTOR_RANGE[-1])
        self.declareProperty(
            IntArrayProperty("InvalidDetectors", [], detector_validator, Direction.Input),
            doc="List of detectors to be marked as invalid (3-198), to be excluded from analysis calculations.",
        )
        self.declareProperty(
            IntArrayProperty("InvalidDetectorsOut", [], detector_validator, Direction.Output),
            doc="List of detectors found as invalid to be output.",
        )

        self.declareProperty(
            ITableWorkspaceProperty("InstrumentParameterWorkspace", "", Direction.Input, PropertyMode.Optional),
            doc="Workspace contain instrument parameters.",
        )

        self.declareProperty("CreateOutput", False, doc="Create fitting workspaces for each of the parameters.")

        self.declareProperty("OutputWorkspace", "", StringMandatoryValidator(), doc="Name to call the output workspace.")

    def PyExec(self):
        self._setup()
        self._preprocess()

        if self._fitting_bragg_peaks:
            self._fit_bragg_peaks()
        else:
            self._fit_peaks()

        # create output of fit if required.
        if self._create_output and not self._fitting_bragg_peaks:
            self._generate_fit_workspaces()

        # set invalid detectors output param
        self.setProperty("InvalidDetectorsOut", self._invalid_detectors.get_all_invalid_detectors())

        # clean up workspaces
        if self._param_file != "":
            DeleteWorkspace(self._param_table)

    def _setup(self):
        """
        Setup parameters for fitting.
        """
        self._setup_spectra_list()
        self._setup_run_numbers_and_output_workspace()
        self._setup_function_type()
        self._setup_parameter_workspace()
        self._setup_peaks_and_set_crop_and_fit_ranges()
        self._setup_class_variables_from_properties()

    def _setup_class_variables_from_properties(self):
        self._mode = self.getProperty("Mode").value
        self._energy_estimates = self.getProperty("Energy").value
        self._sample_mass = self.getProperty("Mass").value
        self._create_output = self.getProperty("CreateOutput").value
        self._shared_parameter_fit_type = self.getProperty("SharedParameterFitType").value
        self._invalid_detectors = InvalidDetectors(self.getProperty("InvalidDetectors").value.tolist())

    def _setup_spectra_list(self):
        self._spec_list = self.getProperty("SpectrumRange").value.tolist()
        if len(self._spec_list) > 2:
            self._spec_list = [self._spec_list[0], self._spec_list[-1]]
        elif len(self._spec_list) == 1:
            self._spec_list = [self._spec_list[0]]
        elif len(self._spec_list) < 1:
            raise ValueError("You must specify a spectrum range.")

    def _setup_run_numbers_and_output_workspace(self):
        self._sample_run_numbers = self.getProperty("Samples").value
        self._bkg_run_numbers = self.getProperty("Background").value
        self._output_workspace_name = self.getPropertyValue("OutputWorkspace")
        if self._sample_run_numbers.size() == 0:
            raise ValueError("You must supply at least one sample run number.")
        if self._bkg_run_numbers is not None and self._bkg_run_numbers.size() > 0:
            background_suffix = "_".join(ws.name() for ws in self._bkg_run_numbers)
            self._background = self._output_workspace_name + "_Background_" + background_suffix

        suffix = "_".join(ws.name() for ws in self._sample_run_numbers)
        self._sample = self._output_workspace_name + "_Sample_" + suffix

    def _setup_function_type(self):
        self._peak_function = self.getProperty("Function").value
        self._func_param_names = EVSMiscFunctions.generate_fit_function_header(self._peak_function)
        self._func_param_names_error = EVSMiscFunctions.generate_fit_function_header(self._peak_function, error=True)

    def _setup_parameter_workspace(self):
        self._param_workspace = self.getPropertyValue("InstrumentParameterWorkspace")
        self._param_file = self.getPropertyValue("InstrumentParameterFile")
        if self._param_workspace != "":
            self._param_table = self._param_workspace
        elif self._param_file != "":
            base = os.path.basename(self._param_file)
            self._param_table = os.path.splitext(base)[0]
            EVSMiscFunctions.load_instrument_parameters(self._param_file, self._param_table)

    def _setup_peaks_and_set_crop_and_fit_ranges(self):
        self._d_spacings = self.getProperty("DSpacings").value
        self._d_spacings.sort()
        self._peak_type = self.getPropertyValue("PeakType")

        if self._fitting_bragg_peaks:
            self._ws_crop_range, self._fit_window_range = EVSGlobals.BRAGG_PEAK_CROP_RANGE, EVSGlobals.BRAGG_FIT_WINDOW_RANGE
        elif self._fitting_recoil_peaks:
            self._ws_crop_range, self._fit_window_range = EVSGlobals.RECOIL_PEAK_CROP_RANGE, EVSGlobals.RECOIL_FIT_WINDOW_RANGE
        elif self._fitting_resonance_peaks:
            self._ws_crop_range, self._fit_window_range = EVSGlobals.RESONANCE_PEAK_CROP_RANGE, EVSGlobals.RESONANCE_FIT_WINDOW_RANGE

    @property
    def _fitting_bragg_peaks(self):
        return len(self._d_spacings) > 0

    @property
    def _fitting_recoil_peaks(self):
        return self._peak_type == PeakType.RECOIL.value and not self._fitting_bragg_peaks

    @property
    def _fitting_resonance_peaks(self):
        return self._peak_type == PeakType.RESONANCE.value and not self._fitting_bragg_peaks

    def _preprocess(self):
        """
        Preprocess a workspace. This include optionally dividing by a background
        """
        xmin, xmax = self._ws_crop_range
        self._sum_and_crop_ws(self._sample_run_numbers, self._sample, xmin, xmax)

        if self._background_provided:
            self._sum_and_crop_ws(self._bkg_run_numbers, self._background, xmin, xmax)
            self._normalise_sample_by_background()

        ReplaceSpecialValues(self._sample, NaNValue=0, NaNError=0, InfinityValue=0, InfinityError=0, OutputWorkspace=self._sample)

    @property
    def _background_provided(self):
        return self._bkg_run_numbers is not None and self._bkg_run_numbers.size() > 0

    def _normalise_sample_by_background(self):
        RebinToWorkspace(WorkspaceToRebin=self._background, WorkspaceToMatch=self._sample, OutputWorkspace=self._background)
        Divide(self._sample, self._background, OutputWorkspace=self._sample)
        DeleteWorkspace(self._background)

    def _fit_bragg_peaks(self):
        estimated_peak_positions_all_spec = self._estimate_bragg_peak_positions()
        num_estimated_peaks, num_spectra = estimated_peak_positions_all_spec.shape

        self._prog_reporter = Progress(self, 0.0, 1.0, num_spectra)

        output_parameters_tbl_name = self._output_workspace_name + "_Peak_Parameters"
        self._create_output_parameters_table_ws(output_parameters_tbl_name, num_estimated_peaks)

        output_workspaces = []
        for index, estimated_peak_positions in enumerate(estimated_peak_positions_all_spec.transpose()):
            spec_number = self._spec_list[0] + index
            self._prog_reporter.report(f"Fitting to spectrum {spec_number:d}")

            find_peaks_output_name = f"{self._sample}_peaks_table_{spec_number:d}"
            fit_peaks_output_name = f"{self._output_workspace_name}_Spec_{spec_number:d}"
            fit_results = self._fit_peaks_to_spectra(
                index, spec_number, estimated_peak_positions, find_peaks_output_name, fit_peaks_output_name
            )

            fit_results_unconstrained = None
            if fit_results["status"] != "success":
                self._prog_reporter.report(f"Fitting to spectrum {spec_number:d} without constraining parameters")
                fit_results_unconstrained = self._fit_peaks_to_spectra(
                    index,
                    spec_number,
                    estimated_peak_positions,
                    find_peaks_output_name,
                    fit_peaks_output_name,
                    unconstrained=True,
                    x_range=(fit_results["xmin"], fit_results["xmax"]),
                )

            selected_params, unconstrained_fit_selected = self._select_best_fit_params(spec_number, fit_results, fit_results_unconstrained)

            self._output_params_to_table(spec_number, num_estimated_peaks, selected_params, output_parameters_tbl_name)

            output_workspaces.append(
                self._get_output_and_clean_workspaces(
                    fit_results_unconstrained is not None,
                    (fit_results_unconstrained is not None and fit_results_unconstrained is not False),
                    unconstrained_fit_selected,
                    find_peaks_output_name,
                    fit_peaks_output_name,
                )
            )

        if self._create_output:
            GroupWorkspaces(",".join(output_workspaces), OutputWorkspace=self._output_workspace_name + "_Peak_Fits")

    @staticmethod
    def _get_unconstrained_ws_name(ws_name):
        return ws_name + "_unconstrained"

    def _create_output_parameters_table_ws(self, output_table_name, num_estimated_peaks):
        table = CreateEmptyTableWorkspace(OutputWorkspace=output_table_name)

        col_headers = self._generate_column_headers(num_estimated_peaks)

        table.addColumn("int", "Spectrum")
        for name in col_headers:
            table.addColumn("double", name)
        AnalysisDataService.addOrReplace(output_table_name, table)

    def _generate_column_headers(self, num_estimated_peaks):
        param_names = self._get_param_names(num_estimated_peaks)
        err_names = [name + "_Err" for name in param_names]
        col_headers = [element for tupl in zip(param_names, err_names) for element in tupl]
        return col_headers

    def _get_param_names(self, num_estimated_peaks):
        param_names = ["f0.A0", "f0.A1"]
        for i in range(num_estimated_peaks):
            param_names += ["f" + str(i) + "." + name for name in self._func_param_names.values()]
        return param_names

    def _fit_peaks_to_spectra(
        self,
        workspace_index,
        spec_number,
        peak_estimates_list,
        find_peaks_output_name,
        fit_peaks_output_name,
        unconstrained=False,
        x_range=None,
    ):
        if unconstrained:
            find_peaks_output_name = self._get_unconstrained_ws_name(find_peaks_output_name)
            fit_peaks_output_name = self._get_unconstrained_ws_name(fit_peaks_output_name)
            logger.notice(f"Fitting to spectrum {spec_number:d} without constraining parameters")

        find_peaks_input_params = self._get_find_peak_parameters(spec_number, peak_estimates_list, unconstrained)
        logger.notice(str(spec_number) + "   " + str(find_peaks_input_params))
        peaks_found = self._run_find_peaks(workspace_index, find_peaks_output_name, find_peaks_input_params, unconstrained)

        if peaks_found:
            return self._filter_and_fit_found_peaks(
                workspace_index, peak_estimates_list, find_peaks_output_name, fit_peaks_output_name, x_range, unconstrained
            )
        else:
            return False

    def _filter_and_fit_found_peaks(
        self, workspace_index, peak_estimates_list, find_peaks_output_name, fit_peaks_output_name, x_range, unconstrained
    ):
        if unconstrained:
            linear_bg_coeffs = self._calc_linear_bg_coefficients()
            self._filter_found_peaks(
                find_peaks_output_name,
                peak_estimates_list,
                linear_bg_coeffs,
                peak_height_rel_threshold=EVSGlobals.PEAK_HEIGHT_RELATIVE_THRESHOLD,
            )

        fit_results = self._fit_found_peaks(
            find_peaks_output_name, peak_estimates_list if not unconstrained else None, workspace_index, fit_peaks_output_name, x_range
        )
        fit_results["status"] = (
            "peaks invalid"
            if not self._check_fitted_peak_validity(
                fit_peaks_output_name + "_Parameters",
                peak_estimates_list,
                peak_height_rel_threshold=EVSGlobals.PEAK_HEIGHT_RELATIVE_THRESHOLD,
            )
            else fit_results["status"]
        )
        return fit_results

    def _run_find_peaks(self, workspace_index, find_peaks_output_name, find_peaks_input_params, unconstrained):
        try:
            FindPeaks(
                InputWorkspace=self._sample, WorkspaceIndex=workspace_index, PeaksList=find_peaks_output_name, **find_peaks_input_params
            )
            if mtd[find_peaks_output_name].rowCount() > 0:
                peaks_found = True
            else:
                raise ValueError
        except ValueError:
            peaks_found = False
            if not unconstrained:  # Ignore error if unconstrained, as we will use peaks found during constrained workflow.
                raise ValueError("Error finding peaks.")
        return peaks_found

    def _select_best_fit_params(self, spec_num, fit_results, fit_results_u=None):
        selected_params = fit_results["params"]
        unconstrained_fit_selected = False
        if fit_results_u and fit_results_u["chi2"] < fit_results["chi2"] and fit_results_u["status"] != "peaks invalid":
            selected_params = fit_results_u["params"]
            unconstrained_fit_selected = True
            self._prog_reporter.report(f"Fit to spectrum {spec_num:d} without constraining parameters successful")
        return selected_params, unconstrained_fit_selected

    def _output_params_to_table(self, spec_num, num_estimated_peaks, params, output_table_name):
        fit_values = dict(zip(params.column(0), params.column(1)))
        fit_errors = dict(zip(params.column(0), params.column(2)))

        row_values = [fit_values["f0.A0"], fit_values["f0.A1"]]
        row_errors = [fit_errors["f0.A0"], fit_errors["f0.A1"]]
        param_names = self._get_param_names(num_estimated_peaks)
        row_values += [fit_values["f1." + name] for name in param_names[2:]]
        row_errors += [fit_errors["f1." + name] for name in param_names[2:]]
        row = [element for tupl in zip(row_values, row_errors) for element in tupl]

        mtd[output_table_name].addRow([spec_num] + row)

    def _get_output_and_clean_workspaces(
        self,
        unconstrained_fit_performed,
        unconstrained_peaks_found,
        unconstrained_fit_selected,
        find_peaks_output_name,
        fit_peaks_output_name,
    ):
        find_peaks_output_name_u = self._get_unconstrained_ws_name(find_peaks_output_name)
        fit_peaks_output_name_u = self._get_unconstrained_ws_name(fit_peaks_output_name)

        DeleteWorkspace(fit_peaks_output_name + "_NormalisedCovarianceMatrix")
        DeleteWorkspace(fit_peaks_output_name + "_Parameters")
        DeleteWorkspace(find_peaks_output_name)

        output_workspace = fit_peaks_output_name + "_Workspace"
        if unconstrained_fit_performed:
            # Error in FindPeaks was causing this workspace to be removed from ads already
            with suppress(Exception):
                DeleteWorkspace(find_peaks_output_name_u)

            if unconstrained_peaks_found:
                DeleteWorkspace(fit_peaks_output_name_u + "_NormalisedCovarianceMatrix")
                DeleteWorkspace(fit_peaks_output_name_u + "_Parameters")

                if unconstrained_fit_selected:
                    output_workspace = fit_peaks_output_name_u + "_Workspace"
                    DeleteWorkspace(fit_peaks_output_name + "_Workspace")
                else:
                    DeleteWorkspace(fit_peaks_output_name_u + "_Workspace")
        return output_workspace

    def _calc_linear_bg_coefficients(self):
        temp_bg_fit = Fit(
            Function="(name=LinearBackground,A0=0,A1=0)", InputWorkspace=self._sample, Output="temp_bg_fit", CreateOutput=True
        )
        linear_bg_coeffs = temp_bg_fit.OutputParameters.cell("Value", 0), temp_bg_fit.OutputParameters.cell("Value", 1)
        DeleteWorkspace("temp_bg_fit_Workspace")
        DeleteWorkspace("temp_bg_fit_NormalisedCovarianceMatrix")
        DeleteWorkspace("temp_bg_fit_Parameters")
        return linear_bg_coeffs

    def _check_fitted_peak_validity(self, table_name, estimated_peaks, peak_height_abs_threshold=0.0, peak_height_rel_threshold=0.0):
        check_nans = self._check_nans(table_name)
        check_peak_positions = self._check_peak_positions(table_name, estimated_peaks)
        check_peak_heights = self._check_peak_heights(table_name, peak_height_abs_threshold, peak_height_rel_threshold)

        return bool(check_nans and check_peak_positions and check_peak_heights)

    def _check_nans(self, table_name):
        table_ws = mtd[table_name]
        for i in table_ws.column("Value"):
            if np.isnan(i):
                print("nan found in value common, indicates invalid peak")
                return False
        return True

    def _check_peak_positions(self, table_name, estimated_positions):
        pos_str = self._func_param_names["Position"]

        i = 0
        invalid_positions = []
        for name, value in zip(mtd[table_name].column("Name"), mtd[table_name].column("Value")):
            if pos_str in name:
                if not (
                    value >= estimated_positions[i] - EVSGlobals.BRAGG_PEAK_POSITION_TOLERANCE
                    and value <= estimated_positions[i] + EVSGlobals.BRAGG_PEAK_POSITION_TOLERANCE
                ):
                    invalid_positions.append(value)
                i += 1

        if len(invalid_positions) > 0:
            print(f"Invalid peak positions found: {invalid_positions}")
            return False
        else:
            return True

    def _evaluate_peak_height_against_bg(self, height, position, linear_bg_A0, linear_bg_A1, rel_threshold, abs_threshold):
        bg = position * linear_bg_A1 + linear_bg_A0
        required_height = bg * rel_threshold + abs_threshold
        return not height < required_height

    def _check_peak_heights(self, table_name, abs_threshold_over_bg, rel_threshold_over_bg):
        height_str = self._func_param_names["Height"]
        pos_str = self._func_param_names["Position"]

        peak_heights = []
        peak_positions = []
        for name, value in zip(mtd[table_name].column("Name"), mtd[table_name].column("Value")):
            if height_str in name:
                peak_heights.append(value)
            elif pos_str in name:
                peak_positions.append(value)
            elif name == "f0.A0":
                linear_bg_A0 = value
            elif name == "f0.A1":
                linear_bg_A1 = value

        for height, pos in zip(peak_heights, peak_positions):
            if not self._evaluate_peak_height_against_bg(
                height, pos, linear_bg_A0, linear_bg_A1, abs_threshold_over_bg, rel_threshold_over_bg
            ):
                print(f"Peak height threshold not met. Found height: {height}")
                return False
        return True

    def _filter_found_peaks(
        self, find_peaks_output_name, peak_estimates_list, linear_bg_coeffs, peak_height_abs_threshold=0.0, peak_height_rel_threshold=0.0
    ):
        unfiltered_fits_ws_name = find_peaks_output_name + "_unfiltered"
        CloneWorkspace(InputWorkspace=mtd[find_peaks_output_name], OutputWorkspace=unfiltered_fits_ws_name)

        peak_estimate_deltas = []
        linear_bg_A0, linear_bg_A1 = linear_bg_coeffs
        # with estimates for spectrum, loop through all peaks, get and store delta from peak estimates
        for peak_estimate_index, peak_estimate in enumerate(peak_estimates_list):
            for position_index, (position, height) in enumerate(
                zip(mtd[unfiltered_fits_ws_name].column(2), mtd[unfiltered_fits_ws_name].column(1))
            ):
                if position != 0 and self._evaluate_peak_height_against_bg(
                    height, position, linear_bg_A0, linear_bg_A1, peak_height_abs_threshold, peak_height_rel_threshold
                ):
                    peak_estimate_deltas.append((peak_estimate_index, position_index, abs(position - peak_estimate)))

        # loop through accendings delta, assign peaks until there are none left to be assigned.
        peak_estimate_deltas.sort(key=partial(self._get_x_elem, elem=2))
        index_matches = []
        for peak_estimate_index, position_index, delta in peak_estimate_deltas:
            # if all estimated peaks matched, break
            if len(index_matches) == len(peak_estimates_list):
                break
            # assign match for smallest delta if that position or estimate has not been already matched
            if (
                position_index not in [i[1] for i in index_matches]
                and peak_estimate_index not in [i[0] for i in index_matches]
                and all(x > position_index for x in [i[1] for i in index_matches if i[0] > peak_estimate_index])
                and all(x < position_index for x in [i[1] for i in index_matches if i[0] < peak_estimate_index])
            ):
                index_matches.append((peak_estimate_index, position_index))

        if len(index_matches) > 0:
            index_matches.sort(key=partial(self._get_x_elem, elem=1))
            mtd[find_peaks_output_name].setRowCount(len(peak_estimates_list))
            for col_index in range(mtd[find_peaks_output_name].columnCount()):
                match_n = 0
                for row_index in range(mtd[find_peaks_output_name].rowCount()):
                    if row_index in [x[0] for x in index_matches]:
                        position_row_index = index_matches[match_n][1]
                        mtd[find_peaks_output_name].setCell(
                            row_index, col_index, mtd[unfiltered_fits_ws_name].cell(position_row_index, col_index)
                        )
                        match_n += 1
                    else:  # otherwise just use estimate position
                        pos_str = self._func_param_names["Position"]
                        mtd[find_peaks_output_name].setCell(pos_str, row_index, peak_estimates_list[row_index])
        DeleteWorkspace(unfiltered_fits_ws_name)

    def _get_x_elem(self, input_list, elem):
        return input_list[elem]

    def _fit_found_peaks(self, peak_table, peak_estimates_list, workspace_index, fit_output_name, xLimits=None):
        # get parameters from the peaks table
        peak_params = []
        prefix = ""  # 'f0.'
        position = prefix + self._func_param_names["Position"]

        if peak_estimates_list is not None:  # If no peak estimates list, we are doing an unconstrained fit
            # Don't yet understand what this is doing here
            EVSMiscFunctions.set_table_column(peak_table, position, peak_estimates_list, spec_list=None)
            unconstrained = False
        else:
            unconstrained = True

        for peak_index in range(mtd[peak_table].rowCount()):
            peak_params.append(mtd[peak_table].row(peak_index))

        # build function string
        func_string = self._build_multiple_peak_function(peak_params, workspace_index, unconstrained)

        # select min and max x range for fitting
        positions = [params[position] for params in peak_params]

        if len(positions) < 1:
            raise RuntimeError("No position parameter provided.")

        if xLimits:
            xmin, xmax = xLimits
        else:
            xmin = min(peak_estimates_list) - EVSGlobals.BRAGG_PEAK_POSITION_TOLERANCE - self._fit_window_range
            xmax = max(peak_estimates_list) + EVSGlobals.BRAGG_PEAK_POSITION_TOLERANCE + self._fit_window_range

        # fit function to workspace
        fit_result = (
            Fit(
                Function=func_string,
                InputWorkspace=self._sample,
                IgnoreInvalidData=True,
                StartX=xmin,
                EndX=xmax,
                WorkspaceIndex=workspace_index,
                CalcErrors=True,
                Output=fit_output_name,
                Minimizer="Levenberg-Marquardt,AbsError=0,RelError=1e-8",
            )
            + (xmin,)
            + (xmax,)
        )
        fit_result_key = "status", "chi2", "ncm", "params", "fws", "func", "cost_func", "xmin", "xmax"
        fit_results_dict = dict(zip(fit_result_key, fit_result))
        return fit_results_dict

    def _fit_peaks(self):
        """
        Fit peaks to time of flight data.

        This uses the Mantid algorithms FindPeaks and Fit. Estimates for the centre of the peaks
        are found using either the energy or d-spacings provided. It creates a group workspace
        containing one table workspace per peak with parameters for each detector.
        """

        estimated_peak_positions_all_peaks = self._estimate_peak_positions()
        num_estimated_peaks, num_spectra = estimated_peak_positions_all_peaks.shape

        self._prog_reporter = Progress(self, 0.0, 1.0, num_estimated_peaks * num_spectra)

        self._output_parameter_tables = []
        self._peak_fit_workspaces = []

        for peak_index, estimated_peak_positions in enumerate(estimated_peak_positions_all_peaks):
            self._peak_fit_workspaces_by_spec = []
            output_parameter_table_name = f"{self._output_workspace_name}_Peak_{peak_index:d}_Parameters"
            output_parameter_table_headers = self._create_parameter_table_and_output_headers(output_parameter_table_name)
            for spec_index, peak_position in enumerate(estimated_peak_positions):
                fit_workspace_name = self._fit_peak(
                    peak_index, spec_index, peak_position, output_parameter_table_name, output_parameter_table_headers
                )
                self._peak_fit_workspaces_by_spec.append(fit_workspace_name)

            self._output_parameter_tables.append(output_parameter_table_name)
            self._peak_fit_workspaces.append(self._peak_fit_workspaces_by_spec)

        GroupWorkspaces(self._output_parameter_tables, OutputWorkspace=self._output_workspace_name + "_Peak_Parameters")

        if self._shared_parameter_fit_type != SharedParameterFitType.INDIVIDUAL.value:
            self._shared_parameter_fit(output_parameter_table_name, output_parameter_table_headers)

    def _shared_parameter_fit(self, output_parameter_table_name, output_parameter_table_headers):
        init_Gaussian_FWHM = EVSMiscFunctions.read_table_column(output_parameter_table_name, "f1.GaussianFWHM", self._spec_list)
        init_Gaussian_FWHM = np.nanmean(init_Gaussian_FWHM[init_Gaussian_FWHM != 0])
        init_Lorentz_FWHM = EVSMiscFunctions.read_table_column(output_parameter_table_name, "f1.LorentzFWHM", self._spec_list)
        init_Lorentz_FWHM = np.nanmean(init_Lorentz_FWHM[init_Lorentz_FWHM != 0])
        init_Lorentz_Amp = EVSMiscFunctions.read_table_column(output_parameter_table_name, "f1.LorentzAmp", self._spec_list)
        init_Lorentz_Amp = np.nanmean(init_Lorentz_Amp[init_Lorentz_Amp != 0])
        init_Lorentz_Pos = EVSMiscFunctions.read_table_column(output_parameter_table_name, "f1.LorentzPos", self._spec_list)
        init_Lorentz_Pos = np.nanmean(init_Lorentz_Pos[init_Lorentz_Pos != 0])

        initial_params = {
            "A0": 0.0,
            "A1": 0.0,
            "LorentzAmp": init_Lorentz_Amp,
            "LorentzPos": init_Lorentz_Pos,
            "LorentzFWHM": init_Lorentz_FWHM,
            "GaussianFWHM": init_Gaussian_FWHM,
        }

        self._invalid_detectors.identify_and_set_invalid_detectors_from_range(self._spec_list, output_parameter_table_name)

        self._fit_shared_parameter(
            self._invalid_detectors.get_invalid_detectors_index(self._spec_list), initial_params, output_parameter_table_headers
        )

    def _fit_peak(self, peak_index, spec_index, peak_position, output_parameter_table_name, output_parameter_table_headers):
        spec_number = self._spec_list[0] + spec_index
        self._prog_reporter.report(f"Fitting peak {peak_index:d} to spectrum {spec_number:d}")

        peak_params = self._find_peaks_and_output_params(peak_index, spec_number, spec_index, peak_position)
        fit_func_string = self._build_function_string(peak_params)
        xmin, xmax = self._find_fit_x_window(peak_params)
        fit_output_name = f"__{self._output_workspace_name}_Peak_{peak_index:d}_Spec_{spec_index:d}"
        _status, _chi2, ncm, fit_params, fws, _func, _cost_func = Fit(
            Function=fit_func_string,
            InputWorkspace=self._sample,
            IgnoreInvalidData=True,
            StartX=xmin,
            EndX=xmax,
            WorkspaceIndex=spec_index,
            CalcErrors=True,
            Output=fit_output_name,
            Minimizer="Levenberg-Marquardt,RelError=1e-8",
        )

        self._output_fit_params_to_table_ws(spec_number, fit_params, output_parameter_table_name, output_parameter_table_headers)
        fit_workspace_name = fws.name()
        self._del_fit_workspaces(ncm, fit_params, fws)

        return fit_workspace_name

    def _find_peaks_and_output_params(self, peak_index, spec_number, spec_index, peak_position):
        peak_table_name = f"__{self._sample}_peaks_table_{peak_index:d}_{spec_index:d}"
        find_peak_params = self._get_find_peak_parameters(spec_number, [peak_position])
        FindPeaks(InputWorkspace=self._sample, WorkspaceIndex=spec_index, PeaksList=peak_table_name, **find_peak_params)
        if mtd[peak_table_name].rowCount() == 0:
            logger.error("FindPeaks could not find any peaks matching the parameters:\n" + str(find_peak_params))
            sys.exit()

        peak_params = mtd[peak_table_name].row(0)
        DeleteWorkspace(peak_table_name)
        return peak_params

    def _find_fit_x_window(self, peak_params):
        xmin, xmax = None, None

        position = "" + self._func_param_names["Position"]
        if peak_params[position] > 0:
            xmin = peak_params[position] - self._fit_window_range
            xmax = peak_params[position] + self._fit_window_range
        else:
            logger.warning("Could not specify fit window. Using full spectrum x range.")
        return xmin, xmax

    def _output_fit_params_to_table_ws(self, spec_num, params, output_table_name, output_table_headers):
        fit_values = dict(zip(params.column(0), params.column(1)))
        fit_errors = dict(zip(params.column(0), params.column(2)))
        row_values = [fit_values[name] for name in output_table_headers]
        row_errors = [fit_errors[name] for name in output_table_headers]
        row = [element for tupl in zip(row_values, row_errors) for element in tupl]

        mtd[output_table_name].addRow([spec_num] + row)

    def _del_fit_workspaces(self, ncm, fit_params, fws):
        DeleteWorkspace(ncm)
        DeleteWorkspace(fit_params)
        if not self._create_output:
            DeleteWorkspace(fws)

    def _fit_shared_parameter(self, invalid_spectra, initial_params, param_names):
        """
        Fit peaks to all detectors, with one set of fit parameters for all detectors.
        """

        shared_peak_table = self._output_workspace_name + "_shared_peak_parameters"
        CreateEmptyTableWorkspace(OutputWorkspace=shared_peak_table)

        err_names = [name + "_Err" for name in param_names]
        col_names = [element for tupl in zip(param_names, err_names) for element in tupl]

        mtd[shared_peak_table].addColumn("int", "Spectrum")
        for name in col_names:
            mtd[shared_peak_table].addColumn("double", name)

        fit_func = self._build_function_string(initial_params)
        start_str = "composite=MultiDomainFunction, NumDeriv=1;"
        sample_ws = mtd[self._sample]
        MaskDetectors(Workspace=sample_ws, WorkspaceIndexList=invalid_spectra)
        validSpecs = ExtractUnmaskedSpectra(InputWorkspace=sample_ws, OutputWorkspace="valid_spectra")
        n_valid_specs = validSpecs.getNumberHistograms()

        fit_func = ("(composite=CompositeFunction, NumDeriv=false, $domains=i;" + fit_func[:-1] + ");") * n_valid_specs
        composite_func = start_str + fit_func[:-1]

        ties = ",".join(f"f{i}.f1.{p}=f0.f1.{p}" for p in self._func_param_names.values() for i in range(1, n_valid_specs))
        func_string = composite_func + f";ties=({ties})"

        fit_output_name = "__" + self._output_workspace_name + "_Peak_0"

        xmin, xmax = None, None

        # create new workspace for each spectra
        x = validSpecs.readX(0)
        y = validSpecs.readY(0)
        e = validSpecs.readE(0)
        out_ws = self._sample + "_Spec_0"
        CreateWorkspace(DataX=x, DataY=y, DataE=e, NSpec=1, OutputWorkspace=out_ws)

        other_inputs = [
            CreateWorkspace(
                DataX=validSpecs.readX(j), DataY=validSpecs.readY(j), DataE=validSpecs.readE(j), OutputWorkspace=f"{self._sample}_Spec_{j}"
            )
            for j in range(1, n_valid_specs)
        ]

        added_args = {f"InputWorkspace_{i + 1}": v for i, v in enumerate(other_inputs)}

        print("starting global fit")

        _status, _chi2, ncm, params, fws, _func, _cost_func = Fit(
            Function=func_string,
            InputWorkspace=out_ws,
            IgnoreInvalidData=True,
            StartX=xmin,
            EndX=xmax,
            CalcErrors=True,
            Output=fit_output_name,
            Minimizer="SteepestDescent,RelError=1e-8",
            **added_args,
        )
        [DeleteWorkspace(f"{self._sample}_Spec_{i}") for i in range(n_valid_specs)]

        fit_values = dict(zip(params.column(0), params.column(1)))
        fit_errors = dict(zip(params.column(0), params.column(2)))

        row_values = [fit_values["f0." + name] for name in param_names]
        row_errors = [fit_errors["f0." + name] for name in param_names]
        row = [element for tupl in zip(row_values, row_errors) for element in tupl]

        mtd[shared_peak_table].addRow([0] + row)

        DeleteWorkspace(ncm)
        DeleteWorkspace(params)
        if not self._create_output:
            DeleteWorkspace(fws)

        DeleteWorkspace(validSpecs)

        mtd[self._output_workspace_name + "_Peak_Parameters"].add(shared_peak_table)

    def _get_find_peak_parameters(self, spec_number, peak_centre, unconstrained=False):
        """
        Get find peak parameters

        @param spec_num - the current spectrum number being fitted
        @return dictionary of parameters for find peaks
        """

        find_peak_params = {}
        find_peak_params["PeakFunction"] = self._peak_function
        find_peak_params["RawPeakParameters"] = True
        find_peak_params["BackgroundType"] = "Linear"

        if not unconstrained:
            find_peak_params["PeakPositions"] = peak_centre

        if self._fitting_bragg_peaks and not unconstrained:
            find_peak_params["PeakPositionTolerance"] = EVSGlobals.BRAGG_PEAK_POSITION_TOLERANCE

            if spec_number >= EVSGlobals.FRONTSCATTERING_RANGE[0]:
                find_peak_params["FWHM"] = 70
            else:
                find_peak_params["FWHM"] = 5

        elif not self._fitting_bragg_peaks:
            half_peak_window = 60  # just be recoil
            if self._fitting_resonance_peaks:
                # 25 seems to be able to fit the final peak in the first backscattering spectrum
                half_peak_window = 20 if spec_number >= EVSGlobals.FRONTSCATTERING_RANGE[0] else 25

            fit_windows = [[peak - half_peak_window, peak + half_peak_window] for peak in peak_centre]
            # flatten the list: http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
            find_peak_params["FitWindows"] = [peak for pair in fit_windows for peak in pair]

        return find_peak_params

    def _create_parameter_table_and_output_headers(self, peaks_table):
        """
        Create a table workspace with headers corresponding to the parameters
        output from fitting.

        @param peak_table - name to call the parameter table.
        @return param_names - name of the columns
        """
        CreateEmptyTableWorkspace(OutputWorkspace=peaks_table)

        param_names = ["f0.A0", "f0.A1"]
        param_names += ["f1." + name for name in self._func_param_names.values()]

        err_names = [name + "_Err" for name in param_names]
        col_names = [element for tupl in zip(param_names, err_names) for element in tupl]

        mtd[peaks_table].addColumn("int", "Spectrum")
        for name in col_names:
            mtd[peaks_table].addColumn("double", name)

        return param_names

    def _estimate_peak_positions(self):
        """
        Estimate the position of a peak based on a given energy and previous calibrated parameters.

        @return list of estimates in tof for the peak centres
        """
        kwargs = {
            "table_name": self._param_table,
            "spec_list": self._spec_list,
            "offset": EVSGlobals.DETECTOR_RANGE[0],
        }
        L0 = EVSMiscFunctions.read_table_column(column_name="L0", **kwargs)
        L1 = EVSMiscFunctions.read_table_column(column_name="L1", **kwargs)
        t0 = EVSMiscFunctions.read_table_column(column_name="t0", **kwargs)
        t0 /= 1e6
        thetas = EVSMiscFunctions.read_table_column(column_name="theta", **kwargs)
        r_theta = EVSMiscFunctions.calculate_r_theta(self._sample_mass, thetas)

        self._energy_estimates = self._energy_estimates.reshape(1, self._energy_estimates.size).T
        self._energy_estimates *= EVSGlobals.MEV_CONVERSION

        v1 = np.sqrt(2 * self._energy_estimates / scipy.constants.m_n)
        if self._fitting_recoil_peaks:
            # for recoil peaks:
            tof = ((L0 * r_theta + L1) / v1) + t0
        elif self._fitting_resonance_peaks:
            # for resonances:
            tof = (L0 / v1) + t0
        else:
            raise RuntimeError("Cannot estimate peak positions for unknown peak type")

        tof *= 1e6
        return tof

    def _estimate_bragg_peak_positions(self):
        """
        Estimate the position of bragg peaks in TOF using estimates of the parameters
        and d-spacings of the sample.

        The number of peaks will depend on the number of d-spacings provided.

        @return estimates of peak positions in TOF for all spectra
        """
        kwargs = {
            "table_name": self._param_table,
            "spec_list": self._spec_list,
            "offset": EVSGlobals.DETECTOR_RANGE[0],
        }
        L0 = EVSMiscFunctions.read_table_column(column_name="L0", **kwargs)
        L1 = EVSMiscFunctions.read_table_column(column_name="L1", **kwargs)
        t0 = EVSMiscFunctions.read_table_column(column_name="t0", **kwargs)
        thetas = EVSMiscFunctions.read_table_column(column_name="theta", **kwargs)

        t0 /= 1e6

        self._d_spacings *= 1e-10
        self._d_spacings = self._d_spacings.reshape(1, self._d_spacings.size).T

        lambdas = 2 * self._d_spacings * np.sin(np.radians(thetas) / 2)

        L1_nan_to_num = np.nan_to_num(L1)
        tof = (lambdas * scipy.constants.m_n * (L0 + L1_nan_to_num)) / scipy.constants.h + t0
        tof *= 1e6

        return tof

    def _sum_and_crop_ws(self, group_ws, output_name, xmin, xmax):
        """
        Process the workspaces and sum each of the runs and crop the output.

        @param group_ws - A group workspace with the runs loaded.
        @param output_name - name to call the final workspace.
        @param xmin - The lower limit for the crop.
        @param xmax - The upper limit for the crop.
        """
        for ws_ix in range(group_ws.size()):
            if ws_ix == 0:
                CloneWorkspace(InputWorkspace=group_ws[ws_ix], OutputWorkspace=output_name)
                temp_ws_name = "__EVS_calib_temp_ws"
            else:
                CloneWorkspace(InputWorkspace=group_ws[ws_ix], OutputWorkspace=temp_ws_name)
                Plus(output_name, temp_ws_name, OutputWorkspace=output_name)
                DeleteWorkspace(temp_ws_name)
        CropWorkspace(output_name, XMin=xmin, XMax=xmax, OutputWorkspace=output_name)

    def _build_linear_background_function(self, peak, tie_gradient=False):
        """
        Builds a string describing a peak function that can be passed to Fit.
        This will be either a Gaussian or Lorentzian shape.

        @param peaks - dictionary containing the parameters for the function
        @return the function string
        """

        bkg_intercept, bkg_gradient = peak["A0"], peak["A1"]
        fit_func = f"name=LinearBackground, A0={bkg_intercept:f}, A1={bkg_gradient:f}"
        fit_func += ";"

        return fit_func

    def _build_peak_function(self, peak):
        """
        Builds a string describing a linear background function that can be passed to Fit.

        @param peaks - dictionary containing the parameters for the function
        @return the function string
        """

        fit_func = f"name={self._peak_function}, "

        prefix = ""  # f0.'
        func_list = [name + "=" + str(peak[prefix + name]) for name in self._func_param_names.values()]
        fit_func += ", ".join(func_list) + ";"

        return fit_func

    def _build_multiple_peak_function(self, peaks, workspace_index, unconstrained):
        """
        Builds a string describing a composite function that can be passed to Fit.
        This will be a linear background with multiple peaks of either a Gaussian or Voigt shape.

        @param peaks - list of dictionaries containing the parameters for the function
        @return the function string
        """

        if len(peaks) == 0:
            return ""

        if not unconstrained:
            fit_func = self._build_linear_background_function(peaks[0], False)
        else:
            pos_str = self._func_param_names["Position"]
            if len(peaks) > 1:
                fit_window = [peaks[0][pos_str] - 1000, peaks[len(peaks) - 1][pos_str] + 1000]
            else:
                fit_window = [peaks[0][pos_str] - 1000, peaks[0][pos_str] + 1000]
            FindPeakBackground(
                InputWorkspace=self._sample,
                WorkspaceIndex=workspace_index,
                FitWindow=fit_window,
                BackgroundType="Linear",
                OutputWorkspace="temp_fit_bg",
            )
            fit_func = f"name=LinearBackground, A0={mtd['temp_fit_bg'].cell('bkg0', 0)}, A1={mtd['temp_fit_bg'].cell('bkg1', 0)};"
            DeleteWorkspace("temp_fit_bg")

        fit_func += "("
        for i, peak in enumerate(peaks):
            fit_func += self._build_peak_function(peak)

        fit_func = fit_func[:-1]
        fit_func += ");"

        return fit_func

    def _build_function_string(self, peak):
        """
        Builds a string describing a composite function that can be passed to Fit.
        This will be a linear background with either a Gaussian or Voigt function.

        @param peak - dictionary containing the parameters for the function
        @return the function string
        """

        if len(peak) == 0:
            return ""

        fit_func = self._build_linear_background_function(peak)
        fit_func += self._build_peak_function(peak)

        return fit_func

    def _generate_fit_workspaces(self):
        """
        Output the fit workspace for each spectrum fitted.
        """
        fit_workspaces = map(list, zip(*self._peak_fit_workspaces))
        group_ws_names = []

        for i, peak_fits in enumerate(fit_workspaces):
            ws_name = f"{self._output_workspace_name}_{i:d}_Workspace"
            ExtractSingleSpectrum(InputWorkspace=self._sample, WorkspaceIndex=i, OutputWorkspace=ws_name)

            # transfer fits to individual spectrum
            peak_collection = WorkspaceFactory.create(mtd[ws_name], NVectors=len(peak_fits))
            data_x = mtd[ws_name].readX(0)

            ax = TextAxis.create(len(peak_fits) + 2)
            ax.setLabel(0, "Data")

            for j, peak in enumerate(peak_fits):
                peak_x = mtd[peak].readX(0)
                peak_y = mtd[peak].readY(1)
                peak_e = mtd[peak].readE(1)

                index = int(np.where(data_x == peak_x[0])[0])
                data_y = peak_collection.dataY(j)
                data_y[index : index + peak_y.size] = peak_y

                data_e = peak_collection.dataE(j)
                data_e[index : index + peak_e.size] = peak_e

                peak_collection.setX(j, data_x)
                peak_collection.setY(j, data_y)
                peak_collection.setE(j, data_e)

                ax.setLabel(j + 1, f"Peak_{j + 1:d}")

                DeleteWorkspace(peak)

            peak_ws = "__tmp_peak_workspace"
            mtd.addOrReplace(peak_ws, peak_collection)
            AppendSpectra(ws_name, peak_ws, ValidateInputs=False, OutputWorkspace=ws_name)

            # create sum of peak fits
            SumSpectra(InputWorkspace=peak_ws, OutputWorkspace=peak_ws)
            AppendSpectra(ws_name, peak_ws, ValidateInputs=False, OutputWorkspace=ws_name)
            ax.setLabel(mtd[ws_name].getNumberHistograms() - 1, "Total")
            DeleteWorkspace(peak_ws)

            mtd[ws_name].replaceAxis(1, ax)
            group_ws_names.append(ws_name)

        GroupWorkspaces(group_ws_names, OutputWorkspace=self._output_workspace_name + "_Peak_Fits")
