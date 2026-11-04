import numpy as np
import scipy
from mantid.kernel import (
    StringListValidator,
    Direction,
    IntArrayBoundedValidator,
    IntArrayProperty,
    IntBoundedValidator,
    FloatBoundedValidator,
)
from mantid.api import FileProperty, FileAction, PythonAlgorithm, MatrixWorkspaceProperty, WorkspaceGroupProperty
from mantid.dataobjects import TableWorkspaceProperty
from mantid.simpleapi import (
    mtd,
    CreateEmptyTableWorkspace,
    SumSpectra,
    CloneWorkspace,
    DeleteWorkspace,
    VesuvioCalculateGammaBackground,
    VesuvioCalculateMS,
    Scale,
    RenameWorkspace,
    Minus,
    VesuvioThickness,
    Integration,
    Divide,
    Multiply,
    DeleteWorkspaces,
    CreateWorkspace,
    GroupWorkspaces,
)

from mvesuvio.util.analysis_helpers import (
    numerical_third_derivative,
    load_resolution,
    load_instrument_params,
    extend_range_of_array,
    make_gamma_correction_input_string,
    make_multiple_scattering_input_string,
    pseudo_voigt,
)
from mvesuvio.util.general_helpers import print_table_workspace
from mvesuvio.globals import ModeRunning, ScatteringType
from mvesuvio.util.constraints_transport import deserialize_constraints

np.set_printoptions(suppress=True, precision=4, linewidth=200)

NEUTRON_MASS = 1.008  # a.m.u.
ENERGY_FINAL = 4906.0  # meV
ENERGY_TO_VELOCITY = 4.3737 * 1.0e-4
VELOCITY_FINAL = np.sqrt(ENERGY_FINAL) * ENERGY_TO_VELOCITY  # m/us
H_BAR = 2.0445


class VesuvioAnalysisRoutine(PythonAlgorithm):
    def summary(self):
        return "Runs the analysis reduction routine for VESUVIO."

    def category(self):
        return "VesuvioAnalysis"

    def PyInit(self):
        self.declareProperty(
            MatrixWorkspaceProperty(name="InputWorkspace", defaultValue="", direction=Direction.Input),
            doc="Workspace to fit Neutron Compton Profiles.",
        )
        self.declareProperty(
            TableWorkspaceProperty(name="InputProfiles", defaultValue="", direction=Direction.Input),
            doc="Table workspace containing starting parameters for profiles.",
        )
        self.declareProperty(
            FileProperty(name="InstrumentParametersFile", defaultValue="", action=FileAction.Load, extensions=["par", "dat"]),
            doc="Filename of the instrument parameter file.",
        )
        self.declareProperty(
            name="ChosenMassIndex",
            defaultValue=0,
            validator=IntBoundedValidator(lower=0),
            doc="Index of mass that anchors hydrogen ratio provided by the user.",
        )
        self.declareProperty(
            name="HRatioToChosenMass",
            defaultValue=0.0,
            validator=FloatBoundedValidator(lower=0),
            doc="Intensity ratio between H peak and chosen mass peak. Traditionally estimated using stoichiometry values.",
        )
        self.declareProperty(name="NumberOfIterations", defaultValue=2, validator=IntBoundedValidator(lower=0))
        self.declareProperty(
            IntArrayProperty(name="InvalidDetectors", validator=IntArrayBoundedValidator(lower=3, upper=198), direction=Direction.Input),
            doc="List of invalid detectors whithin range 3-198.",
        )
        self.declareProperty(name="MultipleScatteringCorrection", defaultValue=False, doc="Whether to run multiple scattering correction.")
        self.declareProperty(name="GammaCorrection", defaultValue=False, doc="Whether to run gamma correction.")
        self.declareProperty(
            name="ModeRunning",
            defaultValue=ModeRunning.BACKWARD.value,
            validator=StringListValidator([mode.value for mode in ModeRunning]),
            doc="Whether running backward or forward scattering.",
        )

        self.declareProperty(name="Constraints", defaultValue="", doc="Constraints to use during fitting profiles.")
        self.declareProperty(name="TransmissionGuess", defaultValue=-1.0, validator=FloatBoundedValidator(lower=0, upper=1))
        self.declareProperty(name="MultipleScatteringOrder", defaultValue=-1, validator=IntBoundedValidator(lower=0))
        self.declareProperty(name="NumberOfEvents", defaultValue=-1, validator=IntBoundedValidator(lower=0))
        # Outputs
        self.declareProperty(
            TableWorkspaceProperty(name="OutputMeansTable", defaultValue="", direction=Direction.Output),
            doc="TableWorkspace containing final means of intensity and widths.",
        )
        self.declareProperty(
            WorkspaceGroupProperty(name="OutputNCPGroup", defaultValue="ncp", direction=Direction.Output),
            doc="GroupWorkspace containing Neutron Compton Profiles for each mass and sum.",
        )
        self.declareProperty(
            WorkspaceGroupProperty(name="OutputFSEGroup", defaultValue="fse", direction=Direction.Output),
            doc="GroupWorkspace containing fitted Final State Effects for each mass.",
        )

    def PyExec(self):
        self._setup()
        self.run()

    def _setup(self):
        self._name = self.getPropertyValue("InputWorkspace")
        self._ip_file = self.getProperty("InstrumentParametersFile").value
        self._number_of_iterations = self.getProperty("NumberOfIterations").value
        self._mask_spectra = self.getProperty("InvalidDetectors").value
        self._transmission_guess = self.getProperty("TransmissionGuess").value
        self._multiple_scattering_order = self.getProperty("MultipleScatteringOrder").value
        self._number_of_events = self.getProperty("NumberOfEvents").value
        self._mode_running = self.getProperty("ModeRunning").value
        self._multiple_scattering_correction = self.getProperty("MultipleScatteringCorrection").value
        self._gamma_correction = self.getProperty("GammaCorrection").value
        self._chosen_index_for_h_ratio = self.getProperty("ChosenMassIndex").value
        self._h_ratio = self.getProperty("HRatioToChosenMass").value
        self._constraints = deserialize_constraints(self.getProperty("Constraints").value)
        self._profiles_table = self.getProperty("InputProfiles").value

        self._instrument_params = load_instrument_params(self._ip_file, self.getProperty("InputWorkspace").value.getSpectrumNumbers())
        self._resolution_params = load_resolution(self._instrument_params)

        # Need to transform profiles table into parameter array for optimize.minimize()
        self._initial_fit_parameters = []
        for intensity, width, center in zip(
            self._profiles_table.column("intensity"), self._profiles_table.column("width"), self._profiles_table.column("center")
        ):
            self._initial_fit_parameters.extend([intensity, width, center])

        self._initial_fit_bounds = []

        for intensity_lb, intensity_ub, width_lb, width_ub, center_lb, center_ub in zip(
            self._profiles_table.column("intensity_lb"),
            self._profiles_table.column("intensity_ub"),
            self._profiles_table.column("width_lb"),
            self._profiles_table.column("width_ub"),
            self._profiles_table.column("center_lb"),
            self._profiles_table.column("center_ub"),
        ):
            self._initial_fit_bounds.extend([[intensity_lb, intensity_ub], [width_lb, width_ub], [center_lb, center_ub]])

        # Masses need to be defined in the same order
        self._masses = np.array(self._profiles_table.column("mass"))

        # Variables changing during fit
        self._workspace_for_corrections = self.getProperty("InputWorkspace").value
        self._workspace_being_fit = self.getProperty("InputWorkspace").value
        self._row_being_fit = 0
        self._zero_columns_boolean_mask = None
        self._table_fit_results = None
        self._fit_profiles_workspaces = {}

        assert self._chosen_index_for_h_ratio < self._masses.size, "Index of chosen mass out of range."

    def _update_workspace_data(self):
        self._dataX = self._workspace_being_fit.extractX()
        self._dataY = self._workspace_being_fit.extractY()
        self._dataE = self._workspace_being_fit.extractE()

        self._set_kinematic_arrays(self._dataX)
        self._set_gaussian_resolution()
        self._set_lorentzian_resolution()
        self._set_y_space_arrays()

        self._fit_parameters = np.zeros((len(self._dataY), 3 * self._profiles_table.rowCount() + 2))
        self._row_being_fit = 0
        self._table_fit_results = self._initialize_table_fit_parameters()

        # Initialise workspaces for fitted ncp
        self._fit_profiles_workspaces = {}
        ws_ncp_group = self._initialize_and_group_workspaces("ncp", self._fit_profiles_workspaces)
        self.setPropertyValue("OutputNCPGroup", ws_ncp_group.name())

        # Initialise workspaces for fitted fse
        self._fit_fse_workspaces = {}
        ws_fse_group = self._initialize_and_group_workspaces("fse", self._fit_fse_workspaces)
        self.setPropertyValue("OutputFSEGroup", ws_fse_group.name())

        # Initialise empty means
        self._mean_widths = np.zeros(self._masses.size)
        self._std_widths = np.zeros(self._masses.size)
        self._mean_intensity_ratios = np.zeros(self._masses.size)
        self._std_intensity_ratios = np.zeros(self._masses.size)

    def _initialize_and_group_workspaces(self, suffix, result_dict):
        # Helper to initialize either ncp of fse workspaces and group them together
        assert not result_dict  # Check dict is empty
        for element in self._profiles_table.column("label"):
            result_dict[element] = self._create_emtpy_ncp_workspace(f"_{element}_{suffix}")
        result_dict["total"] = self._create_emtpy_ncp_workspace(f"_total_{suffix}")

        empty_sum_ws = [SumSpectra(ws, OutputWorkspace=ws.name() + "_sum") for ws in result_dict.values()]
        ws_to_group = list(result_dict.values()) + empty_sum_ws
        return GroupWorkspaces(ws_to_group, OutputWorkspace=self._workspace_being_fit.name() + f"_{suffix}_group")

    def _initialize_table_fit_parameters(self):
        table = CreateEmptyTableWorkspace(OutputWorkspace=self._workspace_being_fit.name() + "_fit_results")
        table.setTitle("SciPy Fit Parameters")
        table.addColumn(type="float", name="Spec")
        for label in self._profiles_table.column("label"):
            table.addColumn(type="float", name=f"{label} i")
            table.addColumn(type="float", name=f"{label} w")
            table.addColumn(type="float", name=f"{label} c")
        table.addColumn(type="float", name="NChi2")
        return table

    def _create_emtpy_ncp_workspace(self, suffix):
        return CreateWorkspace(
            DataX=self._dataX,
            DataY=np.zeros(self._dataY.size),
            DataE=np.zeros(self._dataE.size),
            Nspec=self._workspace_being_fit.getNumberHistograms(),
            UnitX="TOF",  # I had hoped for a method like .XUnit() but alas
            OutputWorkspace=self._workspace_being_fit.name() + suffix,
            ParentWorkspace=self._workspace_being_fit,
            Distribution=True,
        )

    def run(self):
        assert self._profiles_table.rowCount() > 0, "Need at least one profile to run the routine!"

        CloneWorkspace(InputWorkspace=self._workspace_being_fit.name(), OutputWorkspace=self._name + "_0")

        for iteration in range(self._number_of_iterations + 1):
            self._workspace_being_fit = mtd[self._name + "_" + str(iteration)]
            self._update_workspace_data()

            self._fit_neutron_compton_profiles()

            self._calculate_summed_workspaces()
            self._set_means_and_std()

            # When last iteration, skip MS and GC
            if iteration == self._number_of_iterations:
                break

            # Do this because MS and Gamma corrections do not accept zero columns
            if iteration == 0:
                self._replace_zeros_with_ncp_for_corrections()

            CloneWorkspace(InputWorkspace=self._workspace_for_corrections.name(), OutputWorkspace="next_iteration")
            self._correct_for_gamma_and_multiple_scattering("next_iteration")

            # Need to remask columns of output of corrections
            self._remask_columns_with_zeros("next_iteration")

            RenameWorkspace(InputWorkspace="next_iteration", OutputWorkspace=self._name + "_" + str(iteration + 1))

        return self

    def _fit_neutron_compton_profiles(self):
        """
        Performs the fit of neutron compton profiles to the workspace being fit.
        The profiles are fit on a spectrum by spectrum basis.
        """
        self.log().notice("\nFitting neutron compton profiles ...\n")

        self._row_being_fit = 0
        while self._row_being_fit != len(self._dataY):
            self._fit_neutron_compton_profiles_to_row()
            self._row_being_fit += 1

        assert np.any(self._fit_parameters), "Fitting parameters cannot be zero for all spectra!"
        print_table_workspace(self._table_fit_results)
        return

    def _set_kinematic_arrays(self, dataX):
        # Extend range due to third derivative cutting edges
        dataX = extend_range_of_array(dataX, 6)

        det, plick, angle, T0, L0, L1 = np.hsplit(self._instrument_params, 6)  # each is of len(dataX)

        # T0 is electronic delay due to instruments
        t_us = dataX - T0
        self._v0 = VELOCITY_FINAL * L0 / (VELOCITY_FINAL * t_us - L1)
        # en_to_vel is a factor used to easily change velocity to energy and vice-versa
        self._E0 = np.square(self._v0 / ENERGY_TO_VELOCITY)
        self._deltaE = self._E0 - ENERGY_FINAL
        delta_Q2 = (
            2.0
            * NEUTRON_MASS
            / H_BAR**2
            * (self._E0 + ENERGY_FINAL - 2.0 * np.sqrt(self._E0 * ENERGY_FINAL) * np.cos(angle / 180.0 * np.pi))
        )
        self._deltaQ = np.sqrt(delta_Q2)
        return

    def _set_y_space_arrays(self):
        delta_Q = self._deltaQ[np.newaxis, :, :]
        delta_E = self._deltaE[np.newaxis, :, :]
        masses = self._masses.reshape(-1, 1, 1)

        energy_recoil = np.square(H_BAR * delta_Q) / 2.0 / masses
        y_spaces = masses / H_BAR**2 / delta_Q * (delta_E - energy_recoil)

        # Swap axis so that first axis selects spectra
        self._y_space_arrays = np.swapaxes(y_spaces, 0, 1)
        return

    def _calculate_summed_workspaces(self):
        SumSpectra(InputWorkspace=self._workspace_being_fit.name(), OutputWorkspace=self._workspace_being_fit.name() + "_sum")

        for ws in self._fit_profiles_workspaces.values():
            SumSpectra(InputWorkspace=ws.name(), OutputWorkspace=ws.name() + "_sum")

        for ws in self._fit_fse_workspaces.values():
            SumSpectra(InputWorkspace=ws.name(), OutputWorkspace=ws.name() + "_sum")

    def _set_means_and_std(self):
        widths = np.zeros((self._profiles_table.rowCount(), self._table_fit_results.rowCount()))
        intensities = np.zeros(widths.shape)

        for i, label in enumerate(self._profiles_table.column("label")):
            widths[i] = self._table_fit_results.column(f"{label} w")
            intensities[i] = self._table_fit_results.column(f"{label} i")
            self._set_means_and_std_arrays(widths, intensities)

        self._create_means_table()
        return

    def _set_means_and_std_arrays(self, widths, intensities):
        # Remove failed fits and masked spectra
        non_zero_columns = np.any(widths != 0, axis=0)
        widths = widths[:, non_zero_columns]
        intensities = intensities[:, non_zero_columns]

        widths_mean = np.mean(widths, axis=1).reshape(-1, 1)
        widths_std = np.std(widths, axis=1).reshape(-1, 1)

        widths_deviations = np.abs(widths - widths_mean)

        # Remove width outliers
        widths[widths_deviations > widths_std] = np.nan
        intensities[widths_deviations > widths_std] = np.nan

        # Use sum instead of nansum to propagate nans
        intensities = intensities / intensities.sum(axis=0)

        self._mean_widths = np.nanmean(widths, axis=1)
        self._std_widths = np.nanstd(widths, axis=1)
        self._mean_intensity_ratios = np.nanmean(intensities, axis=1)
        self._std_intensity_ratios = np.nanstd(intensities, axis=1)
        return

    def _create_means_table(self):
        table = CreateEmptyTableWorkspace(OutputWorkspace=self._workspace_being_fit.name() + "_means")
        table.addColumn(type="str", name="label")
        table.addColumn(type="float", name="mass")
        table.addColumn(type="float", name="mean_width")
        table.addColumn(type="float", name="std_width")
        table.addColumn(type="float", name="mean_intensity")
        table.addColumn(type="float", name="std_intensity")

        for label, mass, mean_width, std_width, mean_intensity, std_intensity in zip(
            self._profiles_table.column("label"),
            self._masses,
            self._mean_widths,
            self._std_widths,
            self._mean_intensity_ratios,
            self._std_intensity_ratios,
        ):
            # Explicit conversion to float required to match profiles table
            table.addRow([label, float(mass), float(mean_width), float(std_width), float(mean_intensity), float(std_intensity)])

        self.setPropertyValue("OutputMeansTable", table.name())
        print_table_workspace(table, precision=5)
        return table

    def _fit_neutron_compton_profiles_to_row(self):
        if np.all(self._dataY[self._row_being_fit] == 0):
            self._table_fit_results.addRow(np.zeros(3 * self._profiles_table.rowCount() + 2))
            spectrum_number = self._instrument_params[self._row_being_fit, 0]
            self.log().notice(f"Skip spectrum {int(spectrum_number):3d}")
            return

        result = scipy.optimize.minimize(
            self._error_function,
            self._initial_fit_parameters,
            method="SLSQP",
            bounds=self._initial_fit_bounds,
            constraints=self._constraints,
            tol=1e-6,
        )
        fitPars = result["x"]

        # Pass fit parameters to results table
        noDegreesOfFreedom = len(self._dataY[self._row_being_fit]) - len(fitPars)
        normalised_chi2 = result["fun"] / noDegreesOfFreedom
        spectrum_number = self._instrument_params[self._row_being_fit, 0]
        tableRow = np.hstack((spectrum_number, fitPars, normalised_chi2))
        self._table_fit_results.addRow(tableRow)

        # Store results for easier access when calculating means
        self._fit_parameters[self._row_being_fit] = tableRow

        self.log().notice(f"Fit spectrum {int(spectrum_number):3d}: \u2713")

        # Pass fit profiles into workspaces
        ncp_for_each_mass, fse_for_each_mass = self._neutron_compton_profiles(fitPars)
        for ncp, fse, element in zip(ncp_for_each_mass, fse_for_each_mass, self._profiles_table.column("label")):
            self._fit_profiles_workspaces[element].dataY(self._row_being_fit)[:] = ncp
            self._fit_fse_workspaces[element].dataY(self._row_being_fit)[:] = fse

        self._fit_profiles_workspaces["total"].dataY(self._row_being_fit)[:] = np.sum(ncp_for_each_mass, axis=0)
        self._fit_fse_workspaces["total"].dataY(self._row_being_fit)[:] = np.sum(fse_for_each_mass, axis=0)
        return

    def _error_function(self, pars):
        """Error function to be minimized, in TOF space"""

        ncp_for_each_mass, fse_for_each_mass = self._neutron_compton_profiles(pars)

        ncp_total = np.sum(ncp_for_each_mass, axis=0)
        data_y = self._dataY[self._row_being_fit]
        data_e = self._dataE[self._row_being_fit]

        # Ignore any masked values on tof range
        nonzero_mask = np.nonzero(data_y)
        ncp_total = ncp_total[nonzero_mask]
        data_y = data_y[nonzero_mask]
        data_e = data_e[nonzero_mask]

        if np.all(data_e == 0):  # When errors not present
            return np.sum((ncp_total - data_y) ** 2)

        return np.sum((ncp_total - data_y) ** 2 / data_e**2)

    def _neutron_compton_profiles(self, pars):
        """
        Neutron Compton Profile distribution on TOF space for a single spectrum.
        Calculated from kinematics, J(y) and resolution functions.
        """
        intensities = pars[::3].reshape(-1, 1)
        widths = pars[1::3].reshape(-1, 1)
        centers = pars[2::3].reshape(-1, 1)
        masses = self._masses.reshape(-1, 1)

        gaussian_width = self._get_gaussian_resolution(centers)
        lorentzian_width = self._get_lorentzian_resolution(centers)
        total_gaussian_width = np.sqrt(widths**2 + gaussian_width**2)

        JOfY = pseudo_voigt(self._y_space_arrays[self._row_being_fit] - centers, total_gaussian_width, lorentzian_width)

        # Third derivative cuts edges of array by 6 indices
        JOfY_third_derivative = numerical_third_derivative(self._y_space_arrays[self._row_being_fit], JOfY)

        deltaQ = self._deltaQ[self._row_being_fit, 6:-6]
        E0 = self._E0[self._row_being_fit, 6:-6]
        JOfY = JOfY[:, 6:-6]

        FSE = -JOfY_third_derivative * widths**4 / deltaQ * 0.72

        NCP = intensities * (JOfY + FSE) * E0 * E0 ** (-0.92) * masses / deltaQ
        FSE = intensities * FSE * E0 * E0 ** (-0.92) * masses / deltaQ
        return NCP, FSE

    def _get_gaussian_resolution(self, centers):
        proximity_to_y_centers = np.abs(self._y_space_arrays[self._row_being_fit] - centers)
        gauss_resolution = self._gaussian_resolution[self._row_being_fit]
        assert proximity_to_y_centers.shape == gauss_resolution.shape
        return np.take_along_axis(gauss_resolution, proximity_to_y_centers.argmin(axis=1, keepdims=True), axis=1)

    def _set_gaussian_resolution(self):
        masses = self._masses.reshape(-1, 1, 1)
        v0 = self._v0
        E0 = self._E0
        delta_Q = self._deltaQ
        det, plick, angle, T0, L0, L1 = np.hsplit(self._instrument_params, 6)
        dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = np.hsplit(self._resolution_params, 6)

        angle = angle * np.pi / 180

        dWdE1 = 1.0 + (E0 / ENERGY_FINAL) ** 1.5 * (L1 / L0)
        dWdTOF = 2.0 * E0 * v0 / L0
        dWdL1 = 2.0 * E0**1.5 / ENERGY_FINAL**0.5 / L0
        dWdL0 = 2.0 * E0 / L0

        dW2 = (dWdE1**2 * dE1**2 + dWdTOF**2 * dTOF**2 + dWdL1**2 * dL1**2 + dWdL0**2 * dL0**2) * np.ones((masses.size, 1, 1))
        # conversion from meV^2 to A^-2, dydW = (M/q)^2
        dW2 *= (masses / H_BAR**2 / delta_Q) ** 2

        dQdE1 = 1.0 - (E0 / ENERGY_FINAL) ** 1.5 * L1 / L0 - np.cos(angle) * ((E0 / ENERGY_FINAL) ** 0.5 - L1 / L0 * E0 / ENERGY_FINAL)
        dQdTOF = 2.0 * E0 * v0 / L0
        dQdL1 = 2.0 * E0**1.5 / L0 / ENERGY_FINAL**0.5
        dQdL0 = 2.0 * E0 / L0
        dQdTheta = 2.0 * np.sqrt(E0 * ENERGY_FINAL) * np.sin(angle)

        dQ2 = (
            dQdE1**2 * dE1**2
            + (dQdTOF**2 * dTOF**2 + dQdL1**2 * dL1**2 + dQdL0**2 * dL0**2) * np.abs(ENERGY_FINAL / E0 * np.cos(angle) - 1)
            + dQdTheta**2 * dTheta**2
        )
        dQ2 *= (NEUTRON_MASS / H_BAR**2 / delta_Q) ** 2

        # in A-1    #same as dy^2 = (dy/dw)^2*dw^2 + (dy/dq)^2*dq^2
        gaussianResWidth = np.sqrt(dW2 + dQ2)
        self._gaussian_resolution = np.swapaxes(gaussianResWidth, 0, 1)
        return

    def _get_lorentzian_resolution(self, centers):
        proximity_to_y_centers = np.abs(self._y_space_arrays[self._row_being_fit] - centers)
        lorentzian_resolution = self._lorentzian_resolution[self._row_being_fit]
        assert proximity_to_y_centers.shape == lorentzian_resolution.shape
        return np.take_along_axis(lorentzian_resolution, proximity_to_y_centers.argmin(axis=1, keepdims=True), axis=1)

    def _set_lorentzian_resolution(self):
        masses = self._masses.reshape(-1, 1, 1)
        E0 = self._E0
        delta_Q = self._deltaQ
        det, plick, angle, T0, L0, L1 = np.hsplit(self._instrument_params, 6)
        dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = np.hsplit(self._resolution_params, 6)

        angle = angle * np.pi / 180

        dWdE1_lor = (1.0 + (E0 / ENERGY_FINAL) ** 1.5 * (L1 / L0)) ** 2 * np.ones((masses.size, 1, 1))
        # conversion from meV^2 to A^-2
        dWdE1_lor *= (masses / H_BAR**2 / delta_Q) ** 2

        dQdE1_lor = (
            1.0 - (E0 / ENERGY_FINAL) ** 1.5 * L1 / L0 - np.cos(angle) * ((E0 / ENERGY_FINAL) ** 0.5 + L1 / L0 * E0 / ENERGY_FINAL)
        ) ** 2
        dQdE1_lor *= (NEUTRON_MASS / H_BAR**2 / delta_Q) ** 2

        lorentzianResWidth = np.sqrt(dWdE1_lor + dQdE1_lor) * dE1_lorz  # in A-1
        self._lorentzian_resolution = np.swapaxes(lorentzianResWidth, 0, 1)
        return

    def _replace_zeros_with_ncp_for_corrections(self):
        """
        If the initial input contains columns with zeros
        (to mask resonance peaks) then these sections must be approximated
        by the total fitted function because multiple scattering and
        gamma correction algorithms do not accept columns with zeros.
        If no masked columns are present then the input workspace
        for corrections is left unchanged.
        """
        dataY = self._workspace_for_corrections.extractY()
        ncp = self._fit_profiles_workspaces["total"].extractY()

        self._zero_columns_boolean_mask = np.all(dataY == 0, axis=0)  # Masked Cols

        for row in range(self._workspace_for_corrections.getNumberHistograms()):
            self._workspace_for_corrections.dataY(row)[self._zero_columns_boolean_mask] = ncp[row, self._zero_columns_boolean_mask]

        SumSpectra(InputWorkspace=self._workspace_for_corrections.name(), OutputWorkspace=self._workspace_for_corrections.name() + "_sum")
        return

    def _remask_columns_with_zeros(self, ws_to_remask_name):
        """
        Uses previously stored information on masked columns in the
        initial workspace to set these columns again to zero on the
        workspace resulting from the multiple scattering or gamma correction.
        """
        ws_to_remask = mtd[ws_to_remask_name]
        for row in range(ws_to_remask.getNumberHistograms()):
            ws_to_remask.dataY(row)[self._zero_columns_boolean_mask] = 0
            ws_to_remask.dataE(row)[self._zero_columns_boolean_mask] = 0
        return

    def _correct_for_gamma_and_multiple_scattering(self, ws_name):
        if self._gamma_correction:
            gamma_correction_ws = self.create_gamma_workspaces()
            Minus(LHSWorkspace=ws_name, RHSWorkspace=gamma_correction_ws.name(), OutputWorkspace=ws_name)

        if self._multiple_scattering_correction:
            multiple_scattering_ws = self.create_multiple_scattering_workspaces()
            Minus(LHSWorkspace=ws_name, RHSWorkspace=multiple_scattering_ws.name(), OutputWorkspace=ws_name)
        return

    def create_multiple_scattering_workspaces(self):
        """Creates _MulScattering and _TotScattering workspaces used for the MS correction"""
        self.log().notice("\nEvaluating multiple scattering correction ...\n")

        # Make local variables
        masses = self._masses
        mean_widths = self._mean_widths
        mean_intensity_ratios = self._mean_intensity_ratios

        # If Backscattering mode and H is present in the sample, add H to MS properties
        if self._mode_running == ScatteringType.BACKWARD:
            if self._h_ratio > 0:  # If H is present, ratio is a number
                HIntensity = self._h_ratio * mean_intensity_ratios[self._chosen_index_for_h_ratio]
                mean_intensity_ratios = np.append(mean_intensity_ratios, HIntensity)
                mean_intensity_ratios /= np.sum(mean_intensity_ratios)
                masses = np.append(masses, 1.0079)
                mean_widths = np.append(mean_widths, 5.0)

        dens, trans = VesuvioThickness(
            Masses=masses,
            Amplitudes=mean_intensity_ratios,
            TransmissionGuess=self._transmission_guess,
            Thickness=0.1,
        )
        ws_corrections_name = self._workspace_for_corrections.name()
        atomic_properties_list = make_multiple_scattering_input_string(masses, mean_widths, mean_intensity_ratios)

        # NOTE: Sample shape in the input workspace of VesuvioCalculateMS matters!
        # The input workspace to VesuvioAnalysisRoutine should have the sample set beorehand!
        VesuvioCalculateMS(
            InputWorkspace=ws_corrections_name,
            NoOfMasses=len(masses),
            SampleDensity=dens.cell(9, 1),
            AtomicProperties=atomic_properties_list,
            BeamRadius=2.5,
            NumScatters=self._multiple_scattering_order,
            NumEventsPerRun=int(self._number_of_events),
            TotalScatteringWS=ws_corrections_name + "_tot_sctr",
            MultipleScatteringWS=ws_corrections_name + "_mltp_sctr",
        )
        data_normalisation = Integration(ws_corrections_name)
        simulation_normalisation = Integration(ws_corrections_name + "_tot_sctr")
        for ws_sctr_name in (ws_corrections_name + "_mltp_sctr", ws_corrections_name + "_tot_sctr"):
            Divide(
                LHSWorkspace=ws_sctr_name,
                RHSWorkspace=simulation_normalisation,
                OutputWorkspace=ws_sctr_name,
            )
            Multiply(
                LHSWorkspace=ws_sctr_name,
                RHSWorkspace=data_normalisation,
                OutputWorkspace=ws_sctr_name,
            )
            # Sum spectra for vizualisation
            SumSpectra(ws_sctr_name, OutputWorkspace=ws_sctr_name + "_sum")

        DeleteWorkspaces([data_normalisation, simulation_normalisation, trans, dens])
        # The only remaining workspaces are the _mltp_sctr and _tot_sctr
        return mtd[ws_corrections_name + "_mltp_sctr"]

    def create_gamma_workspaces(self):
        """Creates _gamma_background correction workspace to be subtracted from the main workspace"""

        ws_corrections_name = self._workspace_for_corrections.name()
        profiles_string = make_gamma_correction_input_string(self._masses, self._mean_widths, self._mean_intensity_ratios)

        background, corrected = VesuvioCalculateGammaBackground(InputWorkspace=ws_corrections_name, ComptonFunction=profiles_string)
        DeleteWorkspace(corrected)
        RenameWorkspace(InputWorkspace=background, OutputWorkspace=ws_corrections_name + "_gamma_backgr")
        Scale(
            InputWorkspace=ws_corrections_name + "_gamma_backgr",
            OutputWorkspace=ws_corrections_name + "_gamma_backgr",
            Factor=0.9,
            Operation="Multiply",
        )
        return mtd[ws_corrections_name + "_gamma_backgr"]
