from typing import ClassVar

import numpy as np
import numpy.typing as npt
import scipy.constants
from mantid.simpleapi import CreateEmptyTableWorkspace, mtd

from mvesuvio.globals import FitFunction


class EVSGlobals:
    # Configuration for Uranium runs / Indium runs
    # ----------------------------------------------------------------------------------------
    # Uranium sample & background run numbers
    U_FRONTSCATTERING_SAMPLE: ClassVar[list[int]] = [14025]  # [14025]  for U foil in the beam,   [19129, 19130] for In foil in the beam

    #  ['12570']  or [19132, 19134, 19136, 19138, 19140, 19142,19144, 19146, 19148, 19150, 19152] or [42209, 42210, 42211, 42212, 42213, 42214, 42215, 42216, 42217, 42218, 42219, 42220, 42221, 42222, 42223, 42224, 42225, 42226, 42227,42228] for Pb 2mm with U foil out
    U_FRONTSCATTERING_BACKGROUND: ClassVar[list[int]] = [
        42209,
        42210,
        42211,
        42212,
        42213,
        42214,
        42215,
        42216,
        42217,
        42218,
        42219,
        42220,
        42221,
        42222,
        42223,
        42224,
        42225,
        42226,
        42227,
        42228,
    ]

    #  ['12570']  or [19132, 19134, 19136, 19138, 19140, 19142,19144, 19146, 19148, 19150, 19152] or [42209, 42210, 42211, 42212, 42213, 42214, 42215, 42216, 42217, 42218, 42219, 42220, 42221, 42222, 42223, 42224, 42225, 42226, 42227,42228] for Pb 2mm with U foil out
    U_BACKSCATTERING_SAMPLE: ClassVar[list[int]] = [
        42209,
        42210,
        42211,
        42212,
        42213,
        42214,
        42215,
        42216,
        42217,
        42218,
        42219,
        42220,
        42221,
        42222,
        42223,
        42224,
        42225,
        42226,
        42227,
        42228,
    ]

    # ['12571'] or [42229,42230,42231,42232,42233,42234,42235,42236,42237,42238,42239,42240,42241,42242,42243,42244,42245,42246,42247,42248,42249,42250,42251,42252,42253] or  [19131, 19133, 19135, 19137, 19139, 19141, 19143, 19145, 19147, 19149, 19151]  for Pb 2mm with U foil in
    U_BACKSCATTERING_BACKGROUND: ClassVar[list[int]] = [
        42229,
        42230,
        42231,
        42232,
        42233,
        42234,
        42235,
        42236,
        42237,
        42238,
        42239,
        42240,
        42241,
        42242,
        42243,
        42244,
        42245,
        42246,
        42247,
        42248,
        42249,
        42250,
        42251,
        42252,
        42253,
    ]

    # peak enegy for U/In in mev
    U_PEAK_ENERGIES: ClassVar[list[int]] = [
        36684,
        20874,
        6672,
    ]  # [36684, 20874, 6672] for uranium, [39681,22723,14599,12056,9088,3855] for indium
    # U mass/ In mass  in amu
    U_MASS: ClassVar[float] = 238.0289  # 113.0 for indium

    # misc global variables
    # ----------------------------------------------------------------------------------------
    # full range of both the front and backscattering banks
    FRONTSCATTERING_RANGE: ClassVar[list[int]] = [135, 198]
    BACKSCATTERING_RANGE: ClassVar[list[int]] = [3, 134]
    DETECTOR_RANGE: ClassVar[list[int]] = [BACKSCATTERING_RANGE[0], FRONTSCATTERING_RANGE[1]]

    # self._fit_window_range applies bith to fitting the resonances and the lead recoil peaks
    # it is defined as the range left and right from the peak centre (i. e. the whole fitting window is twice the fitting range)

    BRAGG_PEAK_CROP_RANGE: ClassVar[tuple[int, int]] = (2000, 20000)
    BRAGG_FIT_WINDOW_RANGE: ClassVar[int] = 500
    BRAGG_PEAK_POSITION_TOLERANCE: ClassVar[int] = 1000

    RECOIL_PEAK_CROP_RANGE: ClassVar[tuple[int, int]] = (300, 500)
    RECOIL_FIT_WINDOW_RANGE: ClassVar[int] = 300

    RESONANCE_PEAK_CROP_RANGE: ClassVar[tuple[int, int]] = (100, 350)
    RESONANCE_FIT_WINDOW_RANGE: ClassVar[int] = 50

    PEAK_HEIGHT_RELATIVE_THRESHOLD: ClassVar[float] = 0.25

    # energy used to estimate peak position
    ENERGY_ESTIMATE: ClassVar[float] = 4897.3

    # physical constants
    # ----------------------------------------------------------------------------------------
    # convert to 1 / v and scale for fitting
    U_NEUTRON_VELOCITY: ClassVar[npt.NDArray[np.float64]] = np.array(
        [83769.7, 63190.5, 35725.4]
    )  # np.array([83769.7, 63190.5, 35725.4]) # for U  # np.array([87124.5,65929.8,52845.8,48023.1,41694.9,27155.7])  # for indium
    U_NEUTRON_VELOCITY = 1.0 / U_NEUTRON_VELOCITY
    U_NEUTRON_VELOCITY *= 1e6

    # mass of a neutron in amu
    NEUTRON_MASS_AMU: ClassVar[float] = scipy.constants.value("neutron mass in u")
    # 1 meV in Joules
    MEV_CONVERSION: ClassVar[float] = 1.602176487e-22


class EVSMiscFunctions:
    @staticmethod
    def calculate_r_theta(sample_mass, thetas):
        """
        Returns the ratio of the final neutron velocity to the initial neutron velocity
        as a function of the scattering angle theta and atomic mass of lead and a neutron.

        @param sample_mass - mass of the sample in amu
        @param thetas - vector containing the values of theta
        @return the ratio of final and incident velocities
        """
        rad_theta = np.radians(thetas)
        r_theta = (np.cos(rad_theta) + np.sqrt((sample_mass / EVSGlobals.NEUTRON_MASS_AMU) ** 2 - np.sin(rad_theta) ** 2)) / (
            (sample_mass / EVSGlobals.NEUTRON_MASS_AMU) + 1
        )
        return r_theta

    # The IP text file load function skips the first 3 rows of the text file.
    # This assumes that there is one line for the file header and 2 lines for
    # parameters of monitor 1 and monitor 2, respectively.
    # One has to be careful here as some IP files have no header!
    # In such a case one has to add header
    # Otherwise, the algorithm will read the number of spectra to be fitted
    # for the calibration of of L0, t0, L1, theta from the vectors:
    # FRONTSCATTERING_RANGE, BACKSCATTERING_RANGE and DETECTOR_RANGE
    # This number will be different from the number of fits performed
    # (which will be given by the length of the variable table_name)
    # The program will start running but will crash after fitting L0 and t0
    @staticmethod
    def load_instrument_parameters(file_path, table_name):
        """
        Load instrument parameters from file into a table workspace

        @param file_path - the location of the file on disk
        @return the name of the table workspace.
        """

        file_data = np.loadtxt(file_path, skiprows=3, usecols=[0, 2, 3, 4, 5])

        CreateEmptyTableWorkspace(OutputWorkspace=table_name)
        table_ws = mtd[table_name]

        table_ws.addColumn("double", "Spectrum")
        table_ws.addColumn("double", "theta")
        table_ws.addColumn("double", "t0")
        table_ws.addColumn("double", "L0")
        table_ws.addColumn("double", "L1")

        for row in file_data:
            table_ws.addRow(row.tolist())

        return table_name

    @staticmethod
    def read_table_column(table_name, column_name, spec_list=EVSGlobals.DETECTOR_RANGE, offset=None):
        """
        Read a column from a table workspace and return the data as an array.

        @param table_name - name of the table workspace
        @param column_name - name of the column to select
        @param spec_list - range of spectra to use
        @param offset - the offset when reading values from the column
        @return numpy array of values in the spec_list range
        """
        offset = spec_list[0] if offset is None else offset
        if len(spec_list) == 1:
            lower = upper = spec_list[0]
        else:
            lower, upper = spec_list

        column_values = mtd[table_name].column(column_name)

        return np.array(column_values[lower - offset : upper - offset + 1])

    @staticmethod
    def set_table_column(table_name, column_name, data, spec_list=None):
        """
        Add a set of data to a table workspace

        @param table_name - name of the table workspace to modify
        @param column_name - name of the column to add data to
        @param data - data array to add to the table
        @param spec_list - range of rows to add the data too
        """
        table_ws = mtd[table_name]

        if column_name not in table_ws.getColumnNames():
            table_ws.addColumn("double", column_name)

        if spec_list is None:
            offset = 0
        else:
            if len(data) < (spec_list[1] - spec_list[0] + 1):
                raise ValueError("Not enough data from spectrum range.")

            offset = spec_list[0] - EVSGlobals.DETECTOR_RANGE[0]

        if isinstance(data, np.ndarray):
            data = data.tolist()

        for i, value in enumerate(data):
            table_ws.setCell(column_name, offset + i, value)

    @staticmethod
    def generate_fit_function_header(function_type, error=False):
        if function_type == FitFunction.VOIGT:
            error_str = "Err" if error else ""
            func_header = {"Height": "LorentzAmp", "Position": "LorentzPos", "Width": "LorentzFWHM", "Width_2": "GaussianFWHM"}
        elif function_type == FitFunction.GAUSSIAN:
            error_str = "_Err" if error else ""
            func_header = {"Height": "Height", "Width": "Sigma", "Position": "PeakCentre"}
        else:
            raise ValueError(f"Unsupported fit function type: {function_type}")

        return {k: v + error_str for k, v in func_header.items()}


class InvalidDetectors:
    def __init__(self, invalid_detector_list):
        if invalid_detector_list:
            self._invalid_detectors_front = self._preset_invalid_detectors(invalid_detector_list, EVSGlobals.FRONTSCATTERING_RANGE)
            self._invalid_detectors_back = self._preset_invalid_detectors(invalid_detector_list, EVSGlobals.BACKSCATTERING_RANGE)
            self._detectors_preset = True
        else:
            self._invalid_detectors_front = np.array([])
            self._invalid_detectors_back = np.array([])
            self._detectors_preset = False

    def add_invalid_detectors(self, invalid_detector_list):
        """
        Takes a list of invalid spectra, adds unique entries to the stored np arrays.
        @param invalid_detector_list - list of detectors to append to detector list, if unique.
        """
        self._invalid_detectors_front = np.array(
            [
                [x]
                for x in sorted(
                    set(self._invalid_detectors_front.ravel()).union(
                        set(self._preset_invalid_detectors(invalid_detector_list, EVSGlobals.FRONTSCATTERING_RANGE).ravel())
                    )
                )
            ]
        )
        self._invalid_detectors_back = np.array(
            [
                [x]
                for x in sorted(
                    set(self._invalid_detectors_back.ravel()).union(
                        set(self._preset_invalid_detectors(invalid_detector_list, EVSGlobals.BACKSCATTERING_RANGE).ravel())
                    )
                )
            ]
        )

    def get_all_invalid_detectors(self):
        return [i + EVSGlobals.BACKSCATTERING_RANGE[0] for i in self._invalid_detectors_back.flatten().tolist()] + [
            j + EVSGlobals.FRONTSCATTERING_RANGE[0] for j in self._invalid_detectors_front.flatten().tolist()
        ]

    def get_invalid_detectors_index(self, desired_range):
        if desired_range == EVSGlobals.FRONTSCATTERING_RANGE:
            return self._invalid_detectors_front.flatten().tolist()
        elif desired_range == EVSGlobals.BACKSCATTERING_RANGE:
            return self._invalid_detectors_back.flatten().tolist()
        else:
            raise AttributeError("desired range invalid - must represent either front or back detectors.")

    @staticmethod
    def _preset_invalid_detectors(invalid_detector_list_full_range, desired_range):
        return np.array([[x - desired_range[0]] for x in invalid_detector_list_full_range if desired_range[0] <= x <= desired_range[-1]])

    def filter_peak_centres_for_invalid_detectors(self, detector_range, peak_table):
        """
        Finds invalid detectors and filters the peak centres. Caches the invalid detectors found to avoid repeat identification.
        @param detector_range detectors to consider, must be either FRONT or BACKSCATTERING range.
        @param peak_table - name of table containing fitted parameters each spectra.

        @returns peak_centres - a list of peak fitted peak centres, with those that represent invalid detectors marked nan.
        """

        invalid_detectors = self.identify_and_set_invalid_detectors_from_range(detector_range, peak_table)
        peak_centres = EVSMiscFunctions.read_table_column(peak_table, "f1.LorentzPos", detector_range)
        peak_centres[invalid_detectors] = np.nan
        return peak_centres

    def identify_and_set_invalid_detectors_from_range(self, detector_range, peak_table):
        """
        Finds invalid detectors and caches the invalid detectors. Will not look to calculate if invalid detectors already exist.
        @param detector_range detectors to consider, must be either FRONT or BACKSCATTERING range.
        @param peak_table - name of table containing fitted parameters each spectra.

        @returns invalid_detectors - a list of the index's of invalid detector, in the context of the range they belong to.
        """

        peak_centres = EVSMiscFunctions.read_table_column(peak_table, "f1.LorentzPos", detector_range)
        peak_centres_errors = EVSMiscFunctions.read_table_column(peak_table, "f1.LorentzPos_Err", detector_range)

        if detector_range == EVSGlobals.FRONTSCATTERING_RANGE:
            if not self._detectors_preset and not self._invalid_detectors_front.any():
                self._invalid_detectors_front = self._identify_invalid_spectra(
                    peak_table, peak_centres, peak_centres_errors, detector_range
                )
                self._print_invalid_detectors(front_detector_range=True)
            return self._invalid_detectors_front
        elif detector_range == EVSGlobals.BACKSCATTERING_RANGE:
            if not self._detectors_preset and not self._invalid_detectors_back.any():
                self._invalid_detectors_back = self._identify_invalid_spectra(peak_table, peak_centres, peak_centres_errors, detector_range)
                self._print_invalid_detectors(front_detector_range=False)
            return self._invalid_detectors_back
        else:
            raise AttributeError("Spec list invalid - must represent either front or back detectors.")

    def _print_invalid_detectors(self, front_detector_range):
        if front_detector_range:
            invalid_detectors = self._invalid_detectors_front
            detector_range = EVSGlobals.FRONTSCATTERING_RANGE
        else:
            invalid_detectors = self._invalid_detectors_back
            detector_range = EVSGlobals.BACKSCATTERING_RANGE

        print(
            f"Invalid Spectra Index Found and Marked NAN: {invalid_detectors} from Spectra Index List:"
            f"{[x - EVSGlobals.DETECTOR_RANGE[0] for x in detector_range]}"
        )

    @staticmethod
    def _identify_invalid_spectra(peak_table, peak_centres, peak_centres_errors, spec_list):
        """
        Inspect fitting results, and identify the fits associated with invalid spectra. These are spectra associated with detectors
        which have lost foil coverage following a recent reduction in distance from source to detectors.

        @param peak_table - name of table containing fitted parameters each spectra.
        @param spec_list - spectrum range to inspect.
        @return a list of invalid spectra.
        """
        peak_Gaussian_FWHM = EVSMiscFunctions.read_table_column(peak_table, "f1.GaussianFWHM", spec_list)
        peak_Gaussian_FWHM_errors = EVSMiscFunctions.read_table_column(peak_table, "f1.GaussianFWHM_Err", spec_list)
        peak_Lorentz_FWHM = EVSMiscFunctions.read_table_column(peak_table, "f1.LorentzFWHM", spec_list)
        peak_Lorentz_FWHM_errors = EVSMiscFunctions.read_table_column(peak_table, "f1.LorentzFWHM_Err", spec_list)
        peak_Lorentz_Amp = EVSMiscFunctions.read_table_column(peak_table, "f1.LorentzAmp", spec_list)
        peak_Lorentz_Amp_errors = EVSMiscFunctions.read_table_column(peak_table, "f1.LorentzAmp_Err", spec_list)

        invalid_spectra = np.argwhere(
            (np.isinf(peak_Lorentz_Amp_errors))
            | (np.isnan(peak_Lorentz_Amp_errors))
            | (np.isinf(peak_centres_errors))
            | (np.isnan(peak_centres_errors))
            | (np.isnan(peak_Gaussian_FWHM_errors))
            | (np.isinf(peak_Gaussian_FWHM_errors))
            | (np.isnan(peak_Lorentz_FWHM_errors))
            | (np.isinf(peak_Lorentz_FWHM_errors))
            | (np.isnan(peak_Lorentz_Amp_errors))
            | (np.isinf(peak_Lorentz_Amp_errors))
            | (np.absolute(peak_Gaussian_FWHM_errors) > np.absolute(peak_Gaussian_FWHM))
            | (np.absolute(peak_Lorentz_FWHM_errors) > np.absolute(peak_Lorentz_FWHM))
            | (np.absolute(peak_Lorentz_Amp_errors) > np.absolute(peak_Lorentz_Amp))
            | (np.absolute(peak_centres_errors) > np.absolute(peak_centres))
        )
        return invalid_spectra
