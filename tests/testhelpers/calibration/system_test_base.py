import unittest
from os import path
from typing import ClassVar

import numpy as np
from mantid.api import WorkspaceGroup
from mantid.simpleapi import ConvertToDistribution, LoadRaw, LoadVesuvio, mtd

from mvesuvio.globals import FitFunction, Mode
from mvesuvio.util.calibration_helpers import EVSGlobals


class TestConstants:
    D_SPACINGS_COPPER: ClassVar[list[float]] = [2.0865, 1.807, 1.278, 1.0897]
    D_SPACINGS_LEAD: ClassVar[list[float]] = [1.489, 1.750, 2.475, 2.858]
    D_SPACINGS_NIOBIUM: ClassVar[list[float]] = [2.3356, 1.6515, 1.3484, 1.1678]

    MASS_COPPER: ClassVar[float] = 63.546
    MASS_LEAD: ClassVar[float] = 207.19
    MASS_NIOBIUM: ClassVar[float] = 92.906

    DEFAULT_RELATIVE_TOLERANCE: ClassVar[float] = 0.1
    IGNORE_DETECTOR: ClassVar[int] = 100  # Detectors to be ignored by the system test, as specified by the user
    INVALID_DETECTOR: ClassVar[int] = 101  # Detectors identified as invalid by the script


class EVSCalibrationTest(unittest.TestCase):
    def load_ip_file(self):
        param_names = ["spectrum", "theta", "t0", "L0", "L1"]
        file_data = np.loadtxt(self._parameter_file, skiprows=3, usecols=[0, 2, 3, 4, 5], unpack=True)

        params = {}
        for name, column in zip(param_names, file_data):
            params[name] = column

        return params

    def _setup_copper_test(self):
        self._selected_mode = Mode.FOIL_OUT.value
        self._selected_spec_range = EVSGlobals.DETECTOR_RANGE
        self._E1_fit_active = False
        self._L0_fit_active = False
        self._run_range = self._load_group_ws([17087, 17088])
        self._background = self._load_group_ws([17086])
        self._load_e1_workspaces([17087, 17088])
        #  Mass of copper in amu
        self._mass = TestConstants.MASS_COPPER
        #  d-spacings of a copper sample
        self._d_spacings = np.array(TestConstants.D_SPACINGS_COPPER)
        self._energy_estimates = np.array(EVSGlobals.ENERGY_ESTIMATE)

    def _setup_lead_test(self):
        self._selected_mode = Mode.FOIL_OUT.value
        self._selected_spec_range = EVSGlobals.DETECTOR_RANGE
        self._E1_fit_active = False
        self._L0_fit_active = False
        self._run_range = self._load_group_ws([17083, 17084])
        self._background = self._load_group_ws([17086])
        self._load_e1_workspaces([17083, 17084])
        #  Mass of a lead in amu
        self._mass = TestConstants.MASS_LEAD
        #  d-spacings of a lead sample
        self._d_spacings = np.array(TestConstants.D_SPACINGS_LEAD)
        self._energy_estimates = np.array(EVSGlobals.ENERGY_ESTIMATE)

    def _setup_niobium_test(self):
        self._selected_mode = Mode.FOIL_OUT.value
        self._selected_spec_range = EVSGlobals.DETECTOR_RANGE
        self._E1_fit_active = False
        self._L0_fit_active = False
        self._run_range = self._load_group_ws([17089, 17090, 17091])
        self._background = self._load_group_ws([17086])
        self._load_e1_workspaces([17089, 17090, 17091])
        #  Mass of a lead in amu
        self._mass = TestConstants.MASS_NIOBIUM
        #  d-spacings of a lead sample
        self._d_spacings = np.array(TestConstants.D_SPACINGS_NIOBIUM)
        self._energy_estimates = np.array(EVSGlobals.ENERGY_ESTIMATE)

    def _setup_uranium_test(self):
        self._function = FitFunction.GAUSSIAN.value
        self._mass = EVSGlobals.U_MASS
        self._d_spacings = []
        self._energy_estimates = np.array(EVSGlobals.U_PEAK_ENERGIES)
        self._energy_estimates.sort()

    def _load_uranium_fit_workspaces(self, samples, backgrounds, spectrum_range):
        self._selected_mode = Mode.FOIL_OUT.value
        self._selected_spec_range = spectrum_range
        self._run_range = self._load_group_ws(samples, "_uranium_sample")
        self._background = self._load_group_ws(backgrounds, "_uranium_background")

    def _load_group_ws(self, run_numbers, suffix=""):
        group = WorkspaceGroup()

        for run in run_numbers:
            output_name = f"EVS{run}{suffix}"
            self._load_file(run, output_name)
            group.addWorkspace(mtd[output_name])
        print("Load Successful")
        return group

    def _load_e1_workspaces(self, run_numbers):
        self._selected_mode = Mode.SINGLE_DIFFERENCE.value
        self._selected_spec_range = EVSGlobals.BACKSCATTERING_RANGE
        self._backscattering_samples = self._load_group_ws(run_numbers, "_back")
        self._selected_spec_range = EVSGlobals.FRONTSCATTERING_RANGE
        self._frontscattering_samples = self._load_group_ws(run_numbers, "_front")

    def _load_l0_workspaces(self):
        self._selected_mode = Mode.FOIL_OUT.value
        self._selected_spec_range = EVSGlobals.FRONTSCATTERING_RANGE
        self._uranium_front_samples = self._load_group_ws(EVSGlobals.U_FRONTSCATTERING_SAMPLE, "_u_front")
        self._uranium_front_background = self._load_group_ws(EVSGlobals.U_FRONTSCATTERING_BACKGROUND, "_u_front_background")
        self._selected_spec_range = EVSGlobals.BACKSCATTERING_RANGE
        self._uranium_back_samples = self._load_group_ws(EVSGlobals.U_BACKSCATTERING_SAMPLE, "_u_back")
        self._uranium_back_background = self._load_group_ws(EVSGlobals.U_BACKSCATTERING_BACKGROUND, "_u_back_background")

    def _background_run_range(self):
        return None if self._E1_fit_active else self._background

    def _get_d_spacings(self):
        return [] if (self._E1_fit_active or self._L0_fit_active) else self._d_spacings

    def _load_file(self, sample_no, output_name):
        for load_method in (self._load_file_vesuvio, self._load_file_raw):
            for prefix in ("EVS", "VESUVIO000"):
                try:
                    load_method(sample_no, output_name, prefix)
                    return
                except RuntimeError:
                    continue
        raise RuntimeError(f"Failed to load sample {sample_no}.")

    def _load_file_vesuvio(self, sample_no, output_name, prefix):
        print("Attempting LoadVesuvio")
        test_directory = path.dirname(path.dirname(path.dirname(__file__)))
        filename = str(path.join(test_directory, "data", "calibration", f"{prefix}{sample_no}.raw"))
        LoadVesuvio(
            Filename=filename,
            Mode=self._selected_mode,
            OutputWorkspace=output_name,
            SpectrumList=f"{self._selected_spec_range[0]:d}-{self._selected_spec_range[1]:d}",
            EnableLogging=False,
        )

    def _load_file_raw(self, sample_no, output_name, prefix):
        print("Attempting LoadRaw")
        test_directory = path.dirname(path.dirname(path.dirname(__file__)))
        filename = str(path.join(test_directory, "data", "calibration", f"{prefix}{sample_no}.raw"))
        LoadRaw(
            filename,
            OutputWorkspace=output_name,
            SpectrumMin=self._selected_spec_range[0],
            SpectrumMax=self._selected_spec_range[-1],
            EnableLogging=False,
        )
        ConvertToDistribution(output_name, EnableLogging=False)

    def tearDown(self):
        mtd.clear()
