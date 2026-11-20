import unittest
from functools import partial
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
from mantid.api import WorkspaceGroup, WorkspaceGroupProperty
from mantid.kernel import Direction, FloatArrayProperty, IntArrayProperty
from mantid.simpleapi import CreateWorkspace, mtd

from mvesuvio.calibrate_fit import EVSCalibrationFit
from mvesuvio.globals import FitFunction, Mode, PeakType, SharedParameterFitType
from mvesuvio.util.calibration_helpers import EVSGlobals


class TestVesuvioCalibrationFit(unittest.TestCase):
    def setUp(self):
        self.set_cell_list = []

    @staticmethod
    def setup_mtd_mock(mock_obj, mock_dict):
        d = {}
        for key, return_obj in mock_dict.items():
            d[key] = return_obj
            mock_obj.__getitem__.side_effect = d.__getitem__

    def side_effect_set_cell(self, arg1, arg2, value):
        self.set_cell_list.append((arg1, arg2, value))

    @staticmethod
    def side_effect_cell(row_index, col_index, peaks):
        return peaks[row_index]

    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    @patch("mvesuvio.calibrate_fit.CloneWorkspace")
    @patch("mvesuvio.calibrate_fit.mtd")
    def test_filter_peaks_perfect_match(self, mock_mtd, mock_clone_workspace, mock_del_workspace):
        alg = EVSCalibrationFit()

        find_peaks_output_name = "find_peaks_output_name"
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_find_peaks_output_name_unfiltered = MagicMock()
        found_peaks = [9440, 13351, 15417]
        return_mock_find_peaks_output_name_unfiltered.column.return_value = found_peaks
        return_mock_find_peaks_output_name_unfiltered.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        return_mock_find_peaks_output_name = MagicMock()
        return_mock_find_peaks_output_name.setRowCount = MagicMock()
        return_mock_find_peaks_output_name.columnCount.return_value = 1
        return_mock_find_peaks_output_name.rowCount.return_value = len(peak_estimates_list)
        return_mock_find_peaks_output_name.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {
            "find_peaks_output_name": return_mock_find_peaks_output_name,
            "find_peaks_output_name_unfiltered": return_mock_find_peaks_output_name_unfiltered,
        }

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._func_param_names = {"Position": "LorentzPos"}
        alg._filter_found_peaks(find_peaks_output_name, peak_estimates_list, linear_bg_coeffs)
        self.assertEqual([(0, 0, found_peaks[0]), (1, 0, found_peaks[1]), (2, 0, found_peaks[2])], self.set_cell_list)
        mock_clone_workspace.assert_called_with(
            InputWorkspace=return_mock_find_peaks_output_name, OutputWorkspace=find_peaks_output_name + "_unfiltered"
        )
        mock_del_workspace.assert_called_with(find_peaks_output_name + "_unfiltered")

    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    @patch("mvesuvio.calibrate_fit.CloneWorkspace")
    @patch("mvesuvio.calibrate_fit.mtd")
    def test_filter_peaks_no_match(self, mock_mtd, mock_clone_workspace, mock_del_workspace):
        alg = EVSCalibrationFit()

        find_peaks_output_name = "find_peaks_output_name"
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_find_peaks_output_name_unfiltered = MagicMock()
        found_peaks = []
        return_mock_find_peaks_output_name_unfiltered.column.return_value = found_peaks
        return_mock_find_peaks_output_name_unfiltered.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        return_mock_find_peaks_output_name = MagicMock()
        return_mock_find_peaks_output_name.setRowCount = MagicMock()
        return_mock_find_peaks_output_name.columnCount.return_value = 1
        return_mock_find_peaks_output_name.rowCount.return_value = len(peak_estimates_list)
        return_mock_find_peaks_output_name.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {
            "find_peaks_output_name": return_mock_find_peaks_output_name,
            "find_peaks_output_name_unfiltered": return_mock_find_peaks_output_name_unfiltered,
        }

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._func_param_names = {"Position": "LorentzPos"}
        alg._filter_found_peaks(find_peaks_output_name, peak_estimates_list, linear_bg_coeffs)
        self.assertEqual([], self.set_cell_list)
        mock_clone_workspace.assert_called_with(
            InputWorkspace=return_mock_find_peaks_output_name, OutputWorkspace=find_peaks_output_name + "_unfiltered"
        )
        mock_del_workspace.assert_called_with(find_peaks_output_name + "_unfiltered")

    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    @patch("mvesuvio.calibrate_fit.CloneWorkspace")
    @patch("mvesuvio.calibrate_fit.mtd")
    def test_filter_peaks_one_match(self, mock_mtd, mock_clone_workspace, mock_del_workspace):
        alg = EVSCalibrationFit()

        find_peaks_output_name = "find_peaks_output_name"
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_find_peaks_output_name_unfiltered = MagicMock()
        found_peaks = [13000]
        return_mock_find_peaks_output_name_unfiltered.column.return_value = found_peaks
        return_mock_find_peaks_output_name_unfiltered.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        return_mock_find_peaks_output_name = MagicMock()
        return_mock_find_peaks_output_name.setRowCount = MagicMock()
        return_mock_find_peaks_output_name.columnCount.return_value = 1
        return_mock_find_peaks_output_name.rowCount.return_value = len(peak_estimates_list)
        return_mock_find_peaks_output_name.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {
            "find_peaks_output_name": return_mock_find_peaks_output_name,
            "find_peaks_output_name_unfiltered": return_mock_find_peaks_output_name_unfiltered,
        }

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._func_param_names = {"Position": "LorentzPos"}
        alg._filter_found_peaks(find_peaks_output_name, peak_estimates_list, linear_bg_coeffs)
        self.assertEqual(
            [("LorentzPos", 0, peak_estimates_list[0]), (1, 0, found_peaks[0]), ("LorentzPos", 2, peak_estimates_list[2])],
            self.set_cell_list,
        )
        mock_clone_workspace.assert_called_with(
            InputWorkspace=return_mock_find_peaks_output_name, OutputWorkspace=find_peaks_output_name + "_unfiltered"
        )
        mock_del_workspace.assert_called_with(find_peaks_output_name + "_unfiltered")

    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    @patch("mvesuvio.calibrate_fit.CloneWorkspace")
    @patch("mvesuvio.calibrate_fit.mtd")
    def test_filter_peaks_two_match(self, mock_mtd, mock_clone_workspace, mock_del_workspace):
        alg = EVSCalibrationFit()

        find_peaks_output_name = "find_peaks_output_name"
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_find_peaks_output_name_unfiltered = MagicMock()
        found_peaks = [9000, 16000]
        return_mock_find_peaks_output_name_unfiltered.column.return_value = found_peaks
        return_mock_find_peaks_output_name_unfiltered.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        return_mock_find_peaks_output_name = MagicMock()
        return_mock_find_peaks_output_name.setRowCount = MagicMock()
        return_mock_find_peaks_output_name.columnCount.return_value = 1
        return_mock_find_peaks_output_name.rowCount.return_value = len(peak_estimates_list)
        return_mock_find_peaks_output_name.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {
            "find_peaks_output_name": return_mock_find_peaks_output_name,
            "find_peaks_output_name_unfiltered": return_mock_find_peaks_output_name_unfiltered,
        }

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._func_param_names = {"Position": "LorentzPos"}
        alg._filter_found_peaks(find_peaks_output_name, peak_estimates_list, linear_bg_coeffs)
        self.assertEqual([(0, 0, found_peaks[0]), ("LorentzPos", 1, peak_estimates_list[1]), (2, 0, found_peaks[1])], self.set_cell_list)
        mock_clone_workspace.assert_called_with(
            InputWorkspace=return_mock_find_peaks_output_name, OutputWorkspace=find_peaks_output_name + "_unfiltered"
        )
        mock_del_workspace.assert_called_with(find_peaks_output_name + "_unfiltered")

    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    @patch("mvesuvio.calibrate_fit.CloneWorkspace")
    @patch("mvesuvio.calibrate_fit.mtd")
    def test_filter_peaks_does_not_include_higher_found_peak(self, mock_mtd, mock_clone_workspace, mock_del_workspace):
        alg = EVSCalibrationFit()

        find_peaks_output_name = "find_peaks_output_name"
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_find_peaks_output_name_unfiltered = MagicMock()
        found_peaks = [9440, 15417, 16000]
        return_mock_find_peaks_output_name_unfiltered.column.return_value = found_peaks
        return_mock_find_peaks_output_name_unfiltered.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        return_mock_find_peaks_output_name = MagicMock()
        return_mock_find_peaks_output_name.setRowCount = MagicMock()
        return_mock_find_peaks_output_name.columnCount.return_value = 1
        return_mock_find_peaks_output_name.rowCount.return_value = len(peak_estimates_list)
        return_mock_find_peaks_output_name.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {
            "find_peaks_output_name": return_mock_find_peaks_output_name,
            "find_peaks_output_name_unfiltered": return_mock_find_peaks_output_name_unfiltered,
        }

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._func_param_names = {"Position": "LorentzPos"}
        alg._filter_found_peaks(find_peaks_output_name, peak_estimates_list, linear_bg_coeffs)
        self.assertEqual([(0, 0, found_peaks[0]), ("LorentzPos", 1, peak_estimates_list[1]), (2, 0, found_peaks[1])], self.set_cell_list)
        mock_clone_workspace.assert_called_with(
            InputWorkspace=return_mock_find_peaks_output_name, OutputWorkspace=find_peaks_output_name + "_unfiltered"
        )
        mock_del_workspace.assert_called_with(find_peaks_output_name + "_unfiltered")

    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    @patch("mvesuvio.calibrate_fit.CloneWorkspace")
    @patch("mvesuvio.calibrate_fit.mtd")
    def test_filter_peaks_does_not_include_lower_found_peak(self, mock_mtd, mock_clone_workspace, mock_del_workspace):
        alg = EVSCalibrationFit()

        find_peaks_output_name = "find_peaks_output_name"
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_find_peaks_output_name_unfiltered = MagicMock()
        found_peaks = [8000, 9440, 15417]
        return_mock_find_peaks_output_name_unfiltered.column.return_value = found_peaks
        return_mock_find_peaks_output_name_unfiltered.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        return_mock_find_peaks_output_name = MagicMock()
        return_mock_find_peaks_output_name.setRowCount = MagicMock()
        return_mock_find_peaks_output_name.columnCount.return_value = 1
        return_mock_find_peaks_output_name.rowCount.return_value = len(peak_estimates_list)
        return_mock_find_peaks_output_name.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {
            "find_peaks_output_name": return_mock_find_peaks_output_name,
            "find_peaks_output_name_unfiltered": return_mock_find_peaks_output_name_unfiltered,
        }

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._func_param_names = {"Position": "LorentzPos"}
        alg._filter_found_peaks(find_peaks_output_name, peak_estimates_list, linear_bg_coeffs)
        self.assertEqual([(0, 0, found_peaks[1]), ("LorentzPos", 1, peak_estimates_list[1]), (2, 0, found_peaks[2])], self.set_cell_list)
        mock_clone_workspace.assert_called_with(
            InputWorkspace=return_mock_find_peaks_output_name, OutputWorkspace=find_peaks_output_name + "_unfiltered"
        )
        mock_del_workspace.assert_called_with(find_peaks_output_name + "_unfiltered")

    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    @patch("mvesuvio.calibrate_fit.CloneWorkspace")
    @patch("mvesuvio.calibrate_fit.mtd")
    def test_filter_peaks_handles_multiple_peaks(self, mock_mtd, mock_clone_workspace, mock_del_workspace):
        alg = EVSCalibrationFit()

        find_peaks_output_name = "find_peaks_output_name"
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_find_peaks_output_name_unfiltered = MagicMock()
        found_peaks = [8000, 9445, 13000, 13355, 15415, 16000]
        return_mock_find_peaks_output_name_unfiltered.column.return_value = found_peaks
        return_mock_find_peaks_output_name_unfiltered.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        return_mock_find_peaks_output_name = MagicMock()
        return_mock_find_peaks_output_name.setRowCount = MagicMock()
        return_mock_find_peaks_output_name.columnCount.return_value = 1
        return_mock_find_peaks_output_name.rowCount.return_value = len(peak_estimates_list)
        return_mock_find_peaks_output_name.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {
            "find_peaks_output_name": return_mock_find_peaks_output_name,
            "find_peaks_output_name_unfiltered": return_mock_find_peaks_output_name_unfiltered,
        }

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._func_param_names = {"Position": "LorentzPos"}
        alg._filter_found_peaks(find_peaks_output_name, peak_estimates_list, linear_bg_coeffs)
        self.assertEqual([(0, 0, found_peaks[1]), (1, 0, found_peaks[3]), (2, 0, found_peaks[4])], self.set_cell_list)
        mock_clone_workspace.assert_called_with(
            InputWorkspace=return_mock_find_peaks_output_name, OutputWorkspace=find_peaks_output_name + "_unfiltered"
        )
        mock_del_workspace.assert_called_with(find_peaks_output_name + "_unfiltered")

    # Found peaks sometimes returns 'zero' peaks, usually at the end of the table workspace.
    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    @patch("mvesuvio.calibrate_fit.CloneWorkspace")
    @patch("mvesuvio.calibrate_fit.mtd")
    def test_filter_peaks_handles_zero_position_in_found_peaks(self, mock_mtd, mock_clone_workspace, mock_del_workspace):
        alg = EVSCalibrationFit()

        find_peaks_output_name = "find_peaks_output_name"
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_find_peaks_output_name_unfiltered = MagicMock()
        found_peaks = [9440, 13351, 0]
        return_mock_find_peaks_output_name_unfiltered.column.return_value = found_peaks
        return_mock_find_peaks_output_name_unfiltered.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        return_mock_find_peaks_output_name = MagicMock()
        return_mock_find_peaks_output_name.setRowCount = MagicMock()
        return_mock_find_peaks_output_name.columnCount.return_value = 1
        return_mock_find_peaks_output_name.rowCount.return_value = len(peak_estimates_list)
        return_mock_find_peaks_output_name.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {
            "find_peaks_output_name": return_mock_find_peaks_output_name,
            "find_peaks_output_name_unfiltered": return_mock_find_peaks_output_name_unfiltered,
        }

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._func_param_names = {"Position": "LorentzPos"}
        alg._filter_found_peaks(find_peaks_output_name, peak_estimates_list, linear_bg_coeffs)
        self.assertEqual([(0, 0, found_peaks[0]), (1, 0, found_peaks[1]), ("LorentzPos", 2, peak_estimates_list[2])], self.set_cell_list)
        mock_clone_workspace.assert_called_with(
            InputWorkspace=return_mock_find_peaks_output_name, OutputWorkspace=find_peaks_output_name + "_unfiltered"
        )
        mock_del_workspace.assert_called_with(find_peaks_output_name + "_unfiltered")

    @patch("mvesuvio.util.calibration_helpers.EVSMiscFunctions.read_table_column")
    def test_estimate_bragg_peak_positions(self, mock_read_table_column):
        def side_effect(column_name, **kwargs):
            return {
                "L0": 11.05,
                "L1": 0.5505,
                "t0": -0.2,
                "theta": 139.5371,
            }.get(column_name)

        alg = EVSCalibrationFit()
        alg._spec_list = [22]
        alg._param_table = "param_table"
        mock_read_table_column.side_effect = side_effect
        alg._d_spacings = np.array([1.75, 2.475, 2.858])

        estimated_positions = alg._estimate_bragg_peak_positions()
        np.testing.assert_almost_equal([9629.84, 13619.43, 15727.03], estimated_positions.flatten().tolist(), 0.01)
        print(estimated_positions)

    @patch("mvesuvio.calibrate_fit.mtd")
    def test_check_nans_false(self, mock_mtd):
        alg = EVSCalibrationFit()
        table_ws = "table_ws"
        data = [9440, np.nan, 15417]
        return_mock_obj_table_ws = MagicMock()
        return_mock_obj_table_ws.column.return_value = data
        return_mock_obj_table_ws.columnCount.return_value = len(data)

        mtd_mock_dict = {"table_ws": return_mock_obj_table_ws}

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        self.assertFalse(alg._check_nans(table_ws))

    @patch("mvesuvio.calibrate_fit.mtd")
    def test_check_nans_true(self, mock_mtd):
        alg = EVSCalibrationFit()
        table_ws = "table_ws"
        data = [9440, 13351, 15417]
        return_mock_obj_table_ws = MagicMock()
        return_mock_obj_table_ws.column.return_value = data
        return_mock_obj_table_ws.columnCount.return_value = len(data)

        mtd_mock_dict = {"table_ws": return_mock_obj_table_ws}

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        self.assertTrue(alg._check_nans(table_ws))

    def test_PyInit_property_defaults(self):
        alg = EVSCalibrationFit()
        alg.PyInit()
        properties = {
            "Samples": None,
            "Background": None,
            "Mode": Mode.FOIL_OUT,
            "Function": "Gaussian",
            "SpectrumRange": EVSGlobals.DETECTOR_RANGE,
            "Mass": 207.19,
            "DSpacings": [],
            "Energy": [EVSGlobals.ENERGY_ESTIMATE],
            "InstrumentParameterFile": "",
            "PeakType": "",
            "InstrumentParameterWorkspace": None,
            "CreateOutput": False,
            "OutputWorkspace": "",
        }
        for prop, actual_value in properties.items():
            expected_value = alg.getProperty(prop).value
            if type(expected_value) is np.ndarray:
                expected_value = list(expected_value)
            self.assertEqual(expected_value, actual_value, f"Property {prop}. Expected: {expected_value},Actual: {actual_value}")

    @patch("mvesuvio.calibrate_fit.EVSCalibrationFit._setup_spectra_list")
    @patch("mvesuvio.calibrate_fit.EVSCalibrationFit._setup_run_numbers_and_output_workspace")
    @patch("mvesuvio.calibrate_fit.EVSCalibrationFit._setup_function_type")
    @patch("mvesuvio.calibrate_fit.EVSCalibrationFit._setup_parameter_workspace")
    @patch("mvesuvio.calibrate_fit.EVSCalibrationFit._setup_peaks_and_set_crop_and_fit_ranges")
    @patch("mvesuvio.calibrate_fit.EVSCalibrationFit._setup_class_variables_from_properties")
    def test_setup_calls_all_functions(
        self, mock_setup_vars, mock_setup_peaks, mock_setup_param_ws, mock_setup_fn_type, mock_setup_run_nos, mock_setup_spec
    ):
        alg = EVSCalibrationFit()
        alg._setup()
        mock_setup_vars.assert_called_once()
        mock_setup_peaks.assert_called_once()
        mock_setup_param_ws.assert_called_once()
        mock_setup_fn_type.assert_called_once()
        mock_setup_run_nos.assert_called_once()
        mock_setup_spec.assert_called_once()

    def test_setup_class_variables_from_properties(self):
        alg = EVSCalibrationFit()
        alg.PyInit()

        mode = alg.getProperty("Mode").value
        energy = alg.getProperty("Energy").value
        mass = alg.getProperty("Mass").value
        create_output = alg.getProperty("CreateOutput").value

        alg._setup_class_variables_from_properties()
        self.assertEqual(mode, alg._mode)
        self.assertEqual(energy, alg._energy_estimates)
        self.assertEqual(mass, alg._sample_mass)
        self.assertEqual(create_output, alg._create_output)

    def test_setup_spectra_list(self):
        test_spec_list = [3, 4, 5, 6]
        alg = EVSCalibrationFit()
        alg.declareProperty(IntArrayProperty("SpectrumRange", test_spec_list))

        alg._setup_spectra_list()
        self.assertEqual([test_spec_list[0], test_spec_list[-1]], alg._spec_list)

    def test_setup_spectra_list_one_spec(self):
        test_spec_list = [3]
        alg = EVSCalibrationFit()
        alg.declareProperty(IntArrayProperty("SpectrumRange", test_spec_list))

        alg._setup_spectra_list()
        self.assertEqual([test_spec_list[0]], alg._spec_list)

    def test_setup_spectra_list_no_spec(self):
        test_spec_list = []
        alg = EVSCalibrationFit()
        alg.declareProperty(IntArrayProperty("SpectrumRange", test_spec_list))
        self.assertRaises(ValueError, alg._setup_spectra_list)

    def test_setup_run_numbers_and_output_workspace(self):
        test_sample_group = WorkspaceGroup()
        test_background_group = WorkspaceGroup()
        CreateWorkspace(DataX=[0, 1], DataY=[3, 4], OutputWorkspace="sample_ws_1")
        CreateWorkspace(DataX=[0, 1], DataY=[3, 4], OutputWorkspace="sample_ws_2")
        CreateWorkspace(DataX=[0, 1], DataY=[3, 4], OutputWorkspace="background_ws")
        test_sample_group.addWorkspace(mtd["sample_ws_1"])
        test_sample_group.addWorkspace(mtd["sample_ws_2"])
        test_background_group.addWorkspace(mtd["background_ws"])
        test_ws_name = "test_ws"
        alg = EVSCalibrationFit()
        alg.declareProperty(WorkspaceGroupProperty("Samples", "", Direction.Input))
        alg.declareProperty(WorkspaceGroupProperty("Background", "", Direction.Input))
        alg.declareProperty("OutputWorkspace", test_ws_name)

        alg.setProperty("Samples", test_sample_group)
        alg.setProperty("Background", test_background_group)

        alg._setup_run_numbers_and_output_workspace()
        self.assertEqual(test_ws_name + "_Background_background_ws", alg._background)
        suffix = "_".join(ws.name() for ws in test_sample_group)
        self.assertEqual(test_ws_name + "_Sample_" + suffix, alg._sample)

    def test_setup_run_numbers_and_output_workspace_no_bg(self):
        test_sample_group = WorkspaceGroup()
        test_background_group = WorkspaceGroup()
        CreateWorkspace(DataX=[0, 1], DataY=[3, 4], OutputWorkspace="sample_ws")
        test_sample_group.addWorkspace(mtd["sample_ws"])
        test_ws_name = "test_ws"
        alg = EVSCalibrationFit()
        alg.declareProperty(WorkspaceGroupProperty("Samples", "", Direction.Input))
        alg.declareProperty(WorkspaceGroupProperty("Background", "", Direction.Input))
        alg.declareProperty("OutputWorkspace", test_ws_name)

        alg.setProperty("Samples", test_sample_group)
        alg.setProperty("Background", test_background_group)

        alg._setup_run_numbers_and_output_workspace()
        self.assertFalse(hasattr(alg, "_background"))
        suffix = "_".join(ws.name() for ws in test_sample_group)
        self.assertEqual(test_ws_name + "_Sample_" + suffix, alg._sample)

    def test_setup_run_numbers_and_output_workspace_no_sample(self):
        test_sample_group = WorkspaceGroup()
        test_background_group = WorkspaceGroup()
        CreateWorkspace(DataX=[0, 1], DataY=[10], OutputWorkspace="background_ws")
        test_background_group.addWorkspace(mtd["background_ws"])
        test_ws_name = "test_ws"
        alg = EVSCalibrationFit()
        alg.declareProperty(WorkspaceGroupProperty("Samples", "", Direction.Input))
        alg.declareProperty(WorkspaceGroupProperty("Background", "", Direction.Input))
        alg.declareProperty("OutputWorkspace", test_ws_name)

        alg.setProperty("Samples", test_sample_group)
        alg.setProperty("Background", test_background_group)

        self.assertRaises(ValueError, alg._setup_run_numbers_and_output_workspace)

    def test_setup_function_type_gaussian(self):
        alg = EVSCalibrationFit()
        alg.declareProperty("Function", FitFunction.GAUSSIAN.value)
        alg._setup_function_type()
        expected_param_names = {"Height": "Height", "Width": "Sigma", "Position": "PeakCentre"}
        self.assertEqual(expected_param_names, alg._func_param_names)
        self.assertEqual({k: v + "_Err" for k, v in expected_param_names.items()}, alg._func_param_names_error)

    def test_setup_function_type_voigt(self):
        alg = EVSCalibrationFit()
        alg.declareProperty("Function", FitFunction.VOIGT.value)
        alg._setup_function_type()
        expected_param_names = {"Height": "LorentzAmp", "Position": "LorentzPos", "Width": "LorentzFWHM", "Width_2": "GaussianFWHM"}
        self.assertEqual(expected_param_names, alg._func_param_names)
        self.assertEqual({k: v + "Err" for k, v in expected_param_names.items()}, alg._func_param_names_error)

    def test_setup_function_type_error(self):
        alg = EVSCalibrationFit()
        alg.declareProperty("Function", "Error")
        self.assertRaises(ValueError, alg._setup_function_type)

    def test_setup_parameter_workspace(self):
        alg = EVSCalibrationFit()
        alg.declareProperty("InstrumentParameterWorkspace", "test_ws")
        alg.declareProperty("InstrumentParameterFile", "")
        alg._setup_parameter_workspace()
        self.assertEqual("test_ws", alg._param_workspace)
        self.assertEqual("test_ws", alg._param_table)

    @patch("mvesuvio.calibrate_fit.EVSMiscFunctions.load_instrument_parameters")
    def test_setup_parameter_workspace_no_ws(self, mock_load_instrument_parameters):
        alg = EVSCalibrationFit()
        alg.declareProperty("InstrumentParameterWorkspace", "")
        alg.declareProperty("InstrumentParameterFile", "/c/test_param_dir/file")
        alg._setup_parameter_workspace()
        mock_load_instrument_parameters.assert_called_once()
        self.assertEqual("file", alg._param_table)

    def test_setup_peaks_and_set_crop_and_fit_ranges_bragg(self):
        test_d_spacings = [3.5, 1.5, 2.5]
        alg = EVSCalibrationFit()
        alg.declareProperty(FloatArrayProperty("DSpacings", test_d_spacings))
        alg.declareProperty("PeakType", PeakType.BRAGG.value)
        alg._setup_peaks_and_set_crop_and_fit_ranges()
        self.assertEqual(sorted(test_d_spacings), list(alg._d_spacings))
        self.assertEqual(EVSGlobals.BRAGG_PEAK_CROP_RANGE, alg._ws_crop_range)
        self.assertEqual(EVSGlobals.BRAGG_FIT_WINDOW_RANGE, alg._fit_window_range)

    def test_setup_peaks_and_set_crop_and_fit_ranges_recoil(self):
        test_d_spacings = []
        alg = EVSCalibrationFit()
        alg.declareProperty(FloatArrayProperty("DSpacings", test_d_spacings))
        alg.declareProperty("PeakType", PeakType.RECOIL.value)
        alg._setup_peaks_and_set_crop_and_fit_ranges()
        self.assertEqual(EVSGlobals.RECOIL_PEAK_CROP_RANGE, alg._ws_crop_range)
        self.assertEqual(EVSGlobals.RECOIL_FIT_WINDOW_RANGE, alg._fit_window_range)

    def test_setup_peaks_and_set_crop_and_fit_ranges_resonance(self):
        test_d_spacings = []
        alg = EVSCalibrationFit()
        alg.declareProperty(FloatArrayProperty("DSpacings", test_d_spacings))
        alg.declareProperty("PeakType", PeakType.RESONANCE.value)
        alg._setup_peaks_and_set_crop_and_fit_ranges()
        self.assertEqual(EVSGlobals.RESONANCE_PEAK_CROP_RANGE, alg._ws_crop_range)
        self.assertEqual(EVSGlobals.RESONANCE_FIT_WINDOW_RANGE, alg._fit_window_range)

    @patch("mvesuvio.calibrate_fit.ReplaceSpecialValues")
    @patch("mvesuvio.calibrate_fit.EVSCalibrationFit._sum_and_crop_ws")
    def test_preprocess_no_bg(self, mock_sum_and_crop_ws, mock_replace_special_values):
        test_sample_group = WorkspaceGroup()
        CreateWorkspace(DataX=[0, 1], DataY=[1], OutputWorkspace="sample_ws_1")
        CreateWorkspace(DataX=[0, 1], DataY=[2], OutputWorkspace="sample_ws_2")
        test_sample_group.addWorkspace(mtd["sample_ws_1"])
        test_sample_group.addWorkspace(mtd["sample_ws_2"])
        test_crop_range = [3, 10]
        test_sample_ws_name = "test_ws"

        alg = EVSCalibrationFit()
        alg._ws_crop_range = test_crop_range
        alg._sample_run_numbers = test_sample_group
        alg._sample = test_sample_ws_name
        alg._bkg_run_numbers = WorkspaceGroup()
        alg._preprocess()
        mock_sum_and_crop_ws.assert_called_once_with(test_sample_group, test_sample_ws_name, test_crop_range[0], test_crop_range[-1])
        mock_replace_special_values.assert_called_once_with(
            test_sample_ws_name, NaNValue=0, NaNError=0, InfinityValue=0, InfinityError=0, OutputWorkspace=test_sample_ws_name
        )

    @patch("mvesuvio.calibrate_fit.EVSCalibrationFit._normalise_sample_by_background")
    @patch("mvesuvio.calibrate_fit.ReplaceSpecialValues")
    @patch("mvesuvio.calibrate_fit.EVSCalibrationFit._sum_and_crop_ws")
    def test_preprocess_with_bg(self, mock_sum_and_crop_ws, mock_replace_special_values, mock_normalise_sample):
        test_sample_group = WorkspaceGroup()
        test_background_group = WorkspaceGroup()
        CreateWorkspace(DataX=[0, 1], DataY=[1], OutputWorkspace="sample_ws_1")
        CreateWorkspace(DataX=[0, 1], DataY=[2], OutputWorkspace="sample_ws_2")
        CreateWorkspace(DataX=[0, 1], DataY=[5], OutputWorkspace="background_ws")
        test_sample_group.addWorkspace(mtd["sample_ws_1"])
        test_sample_group.addWorkspace(mtd["sample_ws_2"])
        test_background_group.addWorkspace(mtd["background_ws"])
        test_crop_range = [3, 10]
        test_sample_ws_name = "test_ws"

        alg = EVSCalibrationFit()
        alg._ws_crop_range = test_crop_range
        alg._sample_run_numbers = test_sample_group
        alg._sample = test_sample_ws_name
        alg._bkg_run_numbers = test_background_group
        alg._background = test_background_group[0].name()
        alg._preprocess()
        mock_sum_and_crop_ws.assert_has_calls(
            [
                call(test_sample_group, test_sample_ws_name, test_crop_range[0], test_crop_range[-1]),
                call(test_background_group, test_background_group[0].name(), test_crop_range[0], test_crop_range[-1]),
            ]
        )
        mock_normalise_sample.assert_called_once()
        mock_replace_special_values.assert_called_once_with(
            test_sample_ws_name, NaNValue=0, NaNError=0, InfinityValue=0, InfinityError=0, OutputWorkspace=test_sample_ws_name
        )

    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    @patch("mvesuvio.calibrate_fit.Divide")
    @patch("mvesuvio.calibrate_fit.RebinToWorkspace")
    def test_normalise_sample_by_background(self, mock_rebin, mock_divide, mock_delete):
        alg = EVSCalibrationFit()
        sample_ws = "test_ws"
        bg_ws = "bg_ws"

        alg._sample = sample_ws
        alg._background = bg_ws

        alg._normalise_sample_by_background()
        mock_rebin.assert_called_once_with(WorkspaceToRebin=bg_ws, WorkspaceToMatch=sample_ws, OutputWorkspace=bg_ws)
        mock_divide.assert_called_once_with(sample_ws, bg_ws, OutputWorkspace=sample_ws)
        mock_delete.assert_called_once_with(bg_ws)

    @patch("mvesuvio.calibrate_fit.CropWorkspace")
    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    @patch("mvesuvio.calibrate_fit.Plus")
    @patch("mvesuvio.calibrate_fit.CloneWorkspace")
    def test_sum_and_crop_ws(self, mock_clone, mock_plus, mock_delete, mock_crop):
        alg = EVSCalibrationFit()
        test_sample_group = WorkspaceGroup()
        CreateWorkspace(DataX=[0, 1], DataY=[1], OutputWorkspace="sample_ws_1")
        CreateWorkspace(DataX=[0, 1], DataY=[2], OutputWorkspace="sample_ws_2")
        test_sample_group.addWorkspace(mtd["sample_ws_1"])
        test_sample_group.addWorkspace(mtd["sample_ws_2"])
        output_name = "test_ws"
        xmin, xmax = 3, 7
        alg._sum_and_crop_ws(test_sample_group, output_name, xmin, xmax)
        mock_clone.assert_has_calls(
            [call(InputWorkspace=ANY, OutputWorkspace=output_name), call(InputWorkspace=ANY, OutputWorkspace="__EVS_calib_temp_ws")]
        )
        mock_plus.assert_has_calls([call(output_name, "__EVS_calib_temp_ws", OutputWorkspace="test_ws")])
        mock_delete.assert_has_calls([call("__EVS_calib_temp_ws")])
        mock_crop.assert_has_calls([call(output_name, XMin=xmin, XMax=xmax, OutputWorkspace=output_name)])

    @patch("mvesuvio.calibrate_fit.GroupWorkspaces")
    def test_fit_peaks_individual(self, group_workspaces_mock):
        alg = EVSCalibrationFit()
        alg._estimate_peak_positions = MagicMock(return_value=np.asarray([[5, 10, 15], [2.5, 7.5, 10.5]]))
        alg._create_parameter_table_and_output_headers = MagicMock(return_value=["a", "b", "c"])
        alg._fit_peak = MagicMock(side_effect=lambda a, b, c, d, e: f"fit_ws_{a}_{b}")
        alg._output_workspace_name = "output_ws_name"
        alg._shared_parameter_fit_type = SharedParameterFitType.INDIVIDUAL.value
        alg._fit_peaks()
        alg._create_parameter_table_and_output_headers.assert_has_calls(
            [call("output_ws_name_Peak_0_Parameters"), call("output_ws_name_Peak_1_Parameters")]
        )
        alg._fit_peak.assert_has_calls(
            [
                call(0, 0, 5, "output_ws_name_Peak_0_Parameters", ["a", "b", "c"]),
                call(0, 1, 10, "output_ws_name_Peak_0_Parameters", ["a", "b", "c"]),
                call(0, 2, 15, "output_ws_name_Peak_0_Parameters", ["a", "b", "c"]),
                call(1, 0, 2.5, "output_ws_name_Peak_1_Parameters", ["a", "b", "c"]),
                call(1, 1, 7.5, "output_ws_name_Peak_1_Parameters", ["a", "b", "c"]),
                call(1, 2, 10.5, "output_ws_name_Peak_1_Parameters", ["a", "b", "c"]),
            ]
        )
        self.assertEqual([["fit_ws_0_0", "fit_ws_0_1", "fit_ws_0_2"], ["fit_ws_1_0", "fit_ws_1_1", "fit_ws_1_2"]], alg._peak_fit_workspaces)
        group_workspaces_mock.assert_called_once_with(
            ["output_ws_name_Peak_0_Parameters", "output_ws_name_Peak_1_Parameters"], OutputWorkspace="output_ws_name_Peak_Parameters"
        )

    @patch("mvesuvio.calibrate_fit.EVSCalibrationFit._shared_parameter_fit")
    @patch("mvesuvio.calibrate_fit.GroupWorkspaces")
    def test_fit_peaks_shared(self, group_workspaces_mock, shared_parameter_fit_mock):
        alg = EVSCalibrationFit()
        alg._estimate_peak_positions = MagicMock(return_value=np.asarray([[5, 10, 15], [2.5, 7.5, 10.5]]))
        alg._create_parameter_table_and_output_headers = MagicMock(return_value=["a", "b", "c"])
        alg._fit_peak = MagicMock(side_effect=lambda a, b, c, d, e: f"fit_ws_{a}_{b}")
        alg._output_workspace_name = "output_ws_name"
        alg._shared_parameter_fit_type = SharedParameterFitType.SHARED.value
        alg._fit_peaks()
        alg._create_parameter_table_and_output_headers.assert_has_calls(
            [call("output_ws_name_Peak_0_Parameters"), call("output_ws_name_Peak_1_Parameters")]
        )
        alg._fit_peak.assert_has_calls(
            [
                call(0, 0, 5, "output_ws_name_Peak_0_Parameters", ["a", "b", "c"]),
                call(0, 1, 10, "output_ws_name_Peak_0_Parameters", ["a", "b", "c"]),
                call(0, 2, 15, "output_ws_name_Peak_0_Parameters", ["a", "b", "c"]),
                call(1, 0, 2.5, "output_ws_name_Peak_1_Parameters", ["a", "b", "c"]),
                call(1, 1, 7.5, "output_ws_name_Peak_1_Parameters", ["a", "b", "c"]),
                call(1, 2, 10.5, "output_ws_name_Peak_1_Parameters", ["a", "b", "c"]),
            ]
        )
        self.assertEqual([["fit_ws_0_0", "fit_ws_0_1", "fit_ws_0_2"], ["fit_ws_1_0", "fit_ws_1_1", "fit_ws_1_2"]], alg._peak_fit_workspaces)
        group_workspaces_mock.assert_called_once_with(
            ["output_ws_name_Peak_0_Parameters", "output_ws_name_Peak_1_Parameters"], OutputWorkspace="output_ws_name_Peak_Parameters"
        )
        shared_parameter_fit_mock.assert_called_once_with("output_ws_name_Peak_1_Parameters", ["a", "b", "c"])

    @patch("mvesuvio.calibrate_fit.EVSCalibrationFit._shared_parameter_fit")
    @patch("mvesuvio.calibrate_fit.GroupWorkspaces")
    def test_fit_peaks_both(self, group_workspaces_mock, shared_parameter_fit_mock):
        alg = EVSCalibrationFit()
        alg._estimate_peak_positions = MagicMock(return_value=np.asarray([[5, 10, 15], [2.5, 7.5, 10.5]]))
        alg._create_parameter_table_and_output_headers = MagicMock(return_value=["a", "b", "c"])
        alg._fit_peak = MagicMock(side_effect=lambda a, b, c, d, e: f"fit_ws_{a}_{b}")
        alg._output_workspace_name = "output_ws_name"
        alg._shared_parameter_fit_type = SharedParameterFitType.BOTH.value
        alg._fit_peaks()
        alg._create_parameter_table_and_output_headers.assert_has_calls(
            [call("output_ws_name_Peak_0_Parameters"), call("output_ws_name_Peak_1_Parameters")]
        )
        alg._fit_peak.assert_has_calls(
            [
                call(0, 0, 5, "output_ws_name_Peak_0_Parameters", ["a", "b", "c"]),
                call(0, 1, 10, "output_ws_name_Peak_0_Parameters", ["a", "b", "c"]),
                call(0, 2, 15, "output_ws_name_Peak_0_Parameters", ["a", "b", "c"]),
                call(1, 0, 2.5, "output_ws_name_Peak_1_Parameters", ["a", "b", "c"]),
                call(1, 1, 7.5, "output_ws_name_Peak_1_Parameters", ["a", "b", "c"]),
                call(1, 2, 10.5, "output_ws_name_Peak_1_Parameters", ["a", "b", "c"]),
            ]
        )
        self.assertEqual([["fit_ws_0_0", "fit_ws_0_1", "fit_ws_0_2"], ["fit_ws_1_0", "fit_ws_1_1", "fit_ws_1_2"]], alg._peak_fit_workspaces)
        group_workspaces_mock.assert_called_once_with(
            ["output_ws_name_Peak_0_Parameters", "output_ws_name_Peak_1_Parameters"], OutputWorkspace="output_ws_name_Peak_Parameters"
        )
        shared_parameter_fit_mock.assert_called_once_with("output_ws_name_Peak_1_Parameters", ["a", "b", "c"])

    @staticmethod
    def _setup_alg_mocks_fit_peak():
        alg = EVSCalibrationFit()
        alg._find_peaks_and_output_params = MagicMock(return_value="peak_params")
        alg._build_function_string = MagicMock(return_value="function_string")
        alg._find_fit_x_window = MagicMock(return_value=(0, 10))
        alg._output_fit_params_to_table_ws = MagicMock()
        alg._sample = "sample"
        alg._output_workspace_name = "output"
        alg._spec_list = [30]
        alg._prog_reporter = MagicMock()
        alg._del_fit_workspaces = MagicMock()

        return alg

    @patch("mvesuvio.calibrate_fit.Fit")
    def test_fit_peak(self, mock_fit):
        alg = self._setup_alg_mocks_fit_peak()

        fit_workspace = MagicMock()
        fit_workspace.name.return_value = "fws"
        fit_results = ("success", 0.5, "ncm", "fit_params", fit_workspace, "func", "cost_func")
        mock_fit.return_value = fit_results

        fit_workspace_name = alg._fit_peak(0, 0, 5, "output_Peak_0_Parameters", ["a", "b", "c"])

        alg._find_peaks_and_output_params.assert_called_once_with(0, 30, 0, 5)
        alg._build_function_string.assert_called_once_with("peak_params")
        alg._find_fit_x_window.assert_called_once_with("peak_params")
        mock_fit.assert_called_once_with(
            Function="function_string",
            InputWorkspace="sample",
            IgnoreInvalidData=True,
            StartX=0,
            EndX=10,
            WorkspaceIndex=0,
            CalcErrors=True,
            Output="__output_Peak_0_Spec_0",
            Minimizer="Levenberg-Marquardt,RelError=1e-8",
        )
        alg._output_fit_params_to_table_ws.assert_called_once_with(30, "fit_params", "output_Peak_0_Parameters", ["a", "b", "c"])
        alg._del_fit_workspaces.assert_called_once_with("ncm", "fit_params", fit_workspace)

        self.assertEqual(fit_workspace_name, "fws")

    @patch("mvesuvio.calibrate_fit.mtd")
    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    @patch("mvesuvio.calibrate_fit.FindPeaks")
    def test_find_peaks_and_output_params(self, find_peaks_mock, delete_workspace_mock, mtd_mock):
        alg = EVSCalibrationFit()
        alg._sample = "sample"
        find_peak_params_ret_val = {"a": 1, "b": 2, "c": 3}
        alg._get_find_peak_parameters = MagicMock(return_value=find_peak_params_ret_val)
        peak_table_name_ws = MagicMock()
        peak_table_name_ws.rowCount.return_value = 5
        mtd_mock.__getitem__.return_value = peak_table_name_ws

        alg._find_peaks_and_output_params(0, 30, 3, 5)

        alg._get_find_peak_parameters.assert_called_once_with(30, [5])
        find_peaks_mock.assert_called_once_with(
            InputWorkspace="sample", WorkspaceIndex=3, PeaksList="__sample_peaks_table_0_3", **find_peak_params_ret_val
        )
        peak_table_name_ws.rowCount.assert_called_once()
        delete_workspace_mock.assert_called_once_with("__sample_peaks_table_0_3")

    @patch("mvesuvio.calibrate_fit.sys")
    @patch("mvesuvio.calibrate_fit.logger.error")
    @patch("mvesuvio.calibrate_fit.mtd")
    @patch("mvesuvio.calibrate_fit.FindPeaks")
    @patch("mvesuvio.calibrate_fit.EVSCalibrationFit._get_find_peak_parameters")
    def test_find_peaks_and_output_params_no_peaks_found(self, find_peak_params_mock, find_peaks_mock, mtd_mock, logger_mock, sys_mock):
        alg = EVSCalibrationFit()
        alg._sample = "sample"
        find_peak_params_ret_val = {"a": 4, "b": 5, "c": 6}
        find_peak_params_mock.return_value = find_peak_params_ret_val
        peak_table_name_ws = MagicMock()
        peak_table_name_ws.rowCount.return_value = 0
        mtd_mock.__getitem__.return_value = peak_table_name_ws
        sys_mock.exit.side_effect = SystemExit("Emulate Sys Exit")

        with self.assertRaisesRegex(SystemExit, "Emulate Sys Exit"):
            alg._find_peaks_and_output_params(2, 32, 2, 7.5)

        find_peak_params_mock.assert_called_once_with(32, [7.5])
        find_peaks_mock.assert_called_once_with(
            InputWorkspace="sample", WorkspaceIndex=2, PeaksList="__sample_peaks_table_2_2", **find_peak_params_ret_val
        )
        peak_table_name_ws.rowCount.assert_called_once()
        logger_mock.assert_called_once()
        sys_mock.exit.assert_called_once()

    def test_find_fit_x_window(self):
        alg = EVSCalibrationFit()
        alg._func_param_names = {"Position": "Position_key"}
        alg._fit_window_range = 2

        xmin, xmax = alg._find_fit_x_window({"Position_key": 5})
        self.assertEqual(3, xmin)
        self.assertEqual(7, xmax)

    @patch("mvesuvio.calibrate_fit.logger.warning")
    def test_find_fit_x_window_position_less_than_1(self, logger_mock):
        alg = EVSCalibrationFit()
        alg._func_param_names = {"Position": "Position_key"}
        alg._fit_window_range = 2

        xmin, xmax = alg._find_fit_x_window({"Position_key": 0})
        self.assertEqual(None, xmin)
        self.assertEqual(None, xmax)
        logger_mock.assert_called_once()

    def _params_column_side_effect(self, col_index):
        if col_index == 0:
            return ["a", "b", "c"]
        elif col_index == 1:
            return [1, 2, 3]
        elif col_index == 2:
            return [0.1, 0.2, 0.3]
        else:
            raise ValueError("incorrect column index supplied")

    @patch("mvesuvio.calibrate_fit.mtd")
    def test_output_fit_params_to_table_ws(self, mtd_mock):
        alg = EVSCalibrationFit()
        spec_num = 45
        params_mock = MagicMock()
        params_mock.column.side_effect = self._params_column_side_effect
        output_table_name = "output_table"
        output_table_headers = ["a", "b", "c"]
        output_table_ws_mock = MagicMock()
        mtd_mock.__getitem__.return_value = output_table_ws_mock

        alg._output_fit_params_to_table_ws(spec_num, params_mock, output_table_name, output_table_headers)
        mtd_mock.__getitem__.assert_called_once_with(output_table_name)
        output_table_ws_mock.addRow.assert_called_once_with([spec_num, 1, 0.1, 2, 0.2, 3, 0.3])

    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    def test_del_fit_workspace(self, del_ws_mock):
        alg = EVSCalibrationFit()
        alg._create_output = True
        alg._del_fit_workspaces("ncm", "params", "fws")
        del_ws_mock.assert_has_calls([call("ncm"), call("params")])

    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    def test_del_fit_workspace_create_output_true(self, del_ws_mock):
        alg = EVSCalibrationFit()
        alg._create_output = False
        alg._del_fit_workspaces("ncm", "params", "fws")
        del_ws_mock.assert_has_calls([call("ncm"), call("params"), call("fws")])

    @patch("mvesuvio.calibrate_fit.CreateEmptyTableWorkspace")
    @patch("mvesuvio.calibrate_fit.AnalysisDataService")
    def test_create_output_parameters_table_ws(self, mock_ADS, mock_create_empty_table_ws):
        output_table_name = "test_output_table"
        num_estimated_peaks = 3
        alg = EVSCalibrationFit()
        alg._generate_column_headers = MagicMock(return_value=["col1", "col2", "col3"])
        mock_output_table = MagicMock()
        mock_create_empty_table_ws.return_value = mock_output_table

        alg._create_output_parameters_table_ws(output_table_name, num_estimated_peaks)
        mock_create_empty_table_ws.assert_called_once()
        mock_ADS.addOrReplace.assert_called_once_with(output_table_name, mock_output_table)

        mock_output_table.addColumn.assert_any_call("int", "Spectrum")
        for name in ["col1", "col2", "col3"]:
            mock_output_table.addColumn.assert_any_call("double", name)
            alg._generate_column_headers.assert_called_once_with(num_estimated_peaks)

    def test_get_param_names(self):
        num_estimated_peaks = 3
        alg = EVSCalibrationFit()
        alg._func_param_names = {"key1": "val1", "key2": "val2", "key3": "val3"}

        expected_param_names = ["f0.A0", "f0.A1"]
        for i in range(num_estimated_peaks):
            expected_param_names += ["f" + str(i) + "." + name for name in alg._func_param_names.values()]

        param_names = alg._get_param_names(num_estimated_peaks)
        self.assertEqual(param_names, expected_param_names)

    def _setup_select_best_fit_params(self):
        alg = EVSCalibrationFit()
        spec_num = 1

        fit_results = {"params": [1, 2, 3], "chi2": 10, "status": "success"}
        fit_results_u = {"params": [4, 5, 6], "chi2": 8, "status": "success"}

        alg._prog_reporter = MagicMock()
        alg._prog_reporter.report = MagicMock()

        return alg, spec_num, fit_results, fit_results_u

    def test_select_best_fit_params_unconstrained_is_better(self):
        alg, spec_num, fit_results, fit_results_u = self._setup_select_best_fit_params()

        selected_params, unconstrained_fit_selected = alg._select_best_fit_params(spec_num, fit_results, fit_results_u)

        self.assertEqual(selected_params, fit_results_u["params"])
        self.assertTrue(unconstrained_fit_selected)

    def test_select_best_fit_params_constrained_is_better(self):
        alg, spec_num, fit_results, fit_results_u = self._setup_select_best_fit_params()
        fit_results["chi2"] = 6

        selected_params, unconstrained_fit_selected = alg._select_best_fit_params(spec_num, fit_results, fit_results_u)
        self.assertEqual(selected_params, fit_results["params"])
        self.assertFalse(unconstrained_fit_selected)

    def test_select_best_fit_params_unconstrained_has_invalid_peaks(self):
        alg, spec_num, fit_results, fit_results_u = self._setup_select_best_fit_params()
        fit_results_u["status"] = "peaks invalid"

        selected_params, unconstrained_fit_selected = alg._select_best_fit_params(spec_num, fit_results, fit_results_u)
        self.assertEqual(selected_params, fit_results["params"])
        self.assertFalse(unconstrained_fit_selected)

    @patch("mvesuvio.calibrate_fit.mtd")
    def test_output_params_to_table(self, mock_mtd_module):
        alg = EVSCalibrationFit()
        spec_num = 1
        num_estimated_peaks = 3
        output_table_name = "test_output_table"

        alg._get_param_names = MagicMock(return_value=["A0", "A1", "param1", "param2", "param3"])

        param_values = [1, 2, 3, 4, 5]
        param_errors = [0.1, 0.2, 0.3, 0.4, 0.5]

        params = MagicMock()
        params.column.side_effect = lambda x: [["f0.A0", "f0.A1", "f1.param1", "f1.param2", "f1.param3"], param_values, param_errors][x]

        mock_output_table = MagicMock()
        mock_output_table.addRow = MagicMock()
        mock_mtd_module.__getitem__.return_value = mock_output_table

        alg._output_params_to_table(spec_num, num_estimated_peaks, params, output_table_name)

        alg._get_param_names.assert_called_once_with(num_estimated_peaks)
        params.column.assert_any_call(0)
        params.column.assert_any_call(1)
        params.column.assert_any_call(2)

        expected_row = [1, 0.1, 2, 0.2, 3, 0.3, 4, 0.4, 5, 0.5]
        mock_output_table.addRow.assert_called_once_with([spec_num] + expected_row)

    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    def test_get_output_and_clean_workspaces_unconstrained_not_performed(self, mock_delete_ws):
        alg = EVSCalibrationFit()
        find_peaks_output_name = "test_find_output_name"
        fit_peaks_output_name = "test_fit_output_name"
        output_ws = alg._get_output_and_clean_workspaces(False, False, False, find_peaks_output_name, fit_peaks_output_name)
        mock_delete_ws.assert_has_calls(
            [
                call(fit_peaks_output_name + "_NormalisedCovarianceMatrix"),
                call(fit_peaks_output_name + "_Parameters"),
                call(find_peaks_output_name),
            ]
        )
        self.assertEqual(output_ws, fit_peaks_output_name + "_Workspace")

    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    def test_get_output_and_clean_workspaces_unconstrained_performed(self, mock_delete_ws):
        alg = EVSCalibrationFit()
        find_peaks_output_name = "test_find_output_name"
        fit_peaks_output_name = "test_fit_output_name"
        output_ws = alg._get_output_and_clean_workspaces(True, True, False, find_peaks_output_name, fit_peaks_output_name)
        mock_delete_ws.assert_has_calls(
            [
                call(fit_peaks_output_name + "_NormalisedCovarianceMatrix"),
                call(fit_peaks_output_name + "_Parameters"),
                call(find_peaks_output_name),
                call(find_peaks_output_name + "_unconstrained"),
                call(fit_peaks_output_name + "_unconstrained_NormalisedCovarianceMatrix"),
                call(fit_peaks_output_name + "_unconstrained_Parameters"),
                call(fit_peaks_output_name + "_unconstrained_Workspace"),
            ]
        )
        self.assertEqual(output_ws, fit_peaks_output_name + "_Workspace")

    @patch("mvesuvio.calibrate_fit.DeleteWorkspace")
    def test_get_output_and_clean_workspaces_unconstrained_performed_and_selected(self, mock_delete_ws):
        alg = EVSCalibrationFit()
        find_peaks_output_name = "test_find_output_name"
        fit_peaks_output_name = "test_fit_output_name"
        output_ws = alg._get_output_and_clean_workspaces(True, True, True, find_peaks_output_name, fit_peaks_output_name)
        mock_delete_ws.assert_has_calls(
            [
                call(fit_peaks_output_name + "_NormalisedCovarianceMatrix"),
                call(fit_peaks_output_name + "_Parameters"),
                call(find_peaks_output_name),
                call(find_peaks_output_name + "_unconstrained"),
                call(fit_peaks_output_name + "_unconstrained_NormalisedCovarianceMatrix"),
                call(fit_peaks_output_name + "_unconstrained_Parameters"),
                call(fit_peaks_output_name + "_Workspace"),
            ]
        )
        self.assertEqual(output_ws, fit_peaks_output_name + "_unconstrained" + "_Workspace")

    def test_generate_column_headers(self):
        alg = EVSCalibrationFit()
        num_estimated_peaks = 3

        alg._get_param_names = MagicMock(return_value=["A0", "A1", "val1", "val2", "val3"])

        col_headers = alg._generate_column_headers(num_estimated_peaks)

        alg._get_param_names.assert_called_once_with(num_estimated_peaks)

        expected_col_headers = ["A0", "A0_Err", "A1", "A1_Err", "val1", "val1_Err", "val2", "val2_Err", "val3", "val3_Err"]
        self.assertEqual(col_headers, expected_col_headers)

    def test_get_unconstrained_ws_name(self):
        alg = EVSCalibrationFit()
        ws_name = "test"
        return_ws_name = alg._get_unconstrained_ws_name(ws_name)
        self.assertEqual(ws_name + "_unconstrained", return_ws_name)

    def _setup_run_find_peaks_test(self, unconstrained):
        alg = EVSCalibrationFit()
        alg._sample = "sample_workspace"
        workspace_index = 0
        find_peaks_output_name = "peaks_list"
        find_peaks_input_params = {"Param1": 1, "Param2": 3}
        fn_args = {
            "workspace_index": workspace_index,
            "find_peaks_output_name": find_peaks_output_name,
            "find_peaks_input_params": find_peaks_input_params,
            "unconstrained": unconstrained,
        }
        return alg, fn_args

    @staticmethod
    def _setup_mtd_mock(mtd_mock_obj, find_peaks_name, peaks_found):
        mock_find_peaks_output = MagicMock()
        mock_find_peaks_output.rowCount.return_value = peaks_found
        mtd_mock_obj.__getitem__.side_effect = lambda name: mock_find_peaks_output if name == find_peaks_name else None

    @patch("mvesuvio.calibrate_fit.FindPeaks")
    @patch("mvesuvio.calibrate_fit.mtd")
    def test_run_find_peaks_peaks_found(self, mock_mtd_module, mock_find_peaks):
        alg, fn_args = self._setup_run_find_peaks_test(unconstrained=False)

        self._setup_mtd_mock(mock_mtd_module, fn_args["find_peaks_output_name"], 1)

        result = alg._run_find_peaks(**fn_args)
        mock_find_peaks.assert_called_once()
        self.assertTrue(result)

    @patch("mvesuvio.calibrate_fit.FindPeaks")
    @patch("mvesuvio.calibrate_fit.mtd")
    def test_run_find_peaks_no_peaks_found_raises_value_error(self, mock_mtd_module, mock_find_peaks):
        alg, fn_args = self._setup_run_find_peaks_test(unconstrained=False)

        self._setup_mtd_mock(mock_mtd_module, fn_args["find_peaks_output_name"], 0)

        with self.assertRaises(ValueError, msg="Error finding peaks."):
            alg._run_find_peaks(**fn_args)
        mock_find_peaks.assert_called_once()

    @patch("mvesuvio.calibrate_fit.FindPeaks")
    @patch("mvesuvio.calibrate_fit.mtd")
    def test_run_find_peaks_unconstrained_no_peaks_found_no_error(self, mock_mtd_module, mock_find_peaks):
        alg, fn_args = self._setup_run_find_peaks_test(unconstrained=True)

        self._setup_mtd_mock(mock_mtd_module, fn_args["find_peaks_output_name"], 0)

        result = alg._run_find_peaks(**fn_args)
        mock_find_peaks.assert_called_once()
        self.assertFalse(result)

    @patch("mvesuvio.calibrate_fit.FindPeaks")
    def test_run_find_peaks_unconstrained_peaks_found_raises_error(self, mock_find_peaks):
        alg, fn_args = self._setup_run_find_peaks_test(unconstrained=True)

        mock_find_peaks.side_effect = ValueError

        result = alg._run_find_peaks(**fn_args)
        mock_find_peaks.assert_called_once()
        self.assertFalse(result)

    @staticmethod
    def _setup_filter_and_fit_found_peaks_mocks():
        alg = EVSCalibrationFit()
        workspace_index = 0
        peak_estimates_list = "peak_estimates_list"
        find_peaks_output_name = "find_peaks_output"
        fit_peaks_output_name = "fit_peaks_output"
        x_range = (1, 2)

        alg._calc_linear_bg_coefficients = MagicMock(return_value="linear_bg_coeffs")
        alg._filter_found_peaks = MagicMock()
        alg._fit_found_peaks = MagicMock(return_value={"status": "test_status"})
        alg._check_fitted_peak_validity = MagicMock(return_value=True)

        return alg, workspace_index, peak_estimates_list, find_peaks_output_name, fit_peaks_output_name, x_range

    def test_filter_and_fit_found_peaks_unconstrained(self):
        alg, workspace_index, peak_estimates_list, find_peaks_output_name, fit_peaks_output_name, x_range = (
            self._setup_filter_and_fit_found_peaks_mocks()
        )

        fit_results = alg._filter_and_fit_found_peaks(
            workspace_index, peak_estimates_list, find_peaks_output_name, fit_peaks_output_name, x_range, unconstrained=True
        )

        alg._calc_linear_bg_coefficients.assert_called_once()
        alg._filter_found_peaks.assert_called_once_with(
            find_peaks_output_name,
            peak_estimates_list,
            "linear_bg_coeffs",
            peak_height_rel_threshold=EVSGlobals.PEAK_HEIGHT_RELATIVE_THRESHOLD,
        )
        alg._fit_found_peaks.assert_called_once_with(find_peaks_output_name, None, workspace_index, fit_peaks_output_name, x_range)
        alg._check_fitted_peak_validity.assert_called_once_with(
            fit_peaks_output_name + "_Parameters", peak_estimates_list, peak_height_rel_threshold=EVSGlobals.PEAK_HEIGHT_RELATIVE_THRESHOLD
        )
        self.assertEqual(fit_results["status"], "test_status")

    def test_filter_and_fit_found_peaks_not_unconstrained(self):
        alg, workspace_index, peak_estimates_list, find_peaks_output_name, fit_peaks_output_name, x_range = (
            self._setup_filter_and_fit_found_peaks_mocks()
        )

        fit_results = alg._filter_and_fit_found_peaks(
            workspace_index, peak_estimates_list, find_peaks_output_name, fit_peaks_output_name, x_range, unconstrained=False
        )

        alg._calc_linear_bg_coefficients.assert_not_called()
        alg._filter_found_peaks.assert_not_called()
        alg._fit_found_peaks.assert_called_once_with(
            find_peaks_output_name, peak_estimates_list, workspace_index, fit_peaks_output_name, x_range
        )
        alg._check_fitted_peak_validity.assert_called_once_with(
            fit_peaks_output_name + "_Parameters", peak_estimates_list, peak_height_rel_threshold=EVSGlobals.PEAK_HEIGHT_RELATIVE_THRESHOLD
        )
        self.assertEqual(fit_results["status"], "test_status")

    def test_filter_and_fit_found_peaks_invalid_peaks(self):
        alg, workspace_index, peak_estimates_list, find_peaks_output_name, fit_peaks_output_name, x_range = (
            self._setup_filter_and_fit_found_peaks_mocks()
        )
        alg._check_fitted_peak_validity.return_value = False

        fit_results = alg._filter_and_fit_found_peaks(
            workspace_index, peak_estimates_list, find_peaks_output_name, fit_peaks_output_name, x_range, unconstrained=False
        )

        alg._calc_linear_bg_coefficients.assert_not_called()
        alg._filter_found_peaks.assert_not_called()
        alg._fit_found_peaks.assert_called_once_with(
            find_peaks_output_name, peak_estimates_list, workspace_index, fit_peaks_output_name, x_range
        )
        alg._check_fitted_peak_validity.assert_called_once_with(
            fit_peaks_output_name + "_Parameters", peak_estimates_list, peak_height_rel_threshold=EVSGlobals.PEAK_HEIGHT_RELATIVE_THRESHOLD
        )
        self.assertEqual(fit_results["status"], "peaks invalid")

    @staticmethod
    def _setup_fit_peaks_to_spectra_mocks(peaks_found):
        alg = EVSCalibrationFit()
        workspace_index = 0
        spec_number = 1
        peak_estimates_list = "peak_estimates_list"
        find_peaks_output_name = "find_peaks_output"
        fit_peaks_output_name = "fit_peaks_output"
        x_range = (1, 2)

        alg._get_unconstrained_ws_name = MagicMock(side_effect=lambda x: x + "_unconstrained")
        alg._get_find_peak_parameters = MagicMock(return_value="find_peaks_params")
        alg._run_find_peaks = MagicMock(return_value=peaks_found)
        alg._filter_and_fit_found_peaks = MagicMock(return_value="fit_results")

        return alg, workspace_index, spec_number, peak_estimates_list, find_peaks_output_name, fit_peaks_output_name, x_range

    def test_fit_peaks_to_spectra_unconstrained_peaks_found(self):
        alg, workspace_index, spec_number, peak_estimates_list, find_peaks_output_name, fit_peaks_output_name, x_range = (
            self._setup_fit_peaks_to_spectra_mocks(peaks_found=True)
        )

        result = alg._fit_peaks_to_spectra(
            workspace_index,
            spec_number,
            peak_estimates_list,
            find_peaks_output_name,
            fit_peaks_output_name,
            unconstrained=True,
            x_range=x_range,
        )

        alg._get_find_peak_parameters.assert_called_once_with(spec_number, peak_estimates_list, True)
        alg._run_find_peaks.assert_called_once_with(workspace_index, find_peaks_output_name + "_unconstrained", "find_peaks_params", True)
        alg._filter_and_fit_found_peaks.assert_called_once_with(
            workspace_index,
            peak_estimates_list,
            find_peaks_output_name + "_unconstrained",
            fit_peaks_output_name + "_unconstrained",
            x_range,
            True,
        )
        self.assertEqual(result, "fit_results")

    def test_fit_peaks_to_spectra_not_unconstrained_peaks_found(self):
        alg, workspace_index, spec_number, peak_estimates_list, find_peaks_output_name, fit_peaks_output_name, x_range = (
            self._setup_fit_peaks_to_spectra_mocks(peaks_found=True)
        )

        result = alg._fit_peaks_to_spectra(
            workspace_index,
            spec_number,
            peak_estimates_list,
            find_peaks_output_name,
            fit_peaks_output_name,
            unconstrained=False,
            x_range=x_range,
        )

        alg._get_find_peak_parameters.assert_called_once_with(spec_number, peak_estimates_list, False)
        alg._run_find_peaks.assert_called_once_with(workspace_index, find_peaks_output_name, "find_peaks_params", False)
        alg._filter_and_fit_found_peaks.assert_called_once_with(
            workspace_index, peak_estimates_list, find_peaks_output_name, fit_peaks_output_name, x_range, False
        )
        self.assertEqual(result, "fit_results")

    def test_fit_peaks_to_spectra_not_unconstrained_peaks_not_found(self):
        alg, workspace_index, spec_number, peak_estimates_list, find_peaks_output_name, fit_peaks_output_name, x_range = (
            self._setup_fit_peaks_to_spectra_mocks(peaks_found=False)
        )

        result = alg._fit_peaks_to_spectra(
            workspace_index,
            spec_number,
            peak_estimates_list,
            find_peaks_output_name,
            fit_peaks_output_name,
            unconstrained=False,
            x_range=x_range,
        )

        alg._get_find_peak_parameters.assert_called_once_with(spec_number, peak_estimates_list, False)
        alg._run_find_peaks.assert_called_once_with(workspace_index, find_peaks_output_name, "find_peaks_params", False)
        alg._filter_and_fit_found_peaks.assert_not_called()
        self.assertEqual(result, False)

    @staticmethod
    def _setup_fit_bragg_peaks_mocks(fit_results, selected_params, output_workspaces):
        alg = EVSCalibrationFit()
        alg._estimate_bragg_peak_positions = MagicMock(return_value=np.array([[1, 2], [3, 4]]).transpose())
        alg._create_output_parameters_table_ws = MagicMock()
        alg._fit_peaks_to_spectra = MagicMock(side_effect=fit_results)
        alg._select_best_fit_params = MagicMock(side_effect=[(selected_params[0], False), (selected_params[1], False)])
        alg._output_params_to_table = MagicMock()
        alg._get_output_and_clean_workspaces = MagicMock(side_effect=output_workspaces)
        alg._spec_list = [1]
        alg._sample = "sample"
        alg._output_workspace_name = "output"
        alg._create_output = True
        alg._prog_reporter = MagicMock()

        return alg

    def _test_calls_individually(self, call_list, calls):
        for j, _call in enumerate(call_list):
            for i, arg in enumerate(_call.args):
                expected_arg = calls[j][i]
                try:
                    self.assertEqual(arg, expected_arg)
                except ValueError:
                    np.testing.assert_array_equal(arg, expected_arg)

    @patch("mvesuvio.calibrate_fit.GroupWorkspaces")
    def test_fit_bragg_peaks_success(self, group_workspaces_mock):
        fit_result_ret_val = {"status": "success"}
        fit_results = lambda *args: fit_result_ret_val
        selected_params = ["selected_params1", "selected_params2"]
        output_workspaces = ["ws1", "ws2"]

        alg = self._setup_fit_bragg_peaks_mocks(fit_results, selected_params, output_workspaces)
        alg._fit_bragg_peaks()

        fit_peaks_to_spectra_call_list = alg._fit_peaks_to_spectra.call_args_list
        self._test_calls_individually(
            fit_peaks_to_spectra_call_list,
            [
                [0, 1, [1, 2], "sample_peaks_table_1", "output_Spec_1", False],
                [1, 2, [3, 4], "sample_peaks_table_2", "output_Spec_2", False],
            ],
        )
        alg._select_best_fit_params.assert_has_calls([call(1, fit_result_ret_val, None), call(2, fit_result_ret_val, None)])
        alg._output_params_to_table.assert_has_calls(
            [call(1, 2, selected_params[0], "output_Peak_Parameters"), call(2, 2, selected_params[1], "output_Peak_Parameters")]
        )
        alg._get_output_and_clean_workspaces.assert_has_calls(
            [
                call(False, False, False, "sample_peaks_table_1", "output_Spec_1"),
                call(False, False, False, "sample_peaks_table_2", "output_Spec_2"),
            ]
        )
        group_workspaces_mock.assert_called_once_with(",".join(output_workspaces), OutputWorkspace="output_Peak_Fits")

    @patch("mvesuvio.calibrate_fit.GroupWorkspaces")
    def test_fit_bragg_peaks_not_success(self, group_workspaces_mock):
        x_range = (1, 2)
        fit_res = {"status": "failure", "xmin": x_range[0], "xmax": x_range[1]}
        rit_u_res = {"status": "success"}
        fit_results = [fit_res, rit_u_res, fit_res, rit_u_res]
        selected_params = ["selected_params1", "selected_params2"]
        output_workspaces = ["ws1", "ws2"]

        alg = self._setup_fit_bragg_peaks_mocks(fit_results, selected_params, output_workspaces)
        alg._fit_bragg_peaks()

        fit_peaks_to_spectra_call_list = alg._fit_peaks_to_spectra.call_args_list
        self._test_calls_individually(
            fit_peaks_to_spectra_call_list,
            [
                [0, 1, [1, 2], "sample_peaks_table_1", "output_Spec_1", False],
                [0, 1, [1, 2], "sample_peaks_table_1", "output_Spec_1", True, x_range],
                [1, 2, [3, 4], "sample_peaks_table_2", "output_Spec_2", False],
                [1, 2, [3, 4], "sample_peaks_table_2", "output_Spec_2", True, x_range],
            ],
        )
        alg._select_best_fit_params.assert_has_calls([call(1, fit_res, rit_u_res), call(2, fit_res, rit_u_res)])
        alg._output_params_to_table.assert_has_calls(
            [call(1, 2, selected_params[0], "output_Peak_Parameters"), call(2, 2, selected_params[1], "output_Peak_Parameters")]
        )
        alg._get_output_and_clean_workspaces.assert_has_calls(
            [
                call(True, True, False, "sample_peaks_table_1", "output_Spec_1"),
                call(True, True, False, "sample_peaks_table_2", "output_Spec_2"),
            ]
        )
        group_workspaces_mock.assert_called_once_with(",".join(output_workspaces), OutputWorkspace="output_Peak_Fits")


if __name__ == "__main__":
    unittest.main()
