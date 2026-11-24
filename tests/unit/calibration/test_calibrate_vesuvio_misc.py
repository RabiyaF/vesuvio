import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from mvesuvio.util.calibration_helpers import EVSMiscFunctions, InvalidDetectors


class TestVesuvioCalibrationMisc(unittest.TestCase):
    def test_generate_header_function_gaussian(self):
        header = EVSMiscFunctions.generate_fit_function_header("Gaussian")
        self.assertEqual({"Height": "Height", "Width": "Sigma", "Position": "PeakCentre"}, header)

    def test_generate_header_function_gaussian_with_error(self):
        header = EVSMiscFunctions.generate_fit_function_header("Gaussian", error=True)
        self.assertEqual({"Height": "Height_Err", "Width": "Sigma_Err", "Position": "PeakCentre_Err"}, header)

    def test_generate_header_function_voigt(self):
        header = EVSMiscFunctions.generate_fit_function_header("Voigt")
        self.assertEqual({"Height": "LorentzAmp", "Position": "LorentzPos", "Width": "LorentzFWHM", "Width_2": "GaussianFWHM"}, header)

    def test_generate_header_function_voigt_with_error(self):
        header = EVSMiscFunctions.generate_fit_function_header("Voigt", error=True)
        self.assertEqual(
            {"Height": "LorentzAmpErr", "Position": "LorentzPosErr", "Width": "LorentzFWHMErr", "Width_2": "GaussianFWHMErr"}, header
        )

    def test_generate_header_function_invalid(self):
        with self.assertRaises(ValueError):
            EVSMiscFunctions.generate_fit_function_header("Invalid")

    @patch("mvesuvio.util.calibration_helpers.mtd")
    def test_identify_invalid_spectra_no_invalid(self, mock_mtd):
        ws_mock = MagicMock()
        mock_column_output_dict = {
            "f1.GaussianFWHM": [1, 2, 3],
            "f1.GaussianFWHM_Err": [0.1, 0.2, 0.3],
            "f1.LorentzFWHM": [1.1, 2.1, 3.1],
            "f1.LorentzFWHM_Err": [0.11, 0.21, 0.31],
            "f1.LorentzAmp": [1.2, 2.2, 3.2],
            "f1.LorentzAmp_Err": [0.12, 0.22, 0.32],
        }
        ws_mock.column.side_effect = lambda key: mock_column_output_dict[key]
        mock_mtd.__getitem__.return_value = ws_mock

        np.testing.assert_equal(
            np.argwhere([]), InvalidDetectors._identify_invalid_spectra("peak_table", [5, 10, 20], [0.1, 0.15, 0.2], [0, 2])
        )

    @patch("mvesuvio.util.calibration_helpers.mtd")
    def test_identify_invalid_spectra_nan_in_errors(self, mock_mtd):
        ws_mock = MagicMock()
        mock_column_output_dict = {
            "f1.GaussianFWHM": [1, 2, 3],
            "f1.GaussianFWHM_Err": [np.nan, 0.2, 0.3],
            "f1.LorentzFWHM": [1.1, 2.1, 3.1],
            "f1.LorentzFWHM_Err": [0.11, 0.21, 0.31],
            "f1.LorentzAmp": [1.2, 2.2, 3.2],
            "f1.LorentzAmp_Err": [0.12, 0.22, np.nan],
        }
        ws_mock.column.side_effect = lambda key: mock_column_output_dict[key]
        mock_mtd.__getitem__.return_value = ws_mock

        np.testing.assert_equal(
            np.argwhere(np.array([True, False, True])),
            InvalidDetectors._identify_invalid_spectra("peak_table", [5, 10, 20], [0.1, 0.15, 0.2], [0, 2]),
        )

    @patch("mvesuvio.util.calibration_helpers.mtd")
    def test_identify_invalid_spectra_inf_in_errors(self, mock_mtd):
        ws_mock = MagicMock()
        mock_column_output_dict = {
            "f1.GaussianFWHM": [1, 2, 3],
            "f1.GaussianFWHM_Err": [0.1, 0.2, 0.3],
            "f1.LorentzFWHM": [1.1, 2.1, 3.1],
            "f1.LorentzFWHM_Err": [np.inf, 0.21, 0.31],
            "f1.LorentzAmp": [1.2, 2.2, 3.2],
            "f1.LorentzAmp_Err": [0.12, np.inf, 0.32],
        }
        ws_mock.column.side_effect = lambda key: mock_column_output_dict[key]
        mock_mtd.__getitem__.return_value = ws_mock

        np.testing.assert_equal(
            np.argwhere(np.array([True, True, False])),
            InvalidDetectors._identify_invalid_spectra("peak_table", [5, 10, 20], [0.1, 0.15, 0.2], [0, 2]),
        )

    @patch("mvesuvio.util.calibration_helpers.mtd")
    def test_identify_invalid_spectra_error_greater_than_peak(self, mock_mtd):
        ws_mock = MagicMock()
        mock_column_output_dict = {
            "f1.GaussianFWHM": [1, 2, 3],
            "f1.GaussianFWHM_Err": [0.1, 0.2, 0.3],
            "f1.LorentzFWHM": [1.1, 2.1, 3.1],
            "f1.LorentzFWHM_Err": [0.11, 10, 0.31],
            "f1.LorentzAmp": [1.2, 2.2, 3.2],
            "f1.LorentzAmp_Err": [0.12, 0.22, 10],
        }
        ws_mock.column.side_effect = lambda key: mock_column_output_dict[key]
        mock_mtd.__getitem__.return_value = ws_mock

        np.testing.assert_equal(
            np.argwhere(np.array([False, True, True])),
            InvalidDetectors._identify_invalid_spectra("peak_table", [5, 10, 20], [0.1, 0.15, 0.2], [0, 2]),
        )

    def test_create_empty_invalid_detectors(self):
        invalid_detectors = InvalidDetectors([])
        self.assertEqual(invalid_detectors._invalid_detectors_back.tolist(), [])
        self.assertEqual(invalid_detectors._invalid_detectors_front.tolist(), [])
        self.assertEqual(invalid_detectors._detectors_preset, False)

    def test_create_invalid_detectors_back(self):
        invalid_detectors = InvalidDetectors([10, 20, 30])
        self.assertEqual(invalid_detectors._invalid_detectors_back.tolist(), [[7], [17], [27]])
        self.assertEqual(invalid_detectors._invalid_detectors_front.tolist(), [])
        self.assertEqual(invalid_detectors._detectors_preset, True)

    def test_create_invalid_detectors_front(self):
        invalid_detectors = InvalidDetectors([150, 160, 170])
        self.assertEqual(invalid_detectors._invalid_detectors_back.tolist(), [])
        self.assertEqual(invalid_detectors._invalid_detectors_front.tolist(), [[15], [25], [35]])
        self.assertEqual(invalid_detectors._detectors_preset, True)

    def test_create_invalid_detectors(self):
        invalid_detectors = InvalidDetectors([10, 20, 30, 150, 160, 170])
        self.assertEqual(invalid_detectors._invalid_detectors_back.tolist(), [[7], [17], [27]])
        self.assertEqual(invalid_detectors._invalid_detectors_front.tolist(), [[15], [25], [35]])

    def test_get_all_detectors(self):
        input_invalid_detectors = [10, 20, 30, 150, 160, 170]
        invalid_detectors = InvalidDetectors([10, 20, 30, 150, 160, 170])
        self.assertEqual(invalid_detectors.get_all_invalid_detectors(), input_invalid_detectors)

    @patch("mvesuvio.util.calibration_helpers.EVSMiscFunctions.read_table_column")
    def test_filter_peak_centres_for_invalid_detectors_front(self, mock_read_table_column):
        invalid_detectors = InvalidDetectors([10, 20, 30, 150, 160, 170])
        peak_table = "input_peak_table"
        mock_read_table_column.return_value = np.array([[float(x)] for x in range(3, 198, 1)])

        out_peak_centres = invalid_detectors.filter_peak_centres_for_invalid_detectors([3, 134], peak_table)
        self.assertEqual(list(np.argwhere(np.isnan(out_peak_centres)).transpose()[0]), [7, 17, 27])

    @patch("mvesuvio.util.calibration_helpers.EVSMiscFunctions.read_table_column")
    def test_filter_peak_centres_for_invalid_detectors_back(self, mock_read_table_column):
        invalid_detectors = InvalidDetectors([10, 20, 30, 150, 160, 170])
        peak_table = "input_peak_table"
        mock_read_table_column.return_value = np.array([[float(x)] for x in range(3, 198, 1)])

        out_peak_centres = invalid_detectors.filter_peak_centres_for_invalid_detectors([135, 198], peak_table)
        self.assertEqual(list(np.argwhere(np.isnan(out_peak_centres)).transpose()[0]), [15, 25, 35])

    @patch("mvesuvio.util.calibration_helpers.EVSMiscFunctions.read_table_column")
    def test_filter_peak_centres_for_invalid_detectors_invalid_range(self, mock_read_table_column):
        invalid_detectors = InvalidDetectors([10, 20, 30, 150, 160, 170])
        peak_table = "input_peak_table"
        with self.assertRaises(AttributeError):
            invalid_detectors.filter_peak_centres_for_invalid_detectors([10, 20], peak_table)

    def test_get_invalid_detectors_index_back(self):
        invalid_detectors = InvalidDetectors([10, 20, 30, 150, 160, 170])
        invalid_detectors_index = invalid_detectors.get_invalid_detectors_index([3, 134])
        self.assertEqual(invalid_detectors_index, [7, 17, 27])

    def test_get_invalid_detectors_index_front(self):
        invalid_detectors = InvalidDetectors([10, 20, 30, 150, 160, 170])
        invalid_detectors_index = invalid_detectors.get_invalid_detectors_index([135, 198])
        self.assertEqual(invalid_detectors_index, [15, 25, 35])

    def test_get_invalid_detectors_index_invalid_range(self):
        invalid_detectors = InvalidDetectors([10, 20, 30, 150, 160, 170])
        with self.assertRaises(AttributeError):
            invalid_detectors.get_invalid_detectors_index([10, 20])

    def test_add_invalid_detectors(self):
        invalid_detectors = InvalidDetectors([10, 20, 30, 150, 160, 170])
        invalid_detectors.add_invalid_detectors([10, 20, 25, 30, 180, 190])
        self.assertEqual(invalid_detectors.get_all_invalid_detectors(), [10, 20, 25, 30, 150, 160, 170, 180, 190])

    @patch("mvesuvio.util.calibration_helpers.InvalidDetectors._identify_invalid_spectra")
    @patch("mvesuvio.util.calibration_helpers.EVSMiscFunctions.read_table_column")
    def test_add_invalid_detectors_no_preset_identify_called_front(self, mock_read_table_column, mock_identify):
        invalid_detectors = InvalidDetectors([])
        invalid_detectors.add_invalid_detectors([180, 190])
        self.assertEqual(invalid_detectors.get_all_invalid_detectors(), [180, 190])
        self.assertEqual(invalid_detectors._detectors_preset, False)

        peak_table = "input_peak_table"
        invalid_detectors.filter_peak_centres_for_invalid_detectors([3, 134], peak_table)
        mock_identify.assert_called_once()

    @patch("mvesuvio.util.calibration_helpers.InvalidDetectors._identify_invalid_spectra")
    @patch("mvesuvio.util.calibration_helpers.EVSMiscFunctions.read_table_column")
    def test_add_invalid_detectors_no_preset_identify_called_back(self, mock_read_table_column, mock_identify):
        invalid_detectors = InvalidDetectors([])
        invalid_detectors.add_invalid_detectors([31, 32])
        self.assertEqual(invalid_detectors.get_all_invalid_detectors(), [31, 32])
        self.assertEqual(invalid_detectors._detectors_preset, False)

        peak_table = "input_peak_table"
        invalid_detectors.filter_peak_centres_for_invalid_detectors([135, 198], peak_table)
        mock_identify.assert_called_once()

    @patch("mvesuvio.util.calibration_helpers.InvalidDetectors._identify_invalid_spectra")
    @patch("mvesuvio.util.calibration_helpers.EVSMiscFunctions.read_table_column")
    def test_add_invalid_detectors_preset_identify_not_called(self, mock_read_table_column, mock_identify):
        invalid_detectors = InvalidDetectors([180, 190])
        self.assertEqual(invalid_detectors.get_all_invalid_detectors(), [180, 190])
        self.assertEqual(invalid_detectors._detectors_preset, True)

        peak_table = "input_peak_table"
        invalid_detectors.filter_peak_centres_for_invalid_detectors([3, 134], peak_table)
        mock_identify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
