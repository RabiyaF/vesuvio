import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import dill
import numpy as np
import scipy
from mantid.simpleapi import AnalysisDataService

from mvesuvio.util.analysis_helpers import (
    extend_range_of_array,
    load_instrument_params,
    load_resolution,
    make_gamma_correction_input_string,
    make_multiple_scattering_input_string,
    numerical_third_derivative,
    pseudo_voigt,
)
from mvesuvio.util.general_helpers import print_table_workspace


class TestAnalysisHelpers(unittest.TestCase):
    def tearDown(self):
        AnalysisDataService.clear()

    def test_conversion_of_constraints(self):
        constraints = (
            {"type": "eq", "fun": lambda par: par[0] - 2.7527 * par[3]},
            {"type": "eq", "fun": lambda par: par[3] - 0.7234 * par[6]},
        )
        # Used before passing constraints into Mantid algorithm
        string_constraints = str(dill.dumps(constraints))
        self.assertIsInstance(string_constraints, str)
        # Used inside Mantid algorithm to convert back to SciPy constraints
        converted_constraints = dill.loads(eval(string_constraints))
        self.assertEqual(converted_constraints[0]["fun"]([3, 0, 0, 1]), 3 - 2.7527)
        self.assertEqual(converted_constraints[1]["fun"]([0, 0, 0, 2, 0, 0, 1]), 2 - 0.7234)

    def test_extend_range_of_array_for_increasing_range(self):
        x = np.arange(10)
        x = np.vstack([x, 2 * x])
        x_extended = extend_range_of_array(x, 5)
        np.testing.assert_array_equal(x_extended, np.vstack([np.arange(-5, 15, 1), np.arange(-10, 30, 2)]))

    def test_extend_range_of_array_for_decreasing_range(self):
        x = np.linspace(-5, 5, 21)
        x = np.vstack([x, 2 * x])
        x_extended = extend_range_of_array(x, 5)
        np.testing.assert_array_equal(x_extended, np.vstack([np.linspace(-7.5, 7.5, 31), np.linspace(-15, 15, 31)]))

    def test_pseudo_voigt(self):
        x = np.linspace(-20, 20, 10)
        actual_result = pseudo_voigt(x, 5, 4)
        expected_result = np.array(
            [0.00360493, 0.00695558, 0.01490388, 0.02897969, 0.04368826, 0.04368826, 0.02897969, 0.01490388, 0.00695558, 0.00360493]
        )
        np.testing.assert_allclose(actual_result, expected_result, atol=1e-6)

    def test_numerical_third_derivative(self):
        x = np.linspace(-20, 20, 300)  # Workspaces are about 300 points of range
        x = np.vstack([x, 2 * x])
        y = scipy.special.voigt_profile(x, 5, 5)
        numerical_derivative = numerical_third_derivative(x, y)
        expected_derivative = np.array([np.gradient(np.gradient(np.gradient(y_i, x_i), x_i), x_i)[6:-6] for y_i, x_i in zip(y, x)])
        np.testing.assert_allclose(numerical_derivative, expected_derivative, atol=1e-6)

    def test_make_gamma_correction_input_string(self):
        masses = [1, 12]
        mean_widths = [5, 10]
        mean_intensity_ratios = [0.6, 0.4]

        profiles_string = make_gamma_correction_input_string(masses, mean_widths, mean_intensity_ratios)

        self.assertEqual(
            profiles_string,
            "name=GaussianComptonProfile,Mass=1,Width=5,Intensity=0.6;name=GaussianComptonProfile,Mass=12,Width=10,Intensity=0.4;",
        )

    def test_make_multiple_scattering_input_string(self):

        masses = [1, 12]
        mean_intensity_ratios = [0.6, 0.4]
        mean_widths = [5, 10]

        profiles_list = make_multiple_scattering_input_string(masses, mean_widths, mean_intensity_ratios)

        self.assertEqual(profiles_list, [1.0, 0.6, 5.0, 12.0, 0.4, 10.0])

    def test_print_table_workspace(self):

        mock_table = Mock()
        mock_table.toDict.return_value = {"col1": ["a", "b", "c"], "col2": [1, 2, 3], "col3": [1.0, 2.0, 3.0]}
        mock_table.rowCount.return_value = 3
        mock_table.name.side_effect = lambda: "Mock Table Name"

        with patch("mvesuvio.util.general_helpers.logger") as mock_logger:
            print_table_workspace(mock_table)

            mock_logger.notice.assert_has_calls(
                [
                    call("Table Mock Table Name:"),
                    call(" -------------- "),
                    call("|col1|col2|col3|"),
                    call("|a   |1   |1   |"),
                    call("|b   |2   |2   |"),
                    call("|c   |3   |3   |"),
                    call(" -------------- "),
                ]
            )

    def test_load_resolution(self):

        instrument_parameters = np.vstack([np.arange(130, 140), np.zeros(10), np.zeros(10)]).T
        res_pars = load_resolution(instrument_parameters)

        expected_res_pars = np.array(
            [
                [8.87e01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 4.03e01],
                [8.87e01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 4.03e01],
                [8.87e01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 4.03e01],
                [8.87e01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 4.03e01],
                [8.87e01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 4.03e01],
                [7.30e01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 2.40e01],
                [7.30e01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 2.40e01],
                [7.30e01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 2.40e01],
                [7.30e01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 2.40e01],
                [7.30e01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 2.40e01],
            ]
        )

        np.testing.assert_allclose(res_pars, expected_res_pars)

    def test_load_instrument_params(self):

        ip_file_path = Path(__file__).parent.parent.parent / "data/analysis/unit/ip_example.par"
        ip = load_instrument_params(ip_file_path, np.array([5, 6, 7, 8]))

        print(str(ip).replace("\n", ",\n"))
        expected_ip = np.array(
            [
                [5.0, 5.0, 133.892, -0.2, 11.005, 0.587558],
                [6.0, 6.0, 133.753, -0.2, 11.005, 0.59536],
                [7.0, 7.0, 133.246, -0.2, 11.005, 0.59228],
                [8.0, 8.0, 131.671, -0.2, 11.005, 0.619911],
            ]
        )
        np.testing.assert_allclose(ip, expected_ip)


if __name__ == "__main__":
    unittest.main()
