
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
from mantid.simpleapi import AnalysisDataService, CreateWorkspace, GroupWorkspaces, Integration, Rebin, RenameWorkspace

from mvesuvio.default_config.experiment_template.run_reduction import ForwardAnalysisInputs
from mvesuvio.globals import Masking, Mode, ModeRunning, ScatteringType, Tags
from mvesuvio.util import reduction_helpers
from mvesuvio.util.files_manager import FilesManager


class TestReductionHelpers(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(reduction_helpers, "y_space_reduction", return_value=(MagicMock(), MagicMock()))
        self.mock_y_space_reduction = patcher.start()
        self.addCleanup(patcher.stop)

    def test_save_plots_from_outputs_skips_minimal_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(reduction_helpers.plt, "subplots") as mock_subplots:
                reduction_helpers.save_plots_from_outputs("input_0_ncp_group", Path(temp_dir), minimal_output=True)

        mock_subplots.assert_not_called()

    def test_save_plots_from_outputs_saves_sum_plot_and_ascii(self):
        data_sum = Mock()
        profile_sum = Mock()
        total_profile_sum = Mock()
        ncp_group = Mock()
        ncp_group.getNames.return_value = ["input_0_H_ncp", "input_0_H_ncp_sum", "input_0_total_ncp"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_directory = Path(temp_dir)
            with (
                patch.object(
                    reduction_helpers,
                    "mtd",
                    {
                        "input_0_ncp_group": ncp_group,
                        "input_0_sum": data_sum,
                        "input_0_H_ncp_sum": profile_sum,
                        "input_0_total_ncp_sum": total_profile_sum,
                    },
                ),
                patch.object(reduction_helpers.plt, "subplots", return_value=(Mock(), Mock())) as subplots,
                patch.object(reduction_helpers, "CloneWorkspace") as clone_workspace,
                patch.object(reduction_helpers, "AppendSpectra") as append_spectra,
                patch.object(reduction_helpers.plt, "savefig") as savefig,
                patch.object(reduction_helpers.plt, "close") as close,
                patch.object(reduction_helpers, "SaveAscii") as save_ascii,
            ):
                reduction_helpers.save_plots_from_outputs("input_0_ncp_group", output_directory)

        subplots.assert_called_once_with(subplot_kw={"projection": "mantid"})
        clone_workspace.assert_called_once_with(data_sum, OutputWorkspace="input_0_ncp_fits_sum")
        append_spectra.assert_has_calls(
            [
                call("input_0_ncp_fits_sum", profile_sum, OutputWorkspace="input_0_ncp_fits_sum"),
                call("input_0_ncp_fits_sum", total_profile_sum, OutputWorkspace="input_0_ncp_fits_sum"),
            ],
            any_order=False,
        )
        savefig.assert_called_once_with(output_directory / "figures/input_0_profiles_sum.pdf", bbox_inches="tight")
        close.assert_called_once()
        save_ascii.assert_called_once_with(
            "input_0_ncp_fits_sum",
            str(output_directory / "input_0_profiles_sum.txt"),
            LogList=["Sum of spectra", "Sum of H profile", "Sum of total profile"],
        )

    def test_save_results_from_outputs_saves_all_matching_outputs(self):
        workspaces = [
            "input_0",
            "input_0_ncp",
            "input_0_fse",
            "input_initial_parameters",
            "input_means",
            "input_fit_results",
            "unrelated",
        ]
        ads = Mock()
        ads.getObjectNames.return_value = workspaces
        output_directory = Path("/tmp/results")

        with (
            patch.object(reduction_helpers, "mtd", ads),
            patch.object(reduction_helpers, "SaveAscii") as save_ascii,
            patch.object(reduction_helpers, "SaveNexus") as save_nexus,
        ):
            reduction_helpers.save_results_from_outputs(
                "input", "input_0_ncp_group", output_directory
            )

        self.assertEqual(
            save_ascii.call_args_list,
            [call(name, str(output_directory / f"{name}.txt")) for name in workspaces[1:6]],
        )
        save_nexus.assert_called_once_with("input_0", str(output_directory / "input_0.nxs"))

    def test_save_results_from_outputs_writes_only_final_iteration_in_minimal_mode(self):
        workspaces = [
            "input_0_total_ncp",
            "input_1_total_ncp",
            "input_1_total_fse",
            "input_1_fit_results",
            "input_1_means",
            "input_0_means",
            "unrelated_1_total_ncp",
        ]
        ads = Mock()
        ads.getObjectNames.return_value = workspaces
        output_directory = Path("/tmp/results")

        with (
            patch.object(reduction_helpers, "mtd", ads),
            patch.object(reduction_helpers, "SaveAscii") as save_ascii,
            patch.object(reduction_helpers, "SaveNexus") as save_nexus,
        ):
            reduction_helpers.save_results_from_outputs(
                "input", "input_1_ncp_group", output_directory, minimal_output=True
            )

        self.assertEqual(
            save_ascii.call_args_list,
            [
                call(name, str(output_directory / f"{name}.txt"))
                for name in workspaces[1:5]
            ],
        )
        save_nexus.assert_not_called()

    def test_save_fitting_input_workspaces_uses_helper_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            FilesManager.set_experiment_dir(temp_dir)
            AnalysisDataService.clear()

            ws = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], DataE=[1, 1, 1], NSpec=1, UnitX="TOF")
            RenameWorkspace(ws, "front_0")

            ncp_a = CreateWorkspace(DataX=[0, 1, 2], DataY=[4, 5, 6], DataE=[1, 1, 1], NSpec=1, UnitX="TOF")
            RenameWorkspace(ncp_a, "front_0_1.0079_ncp")

            ncp_b = CreateWorkspace(DataX=[0, 1, 2], DataY=[7, 8, 9], DataE=[1, 1, 1], NSpec=1, UnitX="TOF")
            RenameWorkspace(ncp_b, "front_0_12_ncp")

            GroupWorkspaces(["front_0_1.0079_ncp", "front_0_12_ncp"], OutputWorkspace="front_0_ncp_group")

            with patch.object(reduction_helpers, "SaveNexus") as mock_save_nexus:
                with patch.object(reduction_helpers.fitting_helpers, "calculate_resolution", return_value=MagicMock()):
                    with patch.object(reduction_helpers.fitting_helpers, "isolate_lighest_mass_data", return_value=(MagicMock(), MagicMock())):
                        ForwardAnalysisInputs.name = "front"
                        ForwardAnalysisInputs.name_of_subtracted_workspace = "front"
                        ForwardAnalysisInputs.overwrite_analysis_input_workspace = ""
                        ForwardAnalysisInputs.run_this_scattering_type = True
                        ForwardAnalysisInputs.number_of_iterations_for_corrections = 0
                        ForwardAnalysisInputs.subtract_calculated_fse_from_data = False

                        reduction_helpers.save_fitting_input_workspaces(ForwardAnalysisInputs)

            self.assertEqual(mock_save_nexus.call_count, 3)
            self.assertTrue(FilesManager.get_fitting_inputs_dir().exists())

            AnalysisDataService.clear()
            FilesManager._experiment_dir = None

    def test_save_fitting_input_workspaces_uses_override_workspace_when_present_in_ads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            FilesManager.set_experiment_dir(temp_dir)
            AnalysisDataService.clear()

            ws = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], DataE=[1, 1, 1], NSpec=1, UnitX="TOF")
            RenameWorkspace(ws, "override_front")

            ws = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], DataE=[1, 1, 1], NSpec=1, UnitX="TOF")
            RenameWorkspace(ws, "override_front_0")

            ncp = CreateWorkspace(DataX=[0, 1, 2], DataY=[4, 5, 6], DataE=[1, 1, 1], NSpec=1, UnitX="TOF")
            RenameWorkspace(ncp, "override_front_0_12_ncp")
            GroupWorkspaces(["override_front_0_12_ncp"], OutputWorkspace="override_front_0_ncp_group")

            with patch.object(reduction_helpers, "SaveNexus") as mock_save_nexus:
                with patch.object(reduction_helpers.fitting_helpers, "calculate_resolution", return_value=MagicMock()):
                    with patch.object(
                        reduction_helpers.fitting_helpers,
                        "isolate_lighest_mass_data",
                        return_value=(MagicMock(), MagicMock()),
                    ):
                        ForwardAnalysisInputs.run_this_scattering_type = True
                        ForwardAnalysisInputs.number_of_iterations_for_corrections = 0
                        ForwardAnalysisInputs.subtract_calculated_fse_from_data = False
                        ForwardAnalysisInputs.overwrite_analysis_input_workspace = "override_front"
                        ForwardAnalysisInputs.name_of_subtracted_workspace = "front_subtracted"

                        reduction_helpers.save_fitting_input_workspaces(ForwardAnalysisInputs)

            self.assertEqual(mock_save_nexus.call_count, 3)
            saved_paths = [call_args.args[1] for call_args in mock_save_nexus.call_args_list]
            self.assertTrue(all("override_front_0" in path for path in saved_paths))

            AnalysisDataService.clear()
            FilesManager._experiment_dir = None

    def test_save_fitting_input_workspaces_falls_back_when_override_not_in_ads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            FilesManager.set_experiment_dir(temp_dir)
            AnalysisDataService.clear()

            ws = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], DataE=[1, 1, 1], NSpec=1, UnitX="TOF")
            RenameWorkspace(ws, "front_subtracted_0")

            ncp = CreateWorkspace(DataX=[0, 1, 2], DataY=[4, 5, 6], DataE=[1, 1, 1], NSpec=1, UnitX="TOF")
            RenameWorkspace(ncp, "front_subtracted_0_12_ncp")
            GroupWorkspaces(["front_subtracted_0_12_ncp"], OutputWorkspace="front_subtracted_0_ncp_group")

            with patch.object(reduction_helpers, "SaveNexus") as mock_save_nexus:
                with patch.object(reduction_helpers.fitting_helpers, "calculate_resolution", return_value=MagicMock()):
                    with patch.object(
                        reduction_helpers.fitting_helpers,
                        "isolate_lighest_mass_data",
                        return_value=(MagicMock(), MagicMock()),
                    ):
                        ForwardAnalysisInputs.run_this_scattering_type = True
                        ForwardAnalysisInputs.number_of_iterations_for_corrections = 0
                        ForwardAnalysisInputs.subtract_calculated_fse_from_data = False
                        ForwardAnalysisInputs.overwrite_analysis_input_workspace = "override_front"
                        ForwardAnalysisInputs.name_of_subtracted_workspace = "front_subtracted"

                        reduction_helpers.save_fitting_input_workspaces(ForwardAnalysisInputs)

            self.assertEqual(mock_save_nexus.call_count, 3)
            saved_paths = [call_args.args[1] for call_args in mock_save_nexus.call_args_list]
            self.assertTrue(all("front_subtracted_0" in path for path in saved_paths))

            AnalysisDataService.clear()
            FilesManager._experiment_dir = None

    def test_save_fitting_input_workspaces_ignores_missing_inputs_class(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            FilesManager.set_experiment_dir(temp_dir)
            AnalysisDataService.clear()

            with patch.object(reduction_helpers, "SaveNexus") as mock_save_nexus:
                reduction_helpers.save_fitting_input_workspaces(None)

            mock_save_nexus.assert_not_called()
            self.assertTrue(FilesManager.get_fitting_inputs_dir().exists())

            AnalysisDataService.clear()
            FilesManager._experiment_dir = None

    def test_convert_dict_to_table(self):
        d = {'H': {'label': 'H', 'mass': 1, 'intensity': 1}}
        table = reduction_helpers._convert_dict_to_table(d)
        self.assertEqual(['label', 'mass', 'intensity'], table.getColumnNames())
        self.assertEqual({'label': 'H', 'mass': 1, 'intensity': 1}, table.row(0))

    def test_mask_time_of_flight_bins_with_zeros(self):
        data_x = np.arange(10).reshape(1, -1) * np.ones((3, 1))
        data_y = np.ones((3, 10))
        data_e = np.ones((3, 10))
        workspace_mock = MagicMock()
        actual_data_x = data_x.copy()
        actual_data_y = data_y.copy()
        actual_data_e = data_e.copy()

        workspace_mock.dataX.side_effect = lambda i: actual_data_x[i]
        workspace_mock.mutableY.side_effect = lambda i: actual_data_y[i]
        workspace_mock.getNumberHistograms.return_value = 3
        inputs_class = Mock(mask_time_of_flight_range='4.5-7.3')
        with patch.object(reduction_helpers, "mtd", {"workspace": workspace_mock}):
            reduction_helpers.mask_time_of_flight_bins_with_zeros("workspace", inputs_class)

        np.testing.assert_allclose(actual_data_x, data_x)
        np.testing.assert_allclose(actual_data_e, data_e)
        expected_data_y = np.ones((3, 10))
        expected_data_y[(data_x >= 4.5) & (data_x <= 7.3)] = 0
        np.testing.assert_allclose(actual_data_y, expected_data_y)


    def test_mask_time_of_flight_bins_with_zeros_several_ranges(self):
        data_x = np.arange(0, 10, 0.5).reshape(1, -1) * np.ones((3, 1))
        data_y = np.ones_like(data_x)
        data_e = np.ones_like(data_x)
        workspace_mock = MagicMock()
        actual_data_x = data_x.copy()
        actual_data_y = data_y.copy()
        actual_data_e = data_e.copy()

        workspace_mock.dataX.side_effect = lambda i: actual_data_x[i]
        workspace_mock.mutableY.side_effect = lambda i: actual_data_y[i]
        workspace_mock.getNumberHistograms.return_value = 3

        inputs_class = Mock(mask_time_of_flight_range='1.1-3,3.5,  4.5-7.3, 9')
        with patch.object(reduction_helpers, "mtd", {"workspace": workspace_mock}):
            reduction_helpers.mask_time_of_flight_bins_with_zeros("workspace", inputs_class)

        np.testing.assert_allclose(actual_data_x, data_x)
        np.testing.assert_allclose(actual_data_e, data_e)
        expected_data_y = np.ones_like(data_x)
        expected_data_y[(data_x >= 1.1) & (data_x <= 3)] = 0
        expected_data_y[(data_x == 3.5)] = 0
        expected_data_y[(data_x >= 4.5) & (data_x <= 7.3)] = 0
        expected_data_y[(data_x == 9)] = 0
        np.testing.assert_allclose(actual_data_y, expected_data_y)

    def test_fix_profile_parameters_with_H(self):
        means_table_mock = MagicMock()
        means_table_mock.rowCount.return_value = 3
        means_table_mock.row.side_effect = [
            {"label": "16.0", "mass": 16.0, "mean_width": 8.974, "std_width": 1.401, "mean_intensity": 0.176, "std_intensity": 0.08722},
            {"label": "27.0", "mass": 27.0, "mean_width": 15.397, "std_width": 1.131, "mean_intensity": 0.305, "std_intensity": 0.04895},
            {"label": "12.0", "mass": 12.0, "mean_width": 13.932, "std_width": 0.314, "mean_intensity": 0.517, "std_intensity": 0.09531},
        ]
        profiles_table_mock = MagicMock()
        profiles_table_mock.rowCount.return_value = 4
        profiles_table_mock.row.side_effect = [
            {"label": "1.0079", "mass": 1.0078, "intensity": 1.0, "intensity_lb": 0, "intensity_ub": np.inf, "width": 4.699, "width_lb": 3, "width_ub": 6, "center": 0.0, "center_lb": -3, "center_ub": 1},
            {"label": "16.0", "mass": 16.0, "intensity": 1.0, "intensity_lb": 0, "intensity_ub": np.inf, "width": 12, "width_lb": 0, "width_ub": np.inf, "center": 0.0, "center_lb": -3, "center_ub": 1},
            {"label": "12.0", "mass": 12.0, "intensity": 1.0, "intensity_lb": 0, "intensity_ub": np.inf, "width": 8, "width_lb": 0, "width_ub": np.inf, "center": 0.0, "center_lb": -3, "center_ub": 1},
            {"label": "27.0", "mass": 27.0, "intensity": 1.0, "intensity_lb": 0, "intensity_ub": np.inf, "width": 13, "width_lb": 0, "width_ub": np.inf, "center": 0.0, "center_lb": -3, "center_ub": 1},
        ]

        result_table = reduction_helpers.fix_profile_parameters(means_table_mock, profiles_table_mock, h_ratio=14.7)
        self.assertEqual(
            result_table.row(0),
            {"label": "1.0079", "mass": 1.0077999830245972, "intensity": 0.8839251399040222, "intensity_lb": 0.0, "intensity_ub": np.inf, "width": 4.698999881744385, "width_lb": 3.0, "width_ub": 6.0, "center": 0.0, "center_lb": -3.0, "center_ub": 1.0},
        )

    def test_fix_profile_parameters_without_H(self):
        means_table_mock = MagicMock()
        means_table_mock.rowCount.return_value = 3
        means_table_mock.row.side_effect = [
            {"label": "16.0", "mass": 16.0, "mean_width": 8.974, "std_width": 1.401, "mean_intensity": 0.176, "std_intensity": 0.08722},
            {"label": "27.0", "mass": 27.0, "mean_width": 15.397, "std_width": 1.131, "mean_intensity": 0.305, "std_intensity": 0.04895},
            {"label": "12.0", "mass": 12.0, "mean_width": 13.932, "std_width": 0.314, "mean_intensity": 0.517, "std_intensity": 0.09531},
        ]
        profiles_table_mock = MagicMock()
        profiles_table_mock.rowCount.return_value = 3
        profiles_table_mock.row.side_effect = [
            {"label": "16.0", "mass": 16.0, "intensity": 1.0, "intensity_lb": 0, "intensity_ub": np.inf, "width": 12, "width_lb": 0, "width_ub": np.inf, "center": 0.0, "center_lb": -3, "center_ub": 1},
            {"label": "12.0", "mass": 12.0, "intensity": 1.0, "intensity_lb": 0, "intensity_ub": np.inf, "width": 8, "width_lb": 0, "width_ub": np.inf, "center": 0.0, "center_lb": -3, "center_ub": 1},
            {"label": "27.0", "mass": 27.0, "intensity": 1.0, "intensity_lb": 0, "intensity_ub": np.inf, "width": 13, "width_lb": 0, "width_ub": np.inf, "center": 0.0, "center_lb": -3, "center_ub": 1},
        ]

        result_table = reduction_helpers.fix_profile_parameters(means_table_mock, profiles_table_mock, h_ratio=14.7)
        self.assertEqual(result_table.row(0)["label"], "16.0")
        self.assertEqual(result_table.row(0)["intensity"], 0.17635270953178406)

    def test_calculate_h_ratio(self):
        means_table_mock = MagicMock()
        means_table_mock.column.side_effect = lambda x: [16, 1, 12] if x == "mass" else [0.1, 0.85, 0.05]
        h_ratio = reduction_helpers.calculate_h_ratio(means_table_mock, 12)
        self.assertEqual(h_ratio, 0.85 / 0.05)

    def test_is_hydrogen_present_bad_inputs(self):
        invalid_cases = [
            np.array([1.01]),
            np.array([2.0, 1.0, 12.0]),
            np.array([1.0, 1.0078, 12.0]),
        ]

        for masses in invalid_cases:
            with self.subTest(masses=masses):
                with self.assertRaises(AssertionError):
                    reduction_helpers.is_hydrogen_present(masses)

    def test_create_profiles_table(self):

        mock_ai = Mock()
        mock_ai.masses = [1, 12, 16]
        mock_ai.initial_fitting_parameters = [1, 5, 0, 1, 10, 0, 1, 13, 0]
        mock_ai.fitting_bounds = [[0, None], [2, 6], [-1, 3], [0, None], [8, 12], [-1, 3], [0, np.inf], [11, 15], [-1, 3]]

        with patch('mvesuvio.util.reduction_helpers.CreateEmptyTableWorkspace') as mock_create_table_ws:
            table_mock = MagicMock()
            mock_create_table_ws.return_value = table_mock

            reduction_helpers.create_profiles_table(table_mock, mock_ai)

            table_mock.addColumn.assert_has_calls([
                call(type='str', name='label'),
                call(type='float', name='mass'),
                call(type='float', name='intensity'),
                call(type='float', name='intensity_lb'),
                call(type='float', name='intensity_ub'),
                call(type='float', name='width'),
                call(type='float', name='width_lb'),
                call(type='float', name='width_ub'),
                call(type='float', name='center'),
                call(type='float', name='center_lb'),
                call(type='float', name='center_ub')
            ])
            table_mock.addRow.assert_has_calls([
                call(['1.0', 1.0, 1.0, 0.0, np.inf, 5.0, 2.0, 6.0, 0.0, -1.0, 3.0]),
                call(['12.0', 12.0, 1.0, 0.0, np.inf, 10.0, 8.0, 12.0, 0.0, -1.0, 3.0]),
                call(['16.0', 16.0, 1.0, 0.0, np.inf, 13.0, 11.0, 15.0, 0.0, -1.0, 3.0])
            ])

    def test_is_hydrogen_present(self):
        valid_cases = [
            (np.array([1.0078, 12.0, 16.0]), True),
            (np.array([2.0, 12.0, 16.0]), False),
            (np.array([2.0]), False),
        ]

        for masses, expected in valid_cases:
            with self.subTest(masses=masses):
                self.assertEqual(reduction_helpers.is_hydrogen_present(masses), expected)

    def test_ws_history_matches_inputs_invalid_path(self):
        path = Path("notthere.nxs")
        with patch('mvesuvio.util.reduction_helpers.logger') as mock_logger:
            match = reduction_helpers.ws_history_matches_inputs(0, 0, 0, path)
            mock_logger.notice.assert_has_calls([call('Cached workspace not found at notthere.nxs')])
            self.assertFalse(match)


    @patch('mvesuvio.util.reduction_helpers.DeleteWorkspace')
    @patch('mvesuvio.util.reduction_helpers.Load')
    def test_ws_history_matches_inputs(self, mock_load, mock_delete):
        path = Mock()
        path.is_file.return_value = True
        props = {
            "Filename": "1234-1235",
            "Mode": Mode.SINGLE_DIFFERENCE,
            "InstrumentParFile": "ip_par.txt"
        }
        mock_metadata = Mock()
        mock_metadata.getPropertyValue.side_effect = lambda key: props[key]
        mock_history = Mock()
        mock_history.getAlgorithmHistory.return_value = mock_metadata
        mock_ws = Mock()
        mock_ws.getHistory.return_value = mock_history
        mock_load.return_value = mock_ws

        cases = [
            ("0000", Mode.SINGLE_DIFFERENCE, "ip_par.txt", False, 'Filename in saved workspace did not match: 1234-1235 and 0000'),
            ("1234-1235", Mode.DOUBLE_DIFFERENCE, "ip_par.txt", False, 'Mode in saved workspace did not match: SingleDifference and DoubleDifference'),
            ("1234-1235", Mode.SINGLE_DIFFERENCE, "new_par.txt", False, 'IP files in saved workspace did not match: ip_par.txt and new_par.txt'),
            ("1234-1235", Mode.SINGLE_DIFFERENCE, "ip_par.txt", True, '\nLocally saved workspace metadata matched with analysis inputs.\n'),
        ]

        for runs, mode, ip_file, expected, expected_message in cases:
            with self.subTest(runs=runs, mode=mode, ip_file=ip_file):
                mock_delete.reset_mock()
                with patch('mvesuvio.util.reduction_helpers.logger') as mock_logger:
                    match = reduction_helpers.ws_history_matches_inputs(runs, mode, ip_file, path)
                    mock_logger.notice.assert_has_calls([call(expected_message)])
                    self.assertEqual(match, expected)
                    if expected:
                        mock_delete.assert_called_once_with(mock_ws)
                    else:
                        mock_delete.assert_not_called()


    @patch('mvesuvio.util.reduction_helpers.SaveNexus')
    @patch('mvesuvio.util.reduction_helpers.LoadVesuvio')
    def test_save_ws_from_load_vesuvio_backward(self, mock_load_vesuvio, mock_save_nexus):
        path = Path(f'notthere/raw_{Tags.BACKWARD}.nxs')
        reduction_helpers.save_ws_from_load_vesuvio("1234", Mode.SINGLE_DIFFERENCE, "ipfile.txt", path)
        mock_load_vesuvio.assert_has_calls([
            call(Filename='1234', SpectrumList='3-134', Mode=Mode.SINGLE_DIFFERENCE, InstrumentParFile='ipfile.txt', OutputWorkspace=f'raw_{Tags.BACKWARD}.nxs', LoadLogFiles=False)
        ])
        _, kwargs = mock_save_nexus.call_args
        self.assertEqual(kwargs["Filename"], str(path.absolute()))


    @patch('mvesuvio.util.reduction_helpers.SaveNexus')
    @patch('mvesuvio.util.reduction_helpers.LoadVesuvio')
    def test_save_ws_from_load_vesuvio_forward(self, mock_load_vesuvio, mock_save_nexus):
        path = Path(f'notthere/raw_{Tags.FORWARD}.nxs')
        reduction_helpers.save_ws_from_load_vesuvio("1234", Mode.SINGLE_DIFFERENCE, "ipfile.txt", path)
        mock_load_vesuvio.assert_has_calls([
            call(Filename='1234', SpectrumList="135-198", Mode=Mode.SINGLE_DIFFERENCE, InstrumentParFile='ipfile.txt', OutputWorkspace=f'raw_{Tags.FORWARD}.nxs', LoadLogFiles=False)
        ])
        args, kwargs = mock_save_nexus.call_args
        self.assertEqual(kwargs["Filename"], str(path.absolute()))


    def test_load_and_save_input_ws_if_not_on_path_backward(self):
        class BackwardInputs:
            name = "backward"
            mode = Mode.SINGLE_DIFFERENCE
            runs = "1234"
            empty_runs = "5678"
            instrument_parameters_file = "ipfile.txt"

        reduction_inputs_dir = Path("/tmp/reduction_inputs")
        raw_filename = "experiment_raw_backward.nxs"
        empty_filename = "experiment_empty_backward.nxs"
        raw_path = reduction_inputs_dir / raw_filename
        empty_path = reduction_inputs_dir / empty_filename

        with patch.object(reduction_helpers.FilesManager, "get_reduction_inputs_dir", return_value=reduction_inputs_dir), \
            patch.object(reduction_helpers.FilesManager, "get_backward_raw_filename", return_value=raw_filename), \
            patch.object(reduction_helpers.FilesManager, "get_backward_empty_filename", return_value=empty_filename), \
            patch.object(reduction_helpers.FilesManager, "get_instrument_parameters_dir", return_value=Path("/tmp/ip")), \
            patch.object(reduction_helpers, "ws_history_matches_inputs", side_effect=[False, False]) as mock_matches, \
            patch.object(reduction_helpers, "save_ws_from_load_vesuvio") as mock_save_ws:
            result_raw_path, result_empty_path = reduction_helpers.store_input_ws_if_not_on_path(BackwardInputs)

        self.assertEqual(result_raw_path, raw_path)
        self.assertEqual(result_empty_path, empty_path)
        mock_matches.assert_has_calls([
            call("1234", Mode.SINGLE_DIFFERENCE, "ipfile.txt", raw_path),
            call("5678", Mode.SINGLE_DIFFERENCE, "ipfile.txt", empty_path),
        ])
        ipfile_path = str(Path("/tmp/ip/ipfile.txt"))
        mock_save_ws.assert_has_calls([
            call("1234", Mode.SINGLE_DIFFERENCE, ipfile_path, raw_path),
            call("5678", Mode.SINGLE_DIFFERENCE, ipfile_path, empty_path),
        ])

    def test_store_input_ws_if_not_on_path_rejects_unknown_scattering_type(self):
        class InvalidInputs:
            pass

        with (
            patch.object(reduction_helpers.general_helpers, "_get_scattering_type", return_value=None),
            self.assertRaisesRegex(ValueError, "Input class for workspace not valid: InvalidInputs"),
        ):
            reduction_helpers.store_input_ws_if_not_on_path(InvalidInputs)


    def test_load_and_save_input_ws_if_not_on_path_forward_uses_cached_files(self):
        class ForwardInputs:
            name = "forward"
            mode = Mode.SINGLE_DIFFERENCE
            runs = "2345"
            empty_runs = "6789"
            instrument_parameters_file = "ipfile.txt"

        reduction_inputs_dir = Path("/tmp/reduction_inputs")
        raw_filename = "experiment_raw_forward.nxs"
        empty_filename = "experiment_empty_forward.nxs"
        raw_path = reduction_inputs_dir / raw_filename
        empty_path = reduction_inputs_dir / empty_filename

        with patch.object(reduction_helpers.FilesManager, "get_reduction_inputs_dir", return_value=reduction_inputs_dir), \
            patch.object(reduction_helpers.FilesManager, "get_forward_raw_filename", return_value=raw_filename), \
            patch.object(reduction_helpers.FilesManager, "get_forward_empty_filename", return_value=empty_filename), \
            patch.object(reduction_helpers.FilesManager, "get_instrument_parameters_dir", return_value=Path("/tmp/ip")), \
            patch.object(reduction_helpers, "ws_history_matches_inputs", return_value=True) as mock_matches, \
            patch.object(reduction_helpers, "save_ws_from_load_vesuvio") as mock_save_ws:
            result_raw_path, result_empty_path = reduction_helpers.store_input_ws_if_not_on_path(ForwardInputs)

        self.assertEqual(result_raw_path, raw_path)
        self.assertEqual(result_empty_path, empty_path)
        mock_matches.assert_has_calls([
            call("2345", Mode.SINGLE_DIFFERENCE, "ipfile.txt", raw_path),
            call("6789", Mode.SINGLE_DIFFERENCE, "ipfile.txt", empty_path),
        ])
        mock_save_ws.assert_not_called()

    @patch('mvesuvio.util.reduction_helpers.SumSpectra')
    @patch('mvesuvio.util.reduction_helpers.Load')
    @patch('mvesuvio.util.reduction_helpers.store_input_ws_if_not_on_path')
    def test_load_input_ws_loads_raw_and_empty_workspaces(self, mock_store, mock_load, mock_sum_spectra):
        class DummyInputs:
            name = ScatteringType.BACKWARD.value
            mode = Mode.SINGLE_DIFFERENCE.value
            runs = "1234"
            empty_runs = "5678"
            instrument_parameters_file = "ipfile.txt"

        raw_path = Path("/tmp/reduction_inputs/experiment_raw_backward.nxs")
        empty_path = Path("/tmp/reduction_inputs/experiment_empty_backward.nxs")
        mock_store.return_value = (raw_path, empty_path)

        result = reduction_helpers.load_input_ws(DummyInputs)

        self.assertEqual(result, ("experiment_raw_backward", "experiment_empty_backward"))
        mock_store.assert_called_once_with(DummyInputs)
        mock_load.assert_has_calls([
            call(Filename=str(raw_path), OutputWorkspace="experiment_raw_backward"),
            call(Filename=str(empty_path), OutputWorkspace="experiment_empty_backward"),
        ])
        mock_sum_spectra.assert_has_calls([
            call(InputWorkspace="experiment_raw_backward", OutputWorkspace="experiment_raw_backward_sum"),
            call(InputWorkspace="experiment_empty_backward", OutputWorkspace="experiment_empty_backward_sum"),
        ])

    def test_convert_to_list_of_spectrum_numbers(self):
        cases = [
            ("1, 3-6, 8-8, 10", [1, 3, 4, 5, 6, 8, 10]),
            (["1", 3, 5, 6, "7"], [1, 3, 5, 6, 7]),
            ([1, 3, 5, 6, 7], [1, 3, 5, 6, 7]),
        ]

        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(reduction_helpers.convert_to_list_of_spectrum_numbers(value), expected)

    def test_convert_to_list_of_spectrum_numbers_rejects_unsupported_type(self):
        with self.assertRaisesRegex(ValueError, "Masked detectors should be string, list or array"):
            reduction_helpers.convert_to_list_of_spectrum_numbers(3)

    def test_calculate_h_ratio_returns_none_without_hydrogen(self):
        means_table = Mock()
        means_table.column.side_effect = lambda name: [12, 16] if name == "mass" else [0.2, 0.8]

        self.assertIsNone(reduction_helpers.calculate_h_ratio(means_table, 12))

    def test_h_ratio_is_zero_only_for_active_backward_inputs_with_hydrogen(self):
        back_inputs = Mock(run_this_scattering_type=True, intensity_ratio_of_hydrogen_to_chosen_mass=0)
        front_inputs = Mock(masses=[1, 12])

        with patch.object(reduction_helpers, "is_hydrogen_present", return_value=True):
            self.assertTrue(reduction_helpers.h_ratio_is_zero_when_h_present(back_inputs, front_inputs))
            back_inputs.intensity_ratio_of_hydrogen_to_chosen_mass = 0.5
            self.assertFalse(reduction_helpers.h_ratio_is_zero_when_h_present(back_inputs, front_inputs))

        back_inputs.intensity_ratio_of_hydrogen_to_chosen_mass = 2
        with (
            patch.object(reduction_helpers, "is_hydrogen_present", return_value=False),
            patch.object(reduction_helpers.logger, "warning") as warning,
        ):
            self.assertFalse(reduction_helpers.h_ratio_is_zero_when_h_present(back_inputs, front_inputs))
            self.assertEqual(back_inputs.intensity_ratio_of_hydrogen_to_chosen_mass, 0)
            warning.assert_called_once()

        back_inputs.run_this_scattering_type = False
        self.assertFalse(reduction_helpers.h_ratio_is_zero_when_h_present(back_inputs, front_inputs))

    def test_run_estimate_h_ratio_stops_when_input_is_not_empty(self):
        back_alg = Mock()
        back_alg.getPropertyValue.return_value = ModeRunning.BACKWARD
        front_alg = Mock()
        front_alg.getPropertyValue.return_value = ModeRunning.FORWARD

        with patch("builtins.input", return_value="cancel"), patch.object(reduction_helpers.logger, "error") as error:
            result = reduction_helpers.run_estimate_h_ratio(back_alg, front_alg, [1, 12], 1, 0.01, 3)

        self.assertIsNone(result)
        error.assert_called_once()
        front_alg.execute.assert_not_called()

    def test_run_estimate_h_ratio_stops_after_converging(self):
        back_alg = Mock()
        back_alg.getPropertyValue.return_value = ModeRunning.BACKWARD
        front_alg = Mock()
        front_alg.getPropertyValue.side_effect = lambda name: {
            "ModeRunning": ModeRunning.FORWARD,
            "OutputMeansTable": "means",
        }[name]
        ratio_table = Mock()

        with (
            patch("builtins.input", return_value=""),
            patch.object(reduction_helpers, "CreateEmptyTableWorkspace", return_value=ratio_table),
            patch.object(reduction_helpers, "mtd", {"means": Mock()}),
            patch.object(reduction_helpers, "calculate_h_ratio", side_effect=[0.5, 0.5]),
            patch.object(reduction_helpers, "execute_joint_algorithms") as execute_joint,
            patch.object(reduction_helpers, "print_table_workspace") as print_table,
        ):
            result = reduction_helpers.run_estimate_h_ratio(back_alg, front_alg, [1, 12], 1, 0.01, 3)

        self.assertIs(result, ratio_table)
        ratio_table.addRow.assert_has_calls([call([0.5]), call([0.5])])
        back_alg.setProperty.assert_called_once_with("HRatioToChosenMass", 0.5)
        execute_joint.assert_called_once_with(back_alg, front_alg)
        print_table.assert_called_once_with(ratio_table)

    def test_execute_joint_algorithms_updates_profiles_and_executes_forward_algorithm(self):
        back_alg = Mock()
        back_alg.getPropertyValue.return_value = ModeRunning.BACKWARD
        back_alg.getProperty.return_value.value = 0.5
        front_alg = Mock()
        front_alg.getPropertyValue.return_value = ModeRunning.FORWARD
        front_alg.getPropertyValue.side_effect = lambda name: {
            "ModeRunning": ModeRunning.FORWARD,
            "InputProfiles": "profiles",
        }[name]
        incoming_means = Mock()
        receiving_profiles = Mock()
        receiving_profiles.name.return_value = "profiles"
        fixed_profiles = Mock()
        with (
            patch.object(reduction_helpers, "mtd", {"means": incoming_means, "profiles": receiving_profiles}),
            patch.object(reduction_helpers, "fix_profile_parameters", return_value=fixed_profiles) as fix_profiles,
            patch.object(reduction_helpers, "RenameWorkspace") as rename_workspace,
            patch.object(reduction_helpers, "print_table_workspace") as print_table,
        ):
            back_alg.getPropertyValue.side_effect = lambda name: {
                "ModeRunning": ModeRunning.BACKWARD,
                "OutputMeansTable": "means",
            }[name]
            reduction_helpers.execute_joint_algorithms(back_alg, front_alg)

        fix_profiles.assert_called_once_with(incoming_means, receiving_profiles, 0.5)
        rename_workspace.assert_called_once_with(fixed_profiles, "profiles")
        print_table.assert_called_once_with(receiving_profiles)
        front_alg.setPropertyValue.assert_called_once_with("InputProfiles", "profiles")
        back_alg.execute.assert_called_once()
        front_alg.execute.assert_called_once()

    def test_init_analysis_algorithm_skips_missing_workspace(self):
        with (
            patch.object(reduction_helpers.AnalysisDataService, "doesExist", return_value=False),
            patch.object(reduction_helpers.logger, "warning") as warning,
        ):
            result = reduction_helpers.init_analysis_algorithm("missing", Mock())

        self.assertIsNone(result)
        warning.assert_called_once()

    def test_init_analysis_algorithm_sets_forward_algorithm_properties(self):
        class ForwardTestInputs:
            mode = Mode.SINGLE_DIFFERENCE
            instrument_parameters_file = "ip_file.par"
            chosen_mass_index = 2
            intensity_ratio_of_hydrogen_to_chosen_mass = 0.5
            number_of_iterations_for_corrections = 4
            mask_detectors = "3-4"
            do_multiple_scattering_correction = True
            do_gamma_correction = False
            transmission_guess = 0.9
            multiple_scattering_order = 2
            multiple_scattering_number_of_events = 100
            constraints = ()

        profiles_table = Mock()
        profiles_table.name.return_value = "profiles"
        algorithm = Mock()
        with (
            patch.object(reduction_helpers, "AnalysisDataService", Mock(doesExist=Mock(return_value=True))),
            patch.object(reduction_helpers, "create_profiles_table", return_value=profiles_table),
            patch.object(reduction_helpers.FilesManager, "get_instrument_parameters_dir", return_value=Path("/tmp/ip")),
            patch.object(reduction_helpers.AlgorithmFactory, "subscribe"),
            patch.object(reduction_helpers, "AlgorithmManager", Mock(createUnmanaged=Mock(return_value=algorithm))),
        ):
            result = reduction_helpers.init_analysis_algorithm("input", ForwardTestInputs)

        self.assertIs(result, algorithm)
        self.assertEqual(ForwardTestInputs.chosen_mass_index, 0)
        self.assertEqual(ForwardTestInputs.intensity_ratio_of_hydrogen_to_chosen_mass, 0)
        algorithm.setProperties.assert_called_once()
        properties = algorithm.setProperties.call_args.args[0]
        ipfile_path = str(Path("/tmp/ip/ip_file.par"))
        self.assertEqual(properties["InputWorkspace"], "input")
        self.assertEqual(properties["InputProfiles"], "profiles")
        self.assertEqual(properties["InstrumentParametersFile"], ipfile_path)
        self.assertEqual(properties["InvalidDetectors"], [3, 4])
        self.assertEqual(properties["ModeRunning"], ModeRunning.FORWARD)

    def test_save_plots_from_outputs_rejects_non_group_name(self):
        with self.assertRaisesRegex(ValueError, "Expected an NCP output group"):
            reduction_helpers.save_plots_from_outputs("input", Path("/tmp"))

    def test_save_results_from_outputs_rejects_unrelated_group(self):
        with self.assertRaisesRegex(ValueError, "Expected an NCP output group"):
            reduction_helpers.save_results_from_outputs("input", "other_0_ncp_group", Path("/tmp"))

    def test_crop_and_mask_workspace_crops_masks_and_applies_tof_masking(self):
        workspace = Mock()
        workspace.getIndexFromSpectrumNumber.side_effect = {3: 0, 4: 1}.__getitem__
        inputs = Mock(detectors=" 3 - 4 ", mask_detectors=[4], mask_time_of_flight_range="")

        with (
            patch.object(reduction_helpers, "mtd", {"input": workspace}),
            patch.object(reduction_helpers, "CropWorkspace") as crop_workspace,
            patch.object(reduction_helpers, "MaskDetectors") as mask_detectors,
            patch.object(reduction_helpers, "mask_time_of_flight_bins_with_zeros") as mask_tof,
        ):
            reduction_helpers.crop_and_mask_workspace("input", inputs)

        crop_workspace.assert_called_once_with(
            InputWorkspace="input",
            StartWorkspaceIndex=0,
            EndWorkspaceIndex=1,
            OutputWorkspace="input",
        )
        mask_detectors.assert_called_once_with(Workspace="input", SpectraList=[4])
        mask_tof.assert_called_once_with("input", inputs)

    def test_crop_and_mask_workspace_rejects_invalid_detector_range(self):
        with self.assertRaisesRegex(ValueError, "Invalid detectors range"):
            reduction_helpers.crop_and_mask_workspace("input", Mock(detectors="3,4"))

    def test_save_ws_from_load_vesuvio_rejects_unrecognized_path(self):
        with self.assertRaisesRegex(ValueError, "Invalid name to save workspace"):
            reduction_helpers.save_ws_from_load_vesuvio("1234", Mode.SINGLE_DIFFERENCE, "ipfile.txt", Path("raw.nxs"))

    def test_save_fitting_input_workspaces_skips_inactive_inputs(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(reduction_helpers.FilesManager, "get_fitting_inputs_dir", return_value=Path(temp_dir)),
        ):
            reduction_helpers.save_fitting_input_workspaces(Mock(run_this_scattering_type=False))

    def test_save_fitting_input_workspaces_warns_when_outputs_are_missing(self):
        inputs = Mock(
            run_this_scattering_type=True,
            number_of_iterations_for_corrections=0,
            overwrite_analysis_input_workspace="",
            name_of_subtracted_workspace="input",
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(reduction_helpers.FilesManager, "get_fitting_inputs_dir", return_value=Path(temp_dir)),
            patch.object(reduction_helpers.AnalysisDataService, "doesExist", return_value=False),
            patch.object(reduction_helpers.logger, "warning") as warning,
        ):
            reduction_helpers.save_fitting_input_workspaces(inputs)

        warning.assert_called_once()

    def test_mask_time_of_flight_bins_with_zeros_rejects_invalid_range(self):
        inputs = Mock(mask_time_of_flight_range="1-2,invalid")
        with self.assertRaisesRegex(ValueError, "Invalid mask_time_of_flight_range"):
            reduction_helpers.mask_time_of_flight_bins_with_zeros("workspace", inputs)



class TestRebinMaskedWeighted(unittest.TestCase):
    def tearDown(self):
        AnalysisDataService.clear()

    def test_matches_rebin_when_no_points_are_masked(self):
        data_x = np.tile([0.5, 1.5, 2.5, 3.5], 2)
        data_y = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]
        data_e = np.full(len(data_y), 0.2)
        ws = CreateWorkspace(
            DataX=data_x, DataY=data_y, DataE=data_e, NSpec=2, Distribution=True, OutputWorkspace="unmasked_input"
        )

        expected = Rebin(InputWorkspace=ws, Params=[0, 2, 4], OutputWorkspace="ordinary_rebin")
        actual = reduction_helpers.rebin_masked_weighted(ws, [0, 2, 4], "_weighted_rebin")

        np.testing.assert_allclose(actual.extractY(), expected.extractY())
        np.testing.assert_allclose(actual.extractE(), expected.extractE())

    def test_masked_points_do_not_contribute_to_rebinned_values(self):
        data_x = np.tile([0.5, 1.5, 2.5, 3.5], 2)
        data_y = np.array([1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0])
        data_e = np.full(len(data_y), 0.2)
        masked = [1, 3, 6]
        data_y[masked] = 0
        data_e[masked] = 0
        ws = CreateWorkspace(
            DataX=data_x, DataY=data_y, DataE=data_e, NSpec=2, Distribution=True, OutputWorkspace="masked_input"
        )

        actual = reduction_helpers.rebin_masked_weighted(ws, [0, 2, 4], "_weighted_rebin")

        np.testing.assert_allclose(actual.extractY(), [[1.0, 5.0], [3.0, 8.0]])
        expected_e = [[0.2, 0.2], [0.2 / np.sqrt(2), 0.2]]
        np.testing.assert_allclose(actual.extractE(), expected_e)


class TestReplaceZerosWithNcp(unittest.TestCase):
    def tearDown(self):
        AnalysisDataService.clear()

    def test_replaces_only_fully_zero_columns_and_returns_summed_workspace(self):
        data_x = np.tile([0.5, 1.5, 2.5, 3.5], 2)
        data_y = np.array([[1.0, 0.0, 3.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
        data_e = np.full(data_y.shape, 0.25)
        ncp_y = np.array([[10.0, 20.0, 30.0, 40.0]])
        ws = CreateWorkspace(
            DataX=data_x, DataY=data_y, DataE=data_e, NSpec=2, OutputWorkspace="replace_input"
        )
        ncp = CreateWorkspace(
            DataX=[0.5, 1.5, 2.5, 3.5], DataY=ncp_y[0], DataE=np.ones(4), NSpec=1, OutputWorkspace="replace_ncp"
        )

        actual = reduction_helpers.replace_zeros_with_ncp(ws, ncp.extractY())

        expected_y = [[1.0, 20.0, 3.0, 40.0], [2.0, 20.0, 0.0, 40.0]]
        np.testing.assert_allclose(actual.extractY(), expected_y)
        np.testing.assert_allclose(actual.extractE(), data_e)
        np.testing.assert_allclose(actual.extractX(), ws.extractX())
        np.testing.assert_allclose(ws.extractY(), data_y)
        self.assertEqual(actual.name(), "replace_input_ncpmask")

        summed = AnalysisDataService.retrieve("replace_input_ncpmask_sum")
        np.testing.assert_allclose(summed.extractY(), [[3.0, 40.0, 3.0, 80.0]])


class TestYSpaceReduction(unittest.TestCase):
    def tearDown(self):
        AnalysisDataService.clear()

    @staticmethod
    def _inputs(mask_time_of_flight_range=None, mask_zeros_with=Masking.NAN, do_symmetrisation=False):
        return Mock(
            masses=[1.0],
            range_for_rebinning_in_y_space=[0, 2, 4],
            mask_time_of_flight_range=mask_time_of_flight_range,
            mask_zeros_with=mask_zeros_with,
            do_symmetrisation=do_symmetrisation,
        )

    @staticmethod
    def _run_with_mocks(inputs):
        ws_tof = Mock()
        ws_tof.name.return_value = "tof"
        ws_ncp = Mock()
        joy = Mock()
        joy.name.return_value = "tof_joy"
        rebinned = Mock()
        rebinned.name.return_value = "tof_joy_rebin"
        normalised = Mock()
        normalised.name.return_value = "tof_joy_rebin_norm"
        averaged = Mock()
        replaced = Mock()
        replaced.name.return_value = "tof_replaced"

        with (
            patch.object(reduction_helpers, "ConvertToYSpace", return_value=joy) as convert,
            patch.object(reduction_helpers, "Rebin", return_value=rebinned) as rebin,
            patch.object(reduction_helpers, "rebin_masked_weighted", return_value=rebinned) as masked_rebin,
            patch.object(reduction_helpers, "Integration") as integrate,
            patch.object(reduction_helpers, "Divide", return_value=normalised) as divide,
            patch.object(reduction_helpers, "SumSpectra", return_value=averaged) as sum_spectra,
            patch.object(reduction_helpers, "replace_zeros_with_ncp", return_value=replaced) as replace_zeros,
            patch.object(reduction_helpers, "symmetrize_weighted_mean") as symmetrize,
            patch.object(reduction_helpers, "DeleteWorkspace") as delete,
        ):
            result = reduction_helpers.y_space_reduction(ws_tof, ws_ncp, inputs)

        return (
            result,
            ws_tof,
            ws_ncp,
            joy,
            rebinned,
            normalised,
            averaged,
            convert,
            rebin,
            masked_rebin,
            integrate,
            divide,
            sum_spectra,
            replace_zeros,
            symmetrize,
            delete,
        )

    def test_unmasked_path_uses_mantid_rebin_without_replacement_or_symmetrisation(self):
        (
            result,
            ws_tof,
            ws_ncp,
            joy,
            rebinned,
            normalised,
            averaged,
            convert,
            rebin,
            masked_rebin,
            integrate,
            divide,
            sum_spectra,
            replace_zeros,
            symmetrize,
            delete,
        ) = self._run_with_mocks(self._inputs())

        ws_ncp.extractY.assert_called_once_with()
        convert.assert_called_once_with(ws_tof, Mass=1.0, OutputWorkspace="tof_joy")
        masked_rebin.assert_not_called()
        rebin.assert_called_once_with(InputWorkspace=joy, Params=[0, 2, 4], OutputWorkspace="tof_joy_rebin")
        integrate.assert_called_once_with(rebinned, OutputWorkspace="temp_integral")
        divide.assert_called_once_with(
            LHSWorkspace=rebinned, RHSWorkspace="temp_integral", OutputWorkspace="tof_joy_rebin_norm"
        )
        sum_spectra.assert_called_once_with(
            InputWorkspace=normalised,
            WeightedSum=True,
            MultiplyBySpectra=False,
            OutputWorkspace="tof_joy_rebin_norm_avg",
        )
        replace_zeros.assert_not_called()
        symmetrize.assert_not_called()
        delete.assert_called_once_with("temp_integral")
        self.assertEqual(result, (normalised, averaged))

    def test_ncp_masking_replaces_zeros_before_convert_and_uses_standard_rebin(self):
        inputs = self._inputs(mask_time_of_flight_range="100-200", mask_zeros_with=Masking.NCP)
        (
            _,
            ws_tof,
            ws_ncp,
            joy,
            _,
            _,
            _,
            convert,
            rebin,
            masked_rebin,
            _,
            _,
            _,
            replace_zeros,
            _,
            _,
        ) = self._run_with_mocks(inputs)

        replace_zeros.assert_called_once_with(ws_tof, ws_ncp.extractY.return_value)
        convert.assert_called_once_with(replace_zeros.return_value, Mass=1.0, OutputWorkspace="tof_replaced_joy")
        rebin.assert_called_once_with(InputWorkspace=joy, Params=[0, 2, 4], OutputWorkspace="tof_joy_rebin")
        masked_rebin.assert_not_called()

    def test_nan_masking_uses_weighted_rebin_and_symmetrises_average(self):
        inputs = self._inputs(mask_time_of_flight_range="100-200", mask_zeros_with=Masking.NAN, do_symmetrisation=True)
        (
            result,
            _,
            _,
            joy,
            rebinned,
            normalised,
            averaged,
            _,
            rebin,
            masked_rebin,
            _,
            _,
            _,
            replace_zeros,
            symmetrize,
            _,
        ) = self._run_with_mocks(inputs)

        replace_zeros.assert_not_called()
        rebin.assert_not_called()
        masked_rebin.assert_called_once_with(joy, [0, 2, 4], "_rebin")
        symmetrize.assert_called_once_with(averaged)
        self.assertEqual(result, (normalised, symmetrize.return_value))

    def _run_with_y_space_workspace(self):
        ws_tof = CreateWorkspace(
            DataX=[1, 2, 3, 4, 5],
            DataY=[1, 2, 3, 4],
            DataE=[1, 1, 1, 1],
            NSpec=1,
            OutputWorkspace="normalisation_tof",
        )
        ws_joy = CreateWorkspace(
            DataX=[0, 1, 2, 3, 4],
            DataY=[2, 3, 4, 5, 6, 4, 3, 2],
            DataE=[0.2] * 8,
            NSpec=2,
            Distribution=True,
            OutputWorkspace="normalisation_joy",
        )
        inputs = self._inputs()

        with patch.object(reduction_helpers, "ConvertToYSpace", return_value=ws_joy):
            per_spectrum, average = reduction_helpers.y_space_reduction(ws_tof, ws_joy, inputs)

        return per_spectrum, average

    def test_average_workspace_contains_one_spectrum(self):
        _, average = self._run_with_y_space_workspace()

        self.assertEqual(average.getNumberHistograms(), 1)

    def test_each_rebinned_spectrum_is_normalised_to_unit_integral(self):
        per_spectrum, _ = self._run_with_y_space_workspace()

        integrated = Integration(per_spectrum, OutputWorkspace="normalisation_check")

        np.testing.assert_allclose(integrated.extractY(), np.ones((2, 1)), atol=1e-4)


class TestSymmetrizeWeightedMean(unittest.TestCase):
    def tearDown(self):
        AnalysisDataService.clear()

    @staticmethod
    def _create_ws(dataY, dataE):
        dataX = np.linspace(-5, 5, len(dataY))
        return CreateWorkspace(DataX=dataX, DataY=dataY, DataE=dataE, NSpec=1, OutputWorkspace="avg_ws")

    @staticmethod
    def _expected(dataY, dataE):
        """Inverse variance weighting of each point with its mirror point, masked points are ignored."""
        yFlip, eFlip = dataY[::-1], dataE[::-1]
        wY = np.where(dataE > 0, 1 / np.where(dataE > 0, dataE, 1) ** 2, 0)
        wF = np.where(eFlip > 0, 1 / np.where(eFlip > 0, eFlip, 1) ** 2, 0)
        total = wY + wF
        safe = np.where(total > 0, total, 1)
        expY = np.where(total > 0, (wY * dataY + wF * yFlip) / safe, 0)
        expE = np.where(total > 0, 1 / np.sqrt(safe), 0)
        return expY, expE

    def _assert_symmetrised(self, dataY, dataE):
        ws = self._create_ws(dataY, dataE)
        sym = reduction_helpers.symmetrize_weighted_mean(ws)
        symY, symE = sym.extractY()[0], sym.extractE()[0]
        expY, expE = self._expected(dataY, dataE)
        np.testing.assert_allclose(symY, expY, atol=1e-12)
        np.testing.assert_allclose(symE, expE, atol=1e-12)
        np.testing.assert_allclose(symY, symY[::-1])
        np.testing.assert_allclose(symE, symE[::-1])
        np.testing.assert_allclose(sym.extractX()[0], ws.extractX()[0])
        return symY, symE

    def test_output_workspace_name_and_temporary_workspaces_removed(self):
        ws = self._create_ws(np.arange(1.0, 22), np.full(21, 0.1))
        sym = reduction_helpers.symmetrize_weighted_mean(ws)
        self.assertEqual(sym.name(), "avg_ws_sym")
        self.assertNotIn("temp_lhs_reflected", AnalysisDataService.getObjectNames())
        self.assertNotIn("temp_rhs_reflected", AnalysisDataService.getObjectNames())

    def test_no_masked_points(self):
        rng = np.random.default_rng(0)
        for nbins in (21, 20):
            with self.subTest(nbins=nbins):
                self._assert_symmetrised(rng.random(nbins) + 1, rng.random(nbins) * 0.1 + 0.05)

    def test_masked_points_on_one_side_take_value_of_mirror_point(self):
        rng = np.random.default_rng(1)
        for nbins in (21, 20):
            with self.subTest(nbins=nbins):
                dataY = rng.random(nbins) + 1
                dataE = rng.random(nbins) * 0.1 + 0.05
                dataY[:3] = 0
                dataE[:3] = 0
                symY, symE = self._assert_symmetrised(dataY, dataE)
                np.testing.assert_allclose(symY[:3], dataY[::-1][:3])
                np.testing.assert_allclose(symE[:3], dataE[::-1][:3])

    def test_masked_points_on_both_sides(self):
        rng = np.random.default_rng(2)
        dataY = rng.random(21) + 1
        dataE = rng.random(21) * 0.1 + 0.05
        for idx in (0, 1, 19, 20, 7):  # Includes mirrored masked pairs (0, 20) and (1, 19)
            dataY[idx] = 0
            dataE[idx] = 0
        symY, symE = self._assert_symmetrised(dataY, dataE)
        np.testing.assert_array_equal(symY[[0, 1, 19, 20]], 0)
        np.testing.assert_array_equal(symE[[0, 1, 19, 20]], 0)

    def test_central_point_masked(self):
        rng = np.random.default_rng(3)
        dataY = rng.random(21) + 1
        dataE = rng.random(21) * 0.1 + 0.05
        dataY[10] = 0
        dataE[10] = 0
        self._assert_symmetrised(dataY, dataE)

    def test_raises_for_asymmetric_x(self):
        ws = CreateWorkspace(DataX=np.linspace(-4, 5, 11), DataY=np.ones(11), DataE=np.ones(11), NSpec=1)
        with self.assertRaises(AssertionError):
            reduction_helpers.symmetrize_weighted_mean(ws)
