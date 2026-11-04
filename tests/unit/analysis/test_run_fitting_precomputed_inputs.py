import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mantid.simpleapi import AnalysisDataService, CreateWorkspace

from mvesuvio.default_config.experiment_template.run_fitting import ForwardFittingInputs, run_y_space_reduction_and_fit
from mvesuvio.util.files_manager import FilesManager


class TestRunFittingPrecomputedInputs(unittest.TestCase):
    def tearDown(self):
        AnalysisDataService.clear()

    @patch("mvesuvio.default_config.experiment_template.run_fitting.SaveAscii")
    @patch("mvesuvio.default_config.experiment_template.run_fitting.FitInYSpace")
    def test_runs_fit_when_precomputed_workspaces_exist(self, mock_fit_class, mock_save_ascii):
        with tempfile.TemporaryDirectory() as output_dir:
            FilesManager.set_experiment_dir(output_dir)
            AnalysisDataService.clear()

            ws_prefix = f"{ForwardFittingInputs.name_of_subtracted_workspace}_0"
            CreateWorkspace(
                DataX=[0, 1, 2],
                DataY=[1, 2, 3],
                DataE=[1, 1, 1],
                NSpec=1,
                UnitX="TOF",
                OutputWorkspace=f"{ws_prefix}_ws_resolution",
            )
            CreateWorkspace(
                DataX=[0, 1, 2],
                DataY=[2, 3, 4],
                DataE=[1, 1, 1],
                NSpec=1,
                UnitX="TOF",
                OutputWorkspace=f"{ws_prefix}_joy",
            )
            CreateWorkspace(
                DataX=[0, 1, 2],
                DataY=[3, 4, 5],
                DataE=[1, 1, 1],
                NSpec=1,
                UnitX="TOF",
                OutputWorkspace=f"{ws_prefix}_joy_avg",
            )

            ForwardFittingInputs.number_of_iterations_for_corrections = 0

            result = run_y_space_reduction_and_fit(ForwardFittingInputs)

            self.assertTrue(result)
            mock_fit_class.assert_called_once()
            mock_fit_class.return_value.run.assert_called_once_with()
            mock_save_ascii.assert_called_once_with(
                f"{ws_prefix}_ws_resolution",
                str(Path(output_dir) / "fitting_outputs" / f"{ws_prefix}_ws_resolution"),
            )

    @patch("mvesuvio.default_config.experiment_template.run_fitting.FitInYSpace")
    def test_skips_fit_when_precomputed_workspaces_missing(self, mock_fit_class):
        AnalysisDataService.clear()

        ForwardFittingInputs.name = "front"
        ForwardFittingInputs.number_of_iterations_for_corrections = 0

        with patch("mvesuvio.default_config.experiment_template.run_fitting.logger") as mock_logger:
            result = run_y_space_reduction_and_fit(ForwardFittingInputs)

        self.assertFalse(result)
        mock_fit_class.assert_not_called()
        mock_logger.warning.assert_called_once()


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    unittest.main()
