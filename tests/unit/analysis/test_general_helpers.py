import os
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock, patch

import numpy as np
from mantid.simpleapi import AnalysisDataService, CreateWorkspace, DeleteWorkspace, RenameWorkspace, SaveNexus

from mvesuvio.globals import Mode, ScatteringType
from mvesuvio.util import general_helpers


class TestGeneralHelpers(unittest.TestCase):

    def test_get_scattering_type_returns_backward_enum(self):
        class BackwardInputs:
            mode = Mode.DOUBLE_DIFFERENCE.value

        self.assertIs(general_helpers._get_scattering_type(BackwardInputs), ScatteringType.BACKWARD)

    def test_get_scattering_type_returns_forward_enum(self):
        class ForwardInputs:
            mode = Mode.SINGLE_DIFFERENCE.value

        self.assertIs(general_helpers._get_scattering_type(ForwardInputs), ScatteringType.FORWARD)

    def test_get_scattering_type_rejects_unrecognized_inputs(self):
        class UnknownInputs:
            mode = "unknown"

        with self.assertRaisesRegex(ValueError, "Could not infer scattering type"):
            general_helpers._get_scattering_type(UnknownInputs)

    def test_load_overwritten_workspace_if_specified(self):
        class Inputs:
            overwrite_analysis_input_workspace = "override_front"

        ws = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], DataE=[1, 1, 1], NSpec=1, UnitX="TOF")
        RenameWorkspace(ws, "override_front")

        result = general_helpers.load_overwritten_workspace_if_specified(Inputs)

        self.assertEqual(result, "override_front")
        self.assertTrue(AnalysisDataService.doesExist("override_front_sum"))

        DeleteWorkspace("override_front")
        DeleteWorkspace("override_front_sum")

    def test_load_overwritten_workspace_if_specified_for_path_override(self):
        class Inputs:
            overwrite_analysis_input_workspace = ""

        with tempfile.TemporaryDirectory() as temp_dir:
            ws = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], DataE=[1, 1, 1], NSpec=1, UnitX="TOF")
            path = os.path.join(temp_dir, "override_front_from_path.nxs")
            SaveNexus(ws, path)
            DeleteWorkspace(ws)

            Inputs.overwrite_analysis_input_workspace = path

            result = general_helpers.load_overwritten_workspace_if_specified(Inputs)

            self.assertEqual(result, "override_front_from_path")
            self.assertTrue(AnalysisDataService.doesExist("override_front_from_path_sum"))

            DeleteWorkspace("override_front_from_path")
            DeleteWorkspace("override_front_from_path_sum")

    def test_load_overwritten_workspace_if_specified_for_empty_string(self):
        class Inputs:
            overwrite_analysis_input_workspace = ""

        result = general_helpers.load_overwritten_workspace_if_specified(Inputs)

        self.assertEqual(result, "")

    def test_load_overwritten_workspace_if_specified_rejects_missing_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = str(Path(temp_dir) / "missing.nxs")
            inputs = Mock(overwrite_analysis_input_workspace=missing_path)
            with (
                patch.object(general_helpers, "AnalysisDataService", Mock(doesExist=Mock(return_value=False))),
                self.assertRaisesRegex(ValueError, f"Override workspace not in ADS and not a path: {missing_path}"),
            ):
                general_helpers.load_overwritten_workspace_if_specified(inputs)

    def test_extract_ws(self):
        data = [1, 2, 3]
        ws = CreateWorkspace(DataX=data, DataY=data, DataE=data, NSpec=1, UnitX="some_unit")

        dataX, dataY, dataE = general_helpers.extractWS(ws)
        np.testing.assert_array_equal([data], dataX)
        np.testing.assert_array_equal([data], dataY)
        np.testing.assert_array_equal([data], dataE)

        DeleteWorkspace(ws)

    def test_pass_data_into_single_histogram_workspace_with_1d_arrays(self):
        ws = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], DataE=[1, 1, 1], NSpec=1, UnitX="some_unit")
        data_x = np.array([3, 4, 5])
        data_y = np.array([6, 7, 8])
        data_e = np.array([0.1, 0.2, 0.3])

        result = general_helpers.pass_data_into_ws(data_x, data_y, data_e, ws)

        self.assertIs(result, ws)
        np.testing.assert_array_equal(data_x, ws.readX(0))
        np.testing.assert_array_equal(data_y, ws.readY(0))
        np.testing.assert_array_equal(data_e, ws.readE(0))
        DeleteWorkspace(ws)

    def test_make_summarised_log_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as mock_mantid_log_file, tempfile.NamedTemporaryFile(delete=False) as mock_summary_file:
            mock_mantid_log_file.write(
                dedent(
                    """
                    2025-01-08 10:48:44,832 [0] Notice CreateWorkspace - CreateWorkspace started (child)
                    2025-01-08 10:48:44,844 [0] Notice CreateWorkspace - CreateWorkspace successful, Duration 0.01 seconds
                    2025-01-08 10:48:44,860 [0] Notice VesuvioAnalysisRoutine -
                    Fitting neutron compton profiles ...
                    2025-01-08 10:48:45,319 [0] Notice VesuvioAnalysisRoutine - Fit spectrum 148: ✓
                    2025-01-08 10:48:45,517 [0] Warning Python - Values in x were outside bounds during a minimize step, clipping to bounds
                    2025-01-08 10:48:45,623 [0] Notice VesuvioAnalysisRoutine - Fit spectrum 151: ✓
                    2025-01-08 10:48:48,568 [0] Notice CreateEmptyTableWorkspace - CreateEmptyTableWorkspace started (child)
                    2025-01-08 10:48:48,570 [0] Notice CreateEmptyTableWorkspace - CreateEmptyTableWorkspace successful, Duration 0.00 seconds
                    2025-01-08 10:48:48,573 [0] Notice Python - Table analysis_inputs_fwd_0_means:
                    2025-01-08 10:48:48,574 [0] Notice Python -  ----------------------------------------------------------------
                    2025-01-08 10:48:48,576 [0] Notice Python - |label |mass   |mean_width|std_width|mean_intensity|std_intensity|
                    2025-01-08 10:48:48,578 [0] Notice Python - |1.0079|1.00790|5.29627   |0.19464  |0.91410       |0.00862      |
                    2025-01-08 10:48:48,584 [0] Notice Python -  ----------------------------------------------------------------
                    2025-01-08 10:48:48,588 [0] Notice VesuvioAnalysisRoutine - VesuvioAnalysisRoutine successful, Duration 3.89 seconds
                    2025-01-08 10:48:49,390 [0] Notice Python -
                    Shared Parameters: ['sigma']
                    2025-01-08 10:48:49,391 [0] Notice Python -
                    Unshared Parameters: ['A', 'x0']
                    """
                ).encode()
            )

            mock_mantid_log_file_path = mock_mantid_log_file.name
            mock_summary_file_path = mock_summary_file.name

        with patch("mvesuvio.util.general_helpers.FilesManager.get_mantid_log_file", return_value=mock_mantid_log_file_path):
            with patch("mvesuvio.util.general_helpers.FilesManager.get_summarised_log_file", return_value=mock_summary_file_path):
                general_helpers.make_summarised_log_file()

        with open(mock_summary_file_path, "r", encoding="utf-8") as summary_file:
            current_log_file_content = summary_file.read()

        self.assertEqual(
            dedent(
                """
                2025-01-08 10:48:44,860 [0] Notice VesuvioAnalysisRoutine -
                Fitting neutron compton profiles ...
                2025-01-08 10:48:45,319 [0] Notice VesuvioAnalysisRoutine - Fit spectrum 148: ✓
                2025-01-08 10:48:45,623 [0] Notice VesuvioAnalysisRoutine - Fit spectrum 151: ✓
                2025-01-08 10:48:48,573 [0] Notice Python - Table analysis_inputs_fwd_0_means:
                2025-01-08 10:48:48,574 [0] Notice Python -  ----------------------------------------------------------------
                2025-01-08 10:48:48,576 [0] Notice Python - |label |mass   |mean_width|std_width|mean_intensity|std_intensity|
                2025-01-08 10:48:48,578 [0] Notice Python - |1.0079|1.00790|5.29627   |0.19464  |0.91410       |0.00862      |
                2025-01-08 10:48:48,584 [0] Notice Python -  ----------------------------------------------------------------
                2025-01-08 10:48:48,588 [0] Notice VesuvioAnalysisRoutine - VesuvioAnalysisRoutine successful, Duration 3.89 seconds
                2025-01-08 10:48:49,390 [0] Notice Python -
                Shared Parameters: ['sigma']
                2025-01-08 10:48:49,391 [0] Notice Python -
                Unshared Parameters: ['A', 'x0']
                             """
            ),
            current_log_file_content,
        )

        os.remove(mock_mantid_log_file_path)
        os.remove(mock_summary_file_path)

    def test_make_summarised_log_file_logs_when_mantid_log_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_log = Path(temp_dir) / "missing.log"
            with (
                patch.object(general_helpers.FilesManager, "get_mantid_log_file", return_value=missing_log),
                patch.object(general_helpers.logger, "error") as error,
            ):
                general_helpers.make_summarised_log_file()

        error.assert_called_once_with("Mantid log file not available. Unable to produce a summarized log file for this routine.")
