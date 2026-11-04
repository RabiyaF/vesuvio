import unittest
from pathlib import Path
from unittest.mock import patch

from mvesuvio.globals import Tags
from mvesuvio.util import handle_config
from mvesuvio.util.files_manager import FilesManager


class TestFilesManager(unittest.TestCase):
    def test_get_instrument_parameters_dir_reads_cached_path(self):
        with patch.object(handle_config, "read_cached_var", return_value="/instrument/parameters") as read_cached_var:
            result = FilesManager.get_instrument_parameters_dir()

        self.assertEqual(result, Path("/instrument/parameters"))
        read_cached_var.assert_called_once_with("caching.ipfolder")

    def test_get_experiment_dir_reads_cached_path_when_not_set(self):
        with (
            patch.object(FilesManager, "_experiment_dir", None),
            patch.object(handle_config, "read_cached_var", return_value="/experiment") as read_cached_var,
        ):
            result = FilesManager.get_experiment_dir()

        self.assertEqual(result, Path("/experiment"))
        read_cached_var.assert_called_once_with("caching.inputs")

    def test_detector_filename_includes_experiment_kind_and_detector_tag(self):
        with patch.object(handle_config, "get_experiment_name", return_value="experiment"):
            result = FilesManager.get_backward_raw_filename()

        self.assertEqual(result, f"experiment_raw_{Tags.BACKWARD}.nxs")

    def test_summarised_log_file_is_inside_experiment_directory(self):
        experiment_dir = Path("/experiment")
        with patch.object(FilesManager, "get_experiment_dir", return_value=experiment_dir):
            result = FilesManager.get_summarised_log_file()

        self.assertEqual(result, experiment_dir / "summary.log")
