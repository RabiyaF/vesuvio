import runpy
import unittest
from pathlib import Path
from shutil import copytree, rmtree
from unittest.mock import patch

from mantid.api import AnalysisDataService

from mvesuvio.util import handle_config
from mvesuvio.util.files_manager import FilesManager

TESTS_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_INPUTS_PATH = TESTS_ROOT / "data" / "analysis" / "inputs" / "bootstrap"


class TestBootstrap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FilesManager._experiment_dir = None
        handle_config.refresh_config_dir_and_contents()
        handle_config.set_default_config_vars()
        cls.bootstrap_inputs_path = FilesManager.get_bootstrap_inputs_dir()
        cls.bootstrap_output_path = FilesManager.get_bootstrap_outputs_dir()
        rmtree(cls.bootstrap_inputs_path, ignore_errors=True)
        rmtree(cls.bootstrap_output_path, ignore_errors=True)
        copytree(BOOTSTRAP_INPUTS_PATH, cls.bootstrap_inputs_path, dirs_exist_ok=True)
        AnalysisDataService.clear()

    @classmethod
    def tearDownClass(cls) -> None:
        # NOTE: Very important to reset experiment directory, otherwise subsequent tests may be affected.
        FilesManager._experiment_dir = None
        AnalysisDataService.clear()

    def test_bootstrap_routine(self):
        bootstrap_script = handle_config.PACKAGE_CONFIG_PATH / "experiment_template" / "run_bootstrap.py"
        namespace = runpy.run_path(str(bootstrap_script), run_name="test_bootstrap_run_bootstrap")

        with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"), patch("matplotlib.figure.Figure.savefig"):
            namespace["run_bootstrap"]()

        for sample_index in range(1, 4):
            sample_output_dir = self.bootstrap_output_path / f"boot_{sample_index}"
            self.assertTrue(sample_output_dir.is_dir(), f"Bootstrap output missing for sample {sample_index}")

            expected_dirs = {"fitting_inputs", "fitting_outputs", "reduction_outputs"}
            actual_dirs = {path.name for path in sample_output_dir.iterdir() if path.is_dir()}
            self.assertSetEqual(
                expected_dirs,
                actual_dirs,
                f"Bootstrap output for sample {sample_index} should contain exactly the expected folders: {sorted(expected_dirs)}",
            )

            for subdir_name in sorted(expected_dirs):
                subdir = sample_output_dir / subdir_name
                self.assertTrue(any(subdir.iterdir()), f"Bootstrap output folder {subdir_name} for sample {sample_index} is empty")


if __name__ == "__main__" or __name__ == "mantidqt.widgets.codeeditor.execution":
    unittest.main()
