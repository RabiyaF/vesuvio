import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mvesuvio.util import bootstrap_helpers
from mvesuvio.util.files_manager import FilesManager


class TestBootstrapHelpers(unittest.TestCase):
    def test_get_bootstrap_input_directories_rejects_non_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing"
            with patch.object(bootstrap_helpers.logger, "error") as error:
                result = bootstrap_helpers.get_bootstrap_input_directories(missing_path)

        self.assertEqual(result, ())
        error.assert_called_once()

    def test_get_bootstrap_input_directories_creates_missing_backward_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inputs_dir = Path(temp_dir) / "inputs"
            inputs_dir.mkdir()
            result = bootstrap_helpers.get_bootstrap_input_directories(inputs_dir)

            self.assertEqual(result, ())
            self.assertTrue((inputs_dir / "backward").is_dir())
            self.assertFalse((inputs_dir / "forward").exists())

    def test_get_bootstrap_input_directories_creates_missing_forward_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inputs_dir = Path(temp_dir) / "inputs"
            (inputs_dir / "backward").mkdir(parents=True)
            result = bootstrap_helpers.get_bootstrap_input_directories(inputs_dir)

            self.assertEqual(result, ())
            self.assertTrue((inputs_dir / "forward").is_dir())

    def test_get_bootstrap_input_directories_returns_existing_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inputs_dir = Path(temp_dir) / "inputs"
            backward_dir = inputs_dir / "backward"
            forward_dir = inputs_dir / "forward"
            backward_dir.mkdir(parents=True)
            forward_dir.mkdir()

            result = bootstrap_helpers.get_bootstrap_input_directories(inputs_dir)

        self.assertEqual(result, (inputs_dir, backward_dir, forward_dir))

    def test_get_common_prefix_of_bootstrap_sample_names(self):
        self.assertEqual(bootstrap_helpers.get_common_prefix_of_bootstrap_sample_names("sample_01a", "sample_02a"), "sample")

    def test_get_common_prefix_of_bootstrap_sample_names_rejects_invalid_pairs(self):
        invalid_pairs = [
            ("sample_01", "sample_01"),
            ("sample_01", "sample_02"),
            ("a1", "b1"),
        ]
        with patch.object(bootstrap_helpers.logger, "error") as error:
            for back_name, front_name in invalid_pairs:
                with self.subTest(back_name=back_name, front_name=front_name):
                    self.assertEqual(bootstrap_helpers.get_common_prefix_of_bootstrap_sample_names(back_name, front_name), "")

        self.assertEqual(error.call_count, len(invalid_pairs))

    def test_get_bootstrap_sample_sort_key_uses_last_number_or_filename(self):
        self.assertEqual(bootstrap_helpers.get_bootstrap_sample_sort_key(Path("sample_2_10.nxs")), (10, "sample_2_10.nxs"))
        self.assertEqual(bootstrap_helpers.get_bootstrap_sample_sort_key(Path("sample.nxs")), (0, "sample.nxs"))

    def test_pair_bootstrap_sample_paths_sorts_and_pairs_matching_counts(self):
        backward = [Path("sample_2.nxs"), Path("sample_1.nxs")]
        forward = [Path("sample_1_front.nxs"), Path("sample_2_front.nxs")]

        pairs = bootstrap_helpers.pair_bootstrap_sample_paths(backward, forward)

        self.assertEqual(
            pairs,
            [
                (Path("sample_1.nxs"), Path("sample_1_front.nxs")),
                (Path("sample_2.nxs"), Path("sample_2_front.nxs")),
            ],
        )

    def test_pair_bootstrap_sample_paths_warns_and_truncates_mismatched_counts(self):
        backward = [Path("sample_3.nxs"), Path("sample_1.nxs"), Path("sample_2.nxs")]
        forward = [Path("sample_2_front.nxs"), Path("sample_1_front.nxs")]

        with patch.object(bootstrap_helpers.logger, "warning") as warning:
            pairs = bootstrap_helpers.pair_bootstrap_sample_paths(backward, forward)

        self.assertEqual(
            pairs,
            [
                (Path("sample_1.nxs"), Path("sample_1_front.nxs")),
                (Path("sample_2.nxs"), Path("sample_2_front.nxs")),
            ],
        )
        warning.assert_called_once()

    def test_apply_bootstrap_overrides_sets_paths_names_and_disables_plots(self):
        back_inputs = SimpleNamespace(name="back", show_plots=True)
        front_inputs = SimpleNamespace(name="front", show_plots=True)
        back_path = Path("/inputs/backward/sample_back.nxs")
        front_path = Path("/inputs/forward/sample_front.nxs")

        bootstrap_helpers.apply_bootstrap_overrides(back_inputs, front_inputs, back_path, front_path)

        for inputs, path in ((back_inputs, back_path), (front_inputs, front_path)):
            self.assertEqual(inputs.override_input_workspace, str(path.absolute()))
            self.assertEqual(inputs.overwrite_analysis_input_workspace, str(path.absolute()))
            self.assertEqual(inputs.name, path.stem)
            self.assertFalse(inputs.show_plots)

    def test_apply_bootstrap_overrides_clears_paths_without_optional_plot_attributes(self):
        back_inputs = SimpleNamespace(name="back")
        front_inputs = SimpleNamespace(name="front")

        bootstrap_helpers.apply_bootstrap_overrides(back_inputs, front_inputs, None, None)

        self.assertEqual(back_inputs.override_input_workspace, "")
        self.assertEqual(front_inputs.override_input_workspace, "")
        self.assertEqual(back_inputs.overwrite_analysis_input_workspace, "")
        self.assertEqual(front_inputs.overwrite_analysis_input_workspace, "")
        self.assertEqual(back_inputs.name, "back")
        self.assertEqual(front_inputs.name, "front")

    def test_update_sample_inputs_outputs_creates_directory_updates_overrides_and_sets_experiment_dir(self):
        back_inputs = SimpleNamespace(name="back", show_plots=True)
        front_inputs = SimpleNamespace(name="front", show_plots=True)
        back_path = Path("/inputs/backward/sample_back.nxs")
        front_path = Path("/inputs/forward/sample_front.nxs")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output"
            with patch.object(FilesManager, "set_experiment_dir") as set_experiment_dir:
                bootstrap_helpers.update_sample_inputs_outputs(
                    back_inputs, front_inputs, back_path, front_path, output_path
                )

            self.assertTrue(output_path.is_dir())
        self.assertEqual(back_inputs.name, back_path.stem)
        self.assertEqual(front_inputs.name, front_path.stem)
        self.assertFalse(back_inputs.show_plots)
        self.assertFalse(front_inputs.show_plots)
        set_experiment_dir.assert_called_once_with(output_path)
