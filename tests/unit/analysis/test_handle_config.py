import unittest
from mock import patch
from mvesuvio.util import handle_config
import tempfile
from textwrap import dedent
import os
from pathlib import Path

class TestHandleConfig(unittest.TestCase):

    def test_read_config(self):
        with tempfile.NamedTemporaryFile(delete=False) as file:
            file.write(dedent("""
                caching.inputs=/inputs.py
                caching.ipfolder=/ip_files
                """).encode())
            file_path = file.name

        try:
            lines = getattr(handle_config, "_read_config")(file_path)
            self.assertEqual(lines, ['\n', "caching.inputs=/inputs.py\n", "caching.ipfolder=/ip_files\n"])
        finally:
            os.unlink(file_path)


    def test_read_config_throws(self):
        with self.assertRaises(RuntimeError):
            getattr(handle_config, "_read_config")("/not.there")


    def test_set_config_vars(self):
        with tempfile.TemporaryDirectory() as tempdir:
            mock_file = Path(tempdir, "mock.vesuvio.properties")
            mock_file.write_text("")

            with (
                patch("mvesuvio.util.handle_config._read_config") as mock_read_config,
                patch.object(handle_config, "VESUVIO_PROPERTIES_PATH", mock_file)
            ):
                mock_read_config.return_value = ['\n', 'caching.inputs=\n', 'caching.ipfolder=\n']
                handle_config.set_config_vars({'caching.inputs': '/inputs.py', 'caching.ipfolder': '/ipfiles'})

                with open(mock_file, "r") as file:
                    self.assertEqual(file.read(), "\ncaching.inputs=/inputs.py\ncaching.ipfolder=/ipfiles\n")


    def test_set_default_config_vars(self):
        with tempfile.TemporaryDirectory() as tempdir:
            mock_file = Path(tempdir, "mock.vesuvio.properties")
            mock_file.write_text("")

            with (
                patch("mvesuvio.util.handle_config._read_config") as mock_read_config,
                patch.object(handle_config, "VESUVIO_PROPERTIES_PATH", mock_file),
                patch.object(handle_config, "USER_CONFIG_PATH", Path("path", "to", ".mvesuvio")),
            ):
                mock_read_config.return_value = ['\n', 'caching.inputs=\n', 'caching.ipfolder=\n']
                handle_config.set_default_config_vars()

                with open(mock_file, "r") as file:
                    self.assertEqual(file.read(), f"\ncaching.inputs={str(Path('path', 'to', '.mvesuvio', 'experiment_template'))}\ncaching.ipfolder={str(Path('path', 'to', '.mvesuvio', 'ip_files'))}\n")


    def test_read_config_vars(self):
        with tempfile.TemporaryDirectory() as tempdir:
            mock_file = Path(tempdir, "mock.vesuvio.properties")
            mock_file.write_text("\ncaching.inputs=/inputs.py\ncaching.ipfolder=/ipfiles\n")

            with (
                patch.object(handle_config, "VESUVIO_PROPERTIES_PATH", mock_file),
            ):
                self.assertEqual(handle_config.read_cached_var('caching.inputs'), '/inputs.py')
                self.assertEqual(handle_config.read_cached_var('caching.ipfolder'), '/ipfiles')


    def test_read_config_vars_throws(self):

        with tempfile.TemporaryDirectory() as tempdir:
            mock_file = Path(tempdir, "mock.vesuvio.properties")
            mock_file.write_text("\ncaching.inputs=/inputs.py\ncaching.ipfolder=/ipfiles\n")

            with (
                patch.object(handle_config, "VESUVIO_PROPERTIES_PATH", mock_file),
                self.assertRaises(ValueError)
            ):
                handle_config.read_cached_var('non.existent')


    def test_read_cached_var_uses_default_package_properties_path(self):
        with tempfile.TemporaryDirectory() as tempdir:
            mock_file = Path(tempdir, "mock.vesuvio.properties")
            with (
                patch("mvesuvio.util.handle_config._read_config") as mock_read_config,
                patch.object(handle_config, "VESUVIO_PROPERTIES_PATH", mock_file),
            ):
                mock_read_config.return_value = ["caching.inputs=/inputs.py\n"]
                handle_config.read_cached_var("caching.inputs")
                mock_read_config.assert_called_once_with(mock_file, True)


    def test_set_config_vars_uses_default_package_properties_path(self):
        with tempfile.TemporaryDirectory() as tempdir:
            mock_file = Path(tempdir, "mock.vesuvio.properties")
            with (
                patch("mvesuvio.util.handle_config._read_config") as mock_read_config,
                patch.object(handle_config, "VESUVIO_PROPERTIES_PATH", mock_file),
                patch("mvesuvio.util.handle_config.open", create=True) as mock_open,
            ):
                mock_read_config.return_value = ["caching.inputs=\n"]
                handle_config.set_config_vars({"caching.inputs": "/inputs.py"})

                mock_read_config.assert_called_once_with(mock_file)
                mock_open.assert_called_once_with(mock_file, "w")


    def test_get_experiment_name(self):
        with patch("mvesuvio.util.handle_config.read_cached_var") as mock_read_cached_var:
            mock_read_cached_var.return_value = str(Path("path", "to", "experiment"))
            self.assertEqual(handle_config.get_experiment_name(), "experiment")


    def test_refresh_config_dir_and_contents_creates_user_config_dir_when_missing(self):
        with tempfile.TemporaryDirectory() as tempdir:
            missing_dir = Path(tempdir, "missing_config_dir")

            self.assertFalse(missing_dir.exists())

            with patch.object(handle_config, "USER_CONFIG_PATH", missing_dir):
                handle_config.refresh_config_dir_and_contents()

            self.assertTrue(missing_dir.exists())


    def test_refresh_config_dir_and_contents_when_dir_exists_does_not_reset_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            preserved_file = Path(tempdir, "custom_local_file.txt")
            preserved_file.write_text("keep me")

            with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir)):
                handle_config.refresh_config_dir_and_contents()

            self.assertTrue(preserved_file.exists())
            self.assertEqual(preserved_file.read_text(), "keep me")


    def test_refresh_config_dir_and_contents_copies_dict_config_when_dir_missing(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir)):
                handle_config.refresh_config_dir_and_contents()

            self.assertTrue(Path(tempdir, "ip_files").exists())
            self.assertTrue(Path(tempdir, "vesuvio.plots.mplstyle").exists())
            self.assertTrue(Path(tempdir, "experiment_template", "script_to_create_figures.py").exists())
            self.assertTrue(Path(tempdir, "experiment_template", "run_reduction.py").exists())
            self.assertTrue(Path(tempdir, "experiment_template", "run_fitting.py").exists())


    def test_refresh_config_dir_and_contents_never_copies_dunder_files(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir)):
                handle_config.refresh_config_dir_and_contents()

            self.assertFalse(Path(tempdir, "__init__.py").exists())
            self.assertFalse(Path(tempdir, "__pycache__").exists())
            self.assertFalse(any(Path(tempdir).rglob("__*.py")))


    def test_refresh_config_dir_and_contents_never_copies_vesuvio_user_properties(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir)):
                handle_config.refresh_config_dir_and_contents()

            self.assertFalse(Path(tempdir, "vesuvio.user.properties").exists())


    def test_refresh_config_dir_and_contents_dont_overwrite_copy_if_present_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            template_dir = Path(tempdir, "experiment_template")
            template_dir.mkdir(parents=True, exist_ok=True)
            script_path = template_dir / "script_to_create_figures.py"
            script_path.write_text("mock script")

            with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir)):
                handle_config.refresh_config_dir_and_contents()

            self.assertEqual(script_path.read_text(), "mock script")


    def test_refresh_config_dir_and_contents_dont_overwrite_copy_if_present_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            ip_dir = Path(tempdir, "ip_files")
            ip_dir.mkdir(parents=True, exist_ok=True)
            marker = ip_dir / "ip.par"
            marker.write_text("keep")

            with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir)):
                handle_config.refresh_config_dir_and_contents()

            self.assertEqual(marker.read_text(), "keep")


    def test_refresh_config_dir_and_contents_overwrites_always_copy_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            target = Path(tempdir, "experiment_template", "run_reduction.py")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("old content")

            with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir)):
                handle_config.refresh_config_dir_and_contents()

            self.assertEqual(
                target.read_text(),
                Path(
                    handle_config.PACKAGE_CONFIG_PATH,
                    "experiment_template",
                    "run_reduction.py",
                ).read_text(),
            )


    def test_refresh_config_dir_and_contents_ignores_dunder_files_in_ip_dir_copy(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir)):
                handle_config.refresh_config_dir_and_contents()

            self.assertFalse(Path(tempdir, "ip_files", "__init__.py").exists())

    def test_copy_config_entry_rejects_unknown_entry_type(self):
        with self.assertRaisesRegex(ValueError, "Unknown config entry type: unknown"):
            handle_config._copy_config_entry("entry", "unknown", overwrite=False)

    def test_is_cache_set_returns_true_when_input_path_is_configured(self):
        with patch.object(handle_config, "read_cached_var", return_value="/inputs") as read_cached_var:
            result = handle_config.is_cache_set()

        self.assertTrue(result)
        read_cached_var.assert_called_once_with("caching.inputs", False)

    def test_is_cache_set_returns_false_when_input_path_is_empty(self):
        with patch.object(handle_config, "read_cached_var", return_value="") as read_cached_var:
            result = handle_config.is_cache_set()

        self.assertFalse(result)
        read_cached_var.assert_called_once_with("caching.inputs", False)
