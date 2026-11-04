import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mvesuvio.main import _set_up_parser, _setup_and_parse_args, _setup_config


class TestParser(unittest.TestCase):
    """Test cases for command-line parser setup."""

    def setUp(self):
        self.parser = _set_up_parser()

    def test_set_up_parser_config(self):
        args = self.parser.parse_args(["config", "--experiment-dir", "analysis_inputs.py", "--ip-dir", "mock_ip_folder"])

        self.assertEqual(args.experiment_dir, "analysis_inputs.py")
        self.assertEqual(args.ip_dir, "mock_ip_folder")

    def test_set_up_parser_config_defaults(self):
        args = self.parser.parse_args(["config"])

        self.assertEqual(args.experiment_dir, "")
        self.assertEqual(args.ip_dir, "")

    def test_set_up_parser_run(self):
        args = self.parser.parse_args(["run"])

        self.assertEqual(args.command, "run")

    def test_set_up_parser_bootstrap(self):
        args = self.parser.parse_args(["bootstrap"])

        self.assertEqual(args.command, "bootstrap")

    def test_set_up_parser_config_short_flags(self):
        args = self.parser.parse_args(["config", "-e", "inputs.py", "-i", "/ip/folder"])

        self.assertEqual(args.experiment_dir, "inputs.py")
        self.assertEqual(args.ip_dir, "/ip/folder")

    def test_set_up_parser_requires_command(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])

    @patch("sys.argv", ["mvesuvio", "version"])
    def test_setup_and_parse_args(self):
        args = _setup_and_parse_args()

        self.assertEqual(args.command, "version")


class TestSetupConfig(unittest.TestCase):
    """Test cases for _setup_config function."""

    @patch("mvesuvio.main.handle_config")
    @patch("mantid.kernel.ConfigService")
    def test_setup_config_with_args_but_no_custom_paths(self, mock_config_service, mock_handle_config):
        mock_handle_config.is_cache_set.return_value = True
        mock_handle_config.read_cached_var.side_effect = ["/default/inputs.py", "/default/ip_folder"]

        mock_args = MagicMock()
        mock_args.experiment_dir = None
        mock_args.ip_dir = None

        _setup_config(mock_args)

        mock_handle_config.set_config_vars.assert_called_once()
        call_args = mock_handle_config.set_config_vars.call_args[0][0]
        self.assertEqual(call_args["caching.inputs"], "/default/inputs.py")
        self.assertEqual(call_args["caching.ipfolder"], "/default/ip_folder")

    @patch("mvesuvio.main.Path")
    @patch("mvesuvio.main.handle_config")
    @patch("mantid.kernel.ConfigService")
    def test_setup_config_with_custom_inputs_path(self, mock_config_service, mock_handle_config, mock_path):
        mock_handle_config.is_cache_set.return_value = True
        mock_handle_config.read_cached_var.side_effect = ["/default/inputs.py", "/default/ip_folder"]

        mock_path_obj = MagicMock()
        mock_path_obj.absolute.return_value = "/absolute/custom/inputs.py"
        mock_path.return_value = mock_path_obj

        mock_args = MagicMock()
        mock_args.experiment_dir = "/custom/inputs.py"
        mock_args.ip_dir = None

        _setup_config(mock_args)

        mock_path.assert_called_with("/custom/inputs.py")
        mock_path_obj.absolute.assert_called_once()

        mock_handle_config.set_config_vars.assert_called_once()
        call_args = mock_handle_config.set_config_vars.call_args[0][0]
        self.assertEqual(call_args["caching.inputs"], "/absolute/custom/inputs.py")
        self.assertEqual(call_args["caching.ipfolder"], "/default/ip_folder")

    @patch("mvesuvio.main.Path")
    @patch("mvesuvio.main.handle_config")
    @patch("mantid.kernel.ConfigService")
    def test_setup_config_warns_for_invalid_custom_paths(self, mock_config_service, mock_handle_config, mock_path):
        mock_handle_config.is_cache_set.return_value = True
        mock_handle_config.read_cached_var.side_effect = ["/default/inputs.py", "/default/ip_folder"]

        mock_experiment_path = MagicMock()
        mock_experiment_path.is_dir.return_value = False
        mock_ip_path = MagicMock()
        mock_ip_path.is_dir.return_value = False
        mock_path.side_effect = [mock_experiment_path, mock_ip_path]

        mock_args = MagicMock()
        mock_args.experiment_dir = "/custom/inputs.py"
        mock_args.ip_dir = "/custom/ip_folder"

        with patch("builtins.print") as mock_print:
            _setup_config(mock_args)

        self.assertEqual(mock_print.call_count, 2)
        mock_print.assert_any_call("\nError setting directory: /custom/inputs.py\nUsing default.")
        mock_print.assert_any_call("\nError setting directory: /custom/ip_folder\nUsing default.")
        mock_handle_config.set_config_vars.assert_called_once()
        call_args = mock_handle_config.set_config_vars.call_args[0][0]
        self.assertEqual(call_args["caching.inputs"], "/default/inputs.py")
        self.assertEqual(call_args["caching.ipfolder"], "/default/ip_folder")

    @patch("mvesuvio.main.Path")
    @patch("mvesuvio.main.handle_config")
    @patch("mantid.kernel.ConfigService")
    def test_setup_config_with_custom_ip_folder(self, mock_config_service, mock_handle_config, mock_path):
        mock_handle_config.is_cache_set.return_value = True
        mock_handle_config.read_cached_var.side_effect = ["/default/inputs.py", "/default/ip_folder"]

        mock_path_obj = MagicMock()
        mock_path_obj.absolute.return_value = "/absolute/custom/ip_folder"
        mock_path.return_value = mock_path_obj

        mock_args = MagicMock()
        mock_args.experiment_dir = None
        mock_args.ip_dir = "/custom/ip_folder"

        _setup_config(mock_args)

        mock_path.assert_called_with("/custom/ip_folder")
        mock_path_obj.absolute.assert_called_once()

        mock_handle_config.set_config_vars.assert_called_once()
        call_args = mock_handle_config.set_config_vars.call_args[0][0]
        self.assertEqual(call_args["caching.inputs"], "/default/inputs.py")
        self.assertEqual(call_args["caching.ipfolder"], "/absolute/custom/ip_folder")

    @patch("mvesuvio.main.handle_config")
    @patch("mantid.kernel.ConfigService")
    def test_setup_config_calls_mantid_logging_setup(self, mock_config_service, mock_handle_config):
        mock_handle_config.is_cache_set.return_value = True
        mock_handle_config.read_cached_var.side_effect = ["/default/inputs.py", "/default/ip_folder"]

        _setup_config(None)

        # ConfigService.setString should be called multiple times
        self.assertGreater(mock_config_service.setString.call_count, 0)

    @patch("mvesuvio.main.handle_config")
    @patch("mantid.kernel.ConfigService")
    def test_setup_config_sets_defaults_when_cache_is_not_set(self, mock_config_service, mock_handle_config):
        mock_handle_config.is_cache_set.return_value = False
        mock_handle_config.read_cached_var.side_effect = ["/default/inputs.py", "/default/ip_folder"]

        _setup_config(None)

        mock_handle_config.set_default_config_vars.assert_called_once_with()
        mock_handle_config.set_config_vars.assert_called_once_with(
            {
                "caching.inputs": "/default/inputs.py",
                "caching.ipfolder": "/default/ip_folder",
            }
        )


class TestMainFunction(unittest.TestCase):

    @patch("mvesuvio.version", return_value="1.2.3")
    def test_main_with_version_command(self, mock_version):
        from mvesuvio.main import main

        with patch("builtins.print") as mock_print:
            main(manual_args=MagicMock(command="version"))

        mock_version.assert_called_once_with()
        mock_print.assert_called_once_with("1.2.3")

    @patch("mvesuvio.main._run_bootstrap")
    @patch("mvesuvio.main._run_analysis")
    @patch("mvesuvio.main._setup_config")
    @patch("mvesuvio.main.handle_config")
    def test_main_with_config_command(self, mock_config, mock_setup_config, mock_run_analysis, mock_run_bootstrap):
        from mvesuvio.main import main

        mock_args = MagicMock()
        mock_args.command = "config"

        main(manual_args=mock_args)

        mock_setup_config.assert_called_once_with(mock_args)

    @patch("mvesuvio.main._run_bootstrap")
    @patch("mvesuvio.main._run_analysis")
    @patch("mvesuvio.main._setup_config")
    @patch("mvesuvio.main.handle_config")
    def test_main_with_run_command(self, mock_config, mock_setup_config, mock_run_analysis, mock_run_bootstrap):
        from mvesuvio.main import main

        mock_args = MagicMock()
        mock_args.command = "run"
        mock_config.is_cache_set.return_value = True

        main(manual_args=mock_args)

        mock_run_analysis.assert_called_once_with(mock_args)

    @patch("mvesuvio.main._run_bootstrap")
    @patch("mvesuvio.main._run_analysis")
    @patch("mvesuvio.main._setup_config")
    @patch("mvesuvio.main.handle_config")
    def test_main_setup_config_if_not_configured(self, mock_config, mock_setup_config, mock_run_analysis, mock_run_bootstrap):
        from mvesuvio.main import main

        mock_args = MagicMock()
        mock_args.command = "run"
        mock_config.is_cache_set.return_value = False  # Config is not set

        main(manual_args=mock_args)

        # Config setup should be called before run analysis
        mock_setup_config.assert_called_once_with(None)
        mock_run_analysis.assert_called_once()

    @patch("mvesuvio.main._run_bootstrap")
    @patch("mvesuvio.main._setup_config")
    def test_main_with_bootstrap_command(self, mock_setup_config, mock_run_bootstrap):
        from mvesuvio.main import main

        mock_args = MagicMock(command="bootstrap")

        main(manual_args=mock_args)

        mock_setup_config.assert_called_once_with(None)
        mock_run_bootstrap.assert_called_once_with(mock_args)


class TestRunAnalysis(unittest.TestCase):
    """Test cases for _run_analysis function."""

    @patch("mvesuvio.main.runpy.run_path")
    def test_run_analysis_with_no_args(self, mock_run_path):
        from mvesuvio.main import _run_analysis

        _run_analysis(None)

        self.assertEqual(mock_run_path.call_count, 2)
        first_call_path = mock_run_path.call_args_list[0].args[0]
        second_call_path = mock_run_path.call_args_list[1].args[0]
        self.assertEqual(Path(first_call_path).name, "run_reduction.py")
        self.assertEqual(Path(second_call_path).name, "run_fitting.py")
        self.assertEqual(mock_run_path.call_args_list[0].kwargs["run_name"], "__main__")
        self.assertEqual(mock_run_path.call_args_list[1].kwargs["run_name"], "__main__")

    @patch("mvesuvio.main.runpy.run_path")
    def test_run_analysis_with_all_args(self, mock_run_path):
        from mvesuvio.main import _run_analysis

        mock_args = MagicMock()

        _run_analysis(mock_args)

        self.assertEqual(mock_run_path.call_count, 2)
        first_call_path = mock_run_path.call_args_list[0].args[0]
        second_call_path = mock_run_path.call_args_list[1].args[0]
        self.assertEqual(Path(first_call_path).name, "run_reduction.py")
        self.assertEqual(Path(second_call_path).name, "run_fitting.py")
        self.assertEqual(mock_run_path.call_args_list[0].kwargs["run_name"], "__main__")
        self.assertEqual(mock_run_path.call_args_list[1].kwargs["run_name"], "__main__")


class TestRunBootstrap(unittest.TestCase):
    """Test cases for _run_bootstrap function."""

    @patch("mvesuvio.main.runpy.run_path")
    def test_run_bootstrap_with_no_args(self, mock_run_path):
        from mvesuvio.main import _run_bootstrap

        _run_bootstrap(None)

        mock_run_path.assert_called_once()
        call_path = mock_run_path.call_args.args[0]
        self.assertEqual(Path(call_path).name, "run_bootstrap.py")
        self.assertEqual(mock_run_path.call_args.kwargs["run_name"], "__main__")

    @patch("mvesuvio.main.runpy.run_path")
    def test_run_bootstrap_with_inputs_dir(self, mock_run_path):
        from mvesuvio.main import _run_bootstrap

        mock_args = MagicMock()

        _run_bootstrap(mock_args)

        mock_run_path.assert_called_once()
        call_path = mock_run_path.call_args.args[0]
        self.assertEqual(Path(call_path).name, "run_bootstrap.py")
        self.assertEqual(mock_run_path.call_args.kwargs["run_name"], "__main__")


if __name__ == "__main__":
    unittest.main()

