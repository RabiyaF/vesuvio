import unittest
from unittest.mock import patch

import mvesuvio


class TestPackageAPI(unittest.TestCase):
    @patch("mvesuvio.main")
    def test_config(self, mock_main):

        mvesuvio.config(experiment_dir="mock_file", ip_dir="mock_ipfolder")

        args, _ = mock_main.call_args
        self.assertEqual(args[0].experiment_dir, "mock_file")
        self.assertEqual(args[0].ip_dir, "mock_ipfolder")

    @patch("mvesuvio.main")
    def test_run(self, mock_main):

        mvesuvio.run()

        args, _ = mock_main.call_args
        self.assertEqual(args[0].command, "run")
