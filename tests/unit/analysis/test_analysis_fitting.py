import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
from mantid.simpleapi import AnalysisDataService, CreateWorkspace

from mvesuvio.analysis_fitting import plot_global_fit

np.set_printoptions(suppress=True, precision=6, linewidth=200)


class TestAnalysisFitting(unittest.TestCase):
    def test_plot_global_fit(self):

        def gaussian(x, amplitude, mean, std):
            return amplitude * np.exp(-((x - mean) ** 2) / (2 * std**2))

        np.random.seed(4)
        x_data = np.linspace(0, 10, 100)
        true_params = [5.0, 5.0, 1.0]  # amplitude, mean, std
        y_fit = gaussian(x_data, *true_params)

        ws_prefix = "fake_ws_global_fit_"
        N = 50
        y_fit_sum = np.zeros_like(x_data)
        y_noisy_sum = np.zeros_like(x_data)
        for i in range(N):
            y_err = np.random.normal(0, 0.3, len(x_data))
            y_noisy = y_fit + y_err
            CreateWorkspace(
                DataX=np.concatenate([x_data, x_data, x_data]),
                DataY=np.concatenate([y_noisy, y_fit, y_fit - y_noisy]),
                DataE=np.concatenate([y_err, np.zeros_like(y_err), np.zeros_like(y_err)]),
                Nspec=3,
                OutputWorkspace=ws_prefix + str(i),
                Distribution=True,
            )
            y_fit_sum += y_fit
            y_noisy_sum += y_noisy

        CreateWorkspace(
            DataX=np.concatenate([x_data, x_data, x_data]),
            DataY=np.concatenate([y_noisy_sum, y_fit_sum, y_fit_sum - y_noisy_sum]),
            DataE=np.concatenate([np.random.normal(0, 0.5, len(x_data)), np.zeros_like(x_data), np.zeros_like(x_data)]),
            Nspec=3,
            OutputWorkspace=ws_prefix + "sum",
            Distribution=True,
        )

        mock_ws_group_name = "mock_global_fit_ws"
        mock_ws_group = MagicMock(getNames=MagicMock(return_value=[ws_prefix + "sum", *[ws_prefix + str(i) for i in range(N)]]))
        mock_ws_group.configure_mock(**{"name.return_value": mock_ws_group_name})

        with tempfile.TemporaryDirectory() as tmpdir:
            yfit_class = MagicMock()
            yfit_class.figSavePath = Path(tmpdir)

            fig = plot_global_fit(mock_ws_group, yfit_class)

            # Check that all curves made it to the plot
            self.assertEqual(len(fig.axes[0].lines), 2 * N)
            self.assertEqual(len(fig.axes[1].lines), 3)

            self.assertTrue((yfit_class.figSavePath / (mock_ws_group_name + "_results.png")).exists())

        AnalysisDataService.clear()
        plt.close(fig)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
