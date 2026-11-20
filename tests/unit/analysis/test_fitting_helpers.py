import unittest
from pathlib import Path

import numpy as np
from mantid.simpleapi import CreateWorkspace, GroupWorkspaces, Load, RenameWorkspace

from mvesuvio.util import fitting_helpers

np.set_printoptions(suppress=True, precision=6, linewidth=200)


class TestFittingHelpers(unittest.TestCase):
    def test_isolate_lighest_mass_data_no_fse_subtraction(self):
        np.random.seed(0)

        data_x = np.linspace(0, 10, 100).reshape(1, -1)
        data_e = np.full_like(data_x, 0.1)
        ws_ncp1 = CreateWorkspace(
            DataX=data_x, DataY=np.exp(-((data_x - 3) ** 2)), DataE=data_e, NSpec=1, UnitX="some_unit", OutputWorkspace="_1_ncp"
        )
        ws_ncp2 = CreateWorkspace(
            DataX=data_x, DataY=np.exp(-((data_x - 7) ** 2)), DataE=data_e, NSpec=1, UnitX="some_unit", OutputWorkspace="_2_ncp"
        )
        ws_ncp3 = CreateWorkspace(
            DataX=data_x, DataY=np.exp(-((data_x - 8) ** 2)), DataE=data_e, NSpec=1, UnitX="some_unit", OutputWorkspace="_3_ncp"
        )
        ws_total = ws_ncp1 + ws_ncp2 + ws_ncp3
        RenameWorkspace(ws_total, "_total_ncp")
        data = ws_total.extractY() + (np.random.random(100) - 0.5) * 0.5
        data[0, :10] = 0
        ws_data = CreateWorkspace(DataX=data_x, DataY=data, DataE=1.5 * data_e, NSpec=1, UnitX="some_unit")
        ncp_group = GroupWorkspaces([ws_ncp3, ws_ncp2, ws_ncp1, ws_total])

        ws_res, ws_res_ncp = fitting_helpers.isolate_lighest_mass_data(ws_data, ncp_group, False)

        ws_expected = ws_data - ws_ncp2 - ws_ncp3
        ws_expected.dataY(0)[:10] = 0

        np.testing.assert_allclose(ws_res.extractY(), ws_expected.extractY())
        np.testing.assert_allclose(ws_res.extractE(), ws_data.extractE())
        np.testing.assert_allclose(ws_res.extractX(), ws_data.extractX())
        np.testing.assert_allclose(ws_res_ncp.extractY(), ws_ncp1.extractY())
        np.testing.assert_allclose(ws_res_ncp.extractE(), ws_ncp1.extractE())
        np.testing.assert_allclose(ws_res_ncp.extractX(), ws_ncp1.extractX())

    def test_isolate_lighest_mass_data_with_fse_subtraction(self):
        np.random.seed(0)

        dataX = np.linspace(0, 10, 100).reshape(1, -1)
        dataE = np.full_like(dataX, 0.1)
        ws_ncp1 = CreateWorkspace(
            DataX=dataX, DataY=np.exp(-((dataX - 3) ** 2)), DataE=dataE, NSpec=1, UnitX="some_unit", OutputWorkspace="_1_ncp"
        )
        ws_ncp2 = CreateWorkspace(
            DataX=dataX, DataY=np.exp(-((dataX - 7) ** 2)), DataE=dataE, NSpec=1, UnitX="some_unit", OutputWorkspace="_2_ncp"
        )
        ws_ncp3 = CreateWorkspace(
            DataX=dataX, DataY=np.exp(-((dataX - 8) ** 2)), DataE=dataE, NSpec=1, UnitX="some_unit", OutputWorkspace="_3_ncp"
        )
        ws_total = ws_ncp1 + ws_ncp2 + ws_ncp3
        RenameWorkspace(ws_total, "_total_ncp")
        data = ws_total.extractY() + (np.random.random(100) - 0.5) * 0.5
        ws_data = CreateWorkspace(DataX=dataX, DataY=data, DataE=1.5 * dataE, NSpec=1, UnitX="some_unit")
        ncp_group = GroupWorkspaces([ws_ncp3, ws_ncp2, ws_ncp1, ws_total])
        ws_fse1 = CreateWorkspace(
            DataX=dataX,
            DataY=0.4 * np.sin(dataX) * np.exp(-((dataX - 3) ** 2)),
            DataE=0.5 * dataE,
            NSpec=1,
            UnitX="some_unit",
            OutputWorkspace="_1_fse",
        )

        ws_res, ws_res_ncp = fitting_helpers.isolate_lighest_mass_data(ws_data, ncp_group, True)

        ws_expected = ws_data - ws_ncp2 - ws_ncp3 - ws_fse1
        np.testing.assert_allclose(ws_res.extractY(), ws_expected.extractY())
        np.testing.assert_allclose(ws_res.extractE(), ws_data.extractE())
        np.testing.assert_allclose(ws_res.extractX(), ws_data.extractX())
        np.testing.assert_allclose(ws_res_ncp.extractY(), ws_ncp1.extractY() - ws_fse1.extractY())
        np.testing.assert_allclose(ws_res_ncp.extractE(), ws_ncp1.extractE())
        np.testing.assert_allclose(ws_res_ncp.extractX(), ws_ncp1.extractX())

    def test_vesuvio_resolution(self):
        ws_data = Load(str(Path(__file__).parent.parent.parent / "data/analysis/unit/analysis_fwd_0.nxs"))

        ws_res = fitting_helpers.calculate_resolution(1, ws_data, "-25, 5, 25")

        np.set_printoptions(precision=3)
        np.testing.assert_allclose(
            ws_res.dataY(0)[:],
            np.array([6.140e-05, 1.028e-04, 2.073e-04, 6.221e-04, 9.877e-02, 9.876e-02, 6.218e-04, 2.074e-04, 8.361e-05, 0.000e00]),
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            ws_res.dataY(15)[:],
            np.array([8.188e-05, 1.555e-04, 3.137e-04, 9.517e-04, 9.811e-02, 9.812e-02, 9.517e-04, 3.137e-04, 1.555e-04, 9.288e-05]),
            rtol=1e-3,
        )
        np.testing.assert_allclose(ws_res.dataY(30)[:], np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
