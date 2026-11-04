import unittest
import runpy
import numpy as np
from mock import MagicMock, Mock, patch, call
from mvesuvio.analysis_reduction import VesuvioAnalysisRoutine
from mvesuvio.globals import ModeRunning, ScatteringType
from mvesuvio.util.analysis_helpers import load_resolution
from mvesuvio.util import handle_config
from mantid.simpleapi import CreateWorkspace, CreateSampleWorkspace, CreateEmptyTableWorkspace
from mantid.simpleapi import Load, mtd, CompareWorkspaces
import dill         # To convert constraints to string
import scipy
from pathlib import Path

np.set_printoptions(suppress=True, precision=6, linewidth=200)
np.random.seed(4)


class TestAnalysisReduction(unittest.TestCase):

    def test_import_continues_when_plot_style_is_unavailable(self):
        with patch("matplotlib.pyplot.style.use", side_effect=OSError("style unavailable")) as style_use:
            runpy.run_module("mvesuvio.util.reduction_helpers", run_name="reduction_helpers_import_test")

        style_use.assert_called_once()

    def test_summary_category_and_pyexec(self):
        alg = VesuvioAnalysisRoutine()
        alg._setup = Mock()
        alg.run = Mock()

        self.assertEqual(alg.summary(), "Runs the analysis reduction routine for VESUVIO.")
        self.assertEqual(alg.category(), "VesuvioAnalysis")

        alg.PyExec()

        alg._setup.assert_called_once_with()
        alg.run.assert_called_once_with()


    def test_properites_vesuvio_analysis_algorithm(self):

        kwargs = {
            "InputWorkspace": CreateSampleWorkspace(OutputWorkspace="input-ws").name(),
            "InputProfiles": CreateEmptyTableWorkspace(OutputWorkspace="profiles-table").name(),
            "InstrumentParametersFile": str(Path(handle_config.PACKAGE_CONFIG_PATH) / "ip_files" / "ip2018_3.par"),
            "ChosenMassIndex": 0,
            "HRatioToChosenMass": 0,
            "NumberOfIterations": 4,
            "InvalidDetectors": [3],
            "MultipleScatteringCorrection": False,
            "GammaCorrection": True,
            "ModeRunning": ModeRunning.BACKWARD,
            "TransmissionGuess": 0,
            "MultipleScatteringOrder": 2,
            "NumberOfEvents": 2,
            "Constraints": "()",
            "OutputMeansTable": "output-table",
        }
        alg = VesuvioAnalysisRoutine()
        alg.PyInit()
        alg.setProperties(kwargs)

        for prop in kwargs.keys():
            # Only check is that it should not raise any errors
            alg.getPropertyValue(prop)


    @patch('mvesuvio.analysis_reduction.VesuvioAnalysisRoutine.getPropertyValue')
    @patch('mvesuvio.analysis_reduction.load_instrument_params')
    @patch('mvesuvio.analysis_reduction.load_resolution')
    def test_setup_vesuvio_analysis_algorithm(self, mock_load_resolution, mock_load_instrument_params, mock_get_prop_value):

        mock_load_instrument_params.return_value = None

        # Tests only unpacking of profiles and bounds

        table_dict = {
            "mass" : [1, 12, 16],
            "intensity" : [1.0, 1.1, 1.2],
            "intensity_lb" : [1, 1, 1],
            "intensity_ub" : [None, np.nan, None],
            "width" : [5, 10, 12],
            "width_lb" : [2, 3, 4],
            "width_ub": [6, 7, 8],
            "center" : [0.0, 0.1, 0.2],
            "center_lb" : [-1, -1, -1],
            "center_ub" : [3.0, 3.0, 3.0]
        }
        mock_table_profiles = MagicMock()
        mock_table_profiles.column.side_effect = lambda key: table_dict[key]


        with patch('mvesuvio.analysis_reduction.VesuvioAnalysisRoutine.getProperty') as mock_get_prop:

            def mock_properties(key):
                if key == "InputProfiles":
                    return MagicMock(value=mock_table_profiles)
                if key == "InputWorkspace":
                    return MagicMock(value=MagicMock(getSpectrumNumbers=MagicMock(return_value=3)))
                if key == "InstrumentParametersFile":
                    return MagicMock(value="")
                if key == "Constraints":
                    return MagicMock(value=str(dill.dumps(({'type': 'eq', 'fun': lambda par:  par[0] - 2.7527*par[3] }, {'type': 'eq', 'fun': lambda par:  par[0] - 2.7527*par[3] }))))
                if key == "ChosenMassIndex":
                    return MagicMock(value=0)
                else:
                    return MagicMock(value=None)

            mock_get_prop.side_effect = mock_properties

            alg = VesuvioAnalysisRoutine()
            alg._setup()

            self.assertEqual(alg._initial_fit_parameters, [1.0, 5, 0.0, 1.1, 10, 0.1, 1.2, 12, 0.2])
            self.assertEqual(alg._initial_fit_bounds, [[1, None], [2, 6], [-1, 3.0], [1, np.nan], [3, 7], [-1, 3.0], [1, None], [4, 8], [-1, 3.0]])
            self.assertEqual(list(alg._constraints[0].keys()), ['type', 'fun'])   # Difficult to test actual constraint because function created lives inside alg
            np.testing.assert_allclose(alg._masses, np.array([1, 12, 16]))


    def test_update_workspace_data(self):

        alg = VesuvioAnalysisRoutine()

        # Not possible to mock because required for ParentWorkspace arg
        alg._workspace_being_fit = CreateWorkspace(
            DataX=np.arange(10),
            DataY=np.arange(10),
            DataE=np.ones(10),
            Nspec=1,
            OutputWorkspace="test_ws",
        )

        label = ['1.01', '5', '12']
        alg._masses = np.array([1.01, 5, 12])
        alg._profiles_table = MagicMock(
            rowCount=MagicMock(return_value=3),
            column=MagicMock(side_effect=lambda key: label if key=="label" else None)
        )

        alg._set_kinematic_arrays = MagicMock()
        alg._set_gaussian_resolution = MagicMock()
        alg._set_lorentzian_resolution = MagicMock()
        alg._set_y_space_arrays = MagicMock()
        alg.setPropertyValue = MagicMock()

        alg._update_workspace_data()

        self.assertEqual(alg._fit_parameters.sum(), 0)
        self.assertEqual(alg._row_being_fit, 0)
        self.assertEqual(alg._table_fit_results.rowCount(), 0)

        for lab in label:
            ws_ncp = alg._fit_profiles_workspaces[lab]
            ws_fse = alg._fit_fse_workspaces[lab]
            np.testing.assert_allclose(ws_ncp.dataX(0), np.arange(10))
            np.testing.assert_allclose(ws_ncp.dataY(0), np.zeros(10))
            np.testing.assert_allclose(ws_ncp.dataE(0), np.zeros(10))
            self.assertEqual(ws_ncp.isDistribution(), True)
            self.assertEqual(ws_ncp.getNumberHistograms(), 1)
            np.testing.assert_allclose(ws_fse.dataX(0), np.arange(10))
            np.testing.assert_allclose(ws_fse.dataY(0), np.zeros(10))
            np.testing.assert_allclose(ws_fse.dataE(0), np.zeros(10))
            self.assertEqual(ws_fse.isDistribution(), True)
            self.assertEqual(ws_fse.getNumberHistograms(), 1)


    def test_calculate_kinematics(self):
        alg = VesuvioAnalysisRoutine()
        alg._instrument_params = np.array(
            [[3, 3, 131.12, -0.2, 11.005, 0.6039],
            [4, 4, 132.77, -0.2, 11.005, 0.5789],
            [5, 5, 133.69, -0.2, 11.005, 0.5696],
            ])
        dataX = np.array([
            [110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5],
            [110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5],
            [110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5],
        ])
        alg._set_kinematic_arrays(dataX)

        # Kinematic arrays are calculated from exteded range for dataX

        np.testing.assert_allclose(alg._v0, np.array(
            [[0.12949, 0.127984, 0.126513, 0.125075, 0.12367, 0.122295, 0.120951, 0.119636, 0.11835, 0.117091, 0.115858, 0.114651, 0.113469, 0.112311, 0.111176, 0.110064, 0.108974, 0.107906, 0.106858],
            [0.128259, 0.126781, 0.125337, 0.123926, 0.122546, 0.121196, 0.119876, 0.118584, 0.11732, 0.116083, 0.114871, 0.113684, 0.112522, 0.111383, 0.110267, 0.109173, 0.108101, 0.107049, 0.106018],
            [0.127807, 0.126339, 0.124905, 0.123504, 0.122133, 0.120792, 0.119481, 0.118198, 0.116942, 0.115712, 0.114508, 0.113329, 0.112174, 0.111042, 0.109933, 0.108845, 0.107779, 0.106734, 0.105709]]
        ), atol=1e-6)
        np.testing.assert_allclose(alg._E0, np.array(
            [[87655.042194, 85628.1004, 83670.660984, 81779.582305, 79951.898249, 78184.806586, 76475.658228, 74821.947293, 73221.30191, 71671.475719, 70170.339977, 68715.876247, 67306.169611, 65939.402362, 64613.848144, 63327.866492, 62079.897763, 60868.458397, 59692.136511],
            [85995.604273, 84025.62244, 82122.565654, 80283.436416, 78505.403187, 76785.789479, 75122.063781, 73511.830231, 71952.819995, 70442.883259, 68979.981824, 67562.182219, 66187.649309, 64854.640355, 63561.499488, 62306.652564, 61088.602368, 59905.924152, 58757.261461],
            [85390.300476, 83440.962859, 81557.622084, 79737.332238, 77977.309967, 76274.923823, 74627.684428, 73033.235365, 71489.344748, 69993.897408, 68544.887641, 67140.412484, 65778.665459, 64457.930766, 63176.577872, 61933.056479, 60725.891829, 59553.680333, 58415.085488]]
        ), atol=1e-6)
        np.testing.assert_allclose(alg._deltaQ, np.array(
            [[240.409925, 238.046957, 235.738868, 233.483784, 231.279914, 229.12555, 227.019055, 224.958865, 222.943485, 220.97148, 219.041479, 217.152166, 215.30228, 213.490611, 211.715999, 209.97733, 208.273532, 206.603579, 204.966482],
            [239.365494, 237.045827, 234.779507, 232.564729, 230.399768, 228.282976, 226.212776, 224.18766, 222.206182, 220.26696, 218.368667, 216.510034, 214.689841, 212.906918, 211.160143, 209.448438, 207.770767, 206.126134, 204.513584],
            [239.138969, 236.834796, 234.583415, 232.383047, 230.23199, 228.128619, 226.071376, 224.058774, 222.089387, 220.16185, 218.274853, 216.427142, 214.617513, 212.844811, 211.107927, 209.405797, 207.737397, 206.101744, 204.497891]]
        ), atol=1e-6)
        np.testing.assert_allclose(alg._deltaE, np.array(
            [[82749.042194, 80722.1004, 78764.660984, 76873.582305, 75045.898249, 73278.806586, 71569.658228, 69915.947293, 68315.30191, 66765.475719, 65264.339977, 63809.876247, 62400.169611, 61033.402362, 59707.848144, 58421.866492, 57173.897763, 55962.458397, 54786.136511],
            [81089.604273, 79119.62244, 77216.565654, 75377.436416, 73599.403187, 71879.789479, 70216.063781, 68605.830231, 67046.819995, 65536.883259, 64073.981824, 62656.182219, 61281.649309, 59948.640355, 58655.499488, 57400.652564, 56182.602368, 54999.924152, 53851.261461],
            [80484.300476, 78534.962859, 76651.622084, 74831.332238, 73071.309967, 71368.923823, 69721.684428, 68127.235365, 66583.344748, 65087.897408, 63638.887641, 62234.412484, 60872.665459, 59551.930766, 58270.577872, 57027.056479, 55819.891829, 54647.680333, 53509.085488]]
        ), atol=1e-6)


    def test_set_y_space_arrays(self):
        alg = VesuvioAnalysisRoutine()
        alg._masses = np.array([1, 12, 16])
        alg._deltaQ = np.array([[227.01905, 224.95887, 222.94348, 220.97148, 219.04148],
           [226.21278, 224.18766, 222.20618, 220.26696, 218.36867],
           [226.07138, 224.05877, 222.08939, 220.16185, 218.27485],
        ])
        alg._deltaE = np.array([[71569.65823, 69915.94729, 68315.30191, 66765.47572, 65264.33998],
           [70216.06378, 68605.83023, 67046.81999, 65536.88326, 64073.98182],
           [69721.68443, 68127.23536, 66583.34475, 65087.89741, 63638.88764],
        ])
        alg._set_y_space_arrays()
        y_spaces_expected = np.array(
            [[[-38.0885,-38.12637,-38.16414,-38.20185,-38.23948],
            [791.54274,779.75738,768.21943,756.92089,745.85436],
            [1093.22682,1077.16966,1061.44982,1046.05644,1030.97939]],

            [[-38.84807,-38.88304,-38.91795,-38.95279,-38.98756],
            [777.99344,766.43564,755.11863,744.0348,733.17695],
            [1075.02671,1059.2788,1043.8592,1028.75757,1013.96405]],

            [[-39.25409,-39.28749,-39.32085,-39.35414,-39.38735],
            [772.34348,760.87331,749.64145,738.64055,727.86342],
            [1067.46987,1051.84087,1036.53683,1021.54771,1006.8637,]]])
        np.testing.assert_allclose(alg._y_space_arrays, y_spaces_expected, atol=1e-4)


    def test_fit_neutron_compton_profiles_number_of_calls(self):
        alg = VesuvioAnalysisRoutine()
        alg._dataY = np.array([[1, 1], [2, 2], [3, 3]])
        alg._fit_parameters = np.ones(3)   # To avoid assertion error
        alg._fit_neutron_compton_profiles_to_row = MagicMock(return_value=None)
        alg._table_fit_results = MagicMock(return_value=None)
        alg._fit_neutron_compton_profiles()
        self.assertEqual(alg._fit_neutron_compton_profiles_to_row.call_count, 3)


    def test_neutron_compton_profile_fit_function(self):
        alg = VesuvioAnalysisRoutine()
        alg._dataX = np.arange(113, 430, 6).reshape(1, -1)
        alg._instrument_params = np.array([
            [144, 144, 54.1686, -0.41, 11.005, 0.720184]
            ])
        alg._resolution_params = load_resolution(alg._instrument_params)
        alg._masses = np.array([1, 12, 16, 27])
        alg._set_kinematic_arrays(alg._dataX)
        alg._set_gaussian_resolution()
        alg._set_lorentzian_resolution()
        alg._set_y_space_arrays()
        example_fit_parameters = np.array([7.1, 5.05, 0.02, 0.22, 12.71, 1.0, 0.0, 8.76, -1.1, 0.3, 13.897, 0.64])
        alg._row_being_fit = 0
        NCP, FSE = alg._neutron_compton_profiles(example_fit_parameters)
        expected_NCP = np.array([[ 0.000054, 0.000082, 0.000125, 0.00019 , 0.00029 , 0.000436, 0.000646, 0.000942, 0.001348, 0.00189 , 0.002596, 0.003492, 0.004598, 0.005924, 0.007466, 0.009192, 0.011039, 0.012912, 0.01468 ,
               0.016188, 0.017262, 0.017739, 0.017519, 0.016596, 0.015063, 0.013083, 0.010857, 0.008593, 0.006469, 0.004616, 0.003105, 0.00195 , 0.001125, 0.000575, 0.000236, 0.000048,-0.000042,-0.000072,
              -0.000071,-0.000056,-0.000037,-0.00002 ,-0.000006, 0.000005, 0.000012, 0.000017, 0.00002 , 0.000022, 0.000022, 0.000023, 0.000022, 0.000022, 0.000021],
             [ 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      ,
               0.      , 0.      , 0.      , 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000002, 0.000002, 0.000002, 0.000003, 0.000003, 0.000004, 0.000006, 0.000008, 0.00001 , 0.000012,
               0.000043, 0.000145, 0.000338, 0.000928, 0.002234, 0.003137, 0.002397, 0.000928, 0.000073,-0.000059, 0.000001, 0.000021, 0.000016, 0.000012, 0.00001 ],
             [ 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      ,
               0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      ,
               0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      ],
             [ 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      ,
               0.      , 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000002, 0.000002, 0.000002, 0.000003, 0.000003, 0.000004, 0.000005, 0.000006, 0.000008, 0.00001 , 0.000014,
               0.00002 , 0.000026, 0.000056, 0.000149, 0.000448, 0.003172, 0.006625, 0.003266, 0.000521, 0.000004, 0.000047, 0.000051, 0.00003 , 0.000023, 0.000019]])

        np.testing.assert_allclose(NCP, expected_NCP, atol=1e-6)


    def test_error_function(self):
        alg = VesuvioAnalysisRoutine()
        alg._dataX = np.arange(113, 430, 6).reshape(1, -1)
        alg._instrument_params = np.array([
            [144, 144, 54.1686, -0.41, 11.005, 0.720184]
            ])
        alg._resolution_params = load_resolution(alg._instrument_params)
        alg._masses = np.array([1, 12, 16, 27])
        alg._set_kinematic_arrays(alg._dataX)
        alg._set_gaussian_resolution()
        alg._set_lorentzian_resolution()
        alg._set_y_space_arrays()
        example_fit_parameters = np.array([7.1, 5.05, 0.02, 0.22, 12.71, 1.0, 0.0, 8.76, -1.1, 0.3, 13.897, 0.64])
        alg._row_being_fit = 0

        NCP, FSE = alg._neutron_compton_profiles(example_fit_parameters)
        alg._dataY = NCP.sum(axis=0) + 0.002*(np.random.random_sample(alg._dataX.shape)-0.5)

        alg._dataE = np.zeros_like(alg._dataX)
        chi2_without_errors = alg._error_function(example_fit_parameters)
        self.assertEqual(chi2_without_errors,
                         np.sum((alg._dataY - np.sum(NCP, axis=0))**2))

        alg._dataE = np.full_like(alg._dataX, 0.0015, dtype=np.double)
        chi2_with_errors = alg._error_function(example_fit_parameters)
        self.assertEqual(chi2_with_errors,
                         np.sum((alg._dataY - np.sum(NCP, axis=0))**2 / alg._dataE**2))

        alg._dataY[0, :20] = 0
        chi2_with_errors_with_tof_masked = alg._error_function(example_fit_parameters)
        self.assertEqual(chi2_with_errors_with_tof_masked,
                         np.sum((alg._dataY[0, 20:] - np.sum(NCP, axis=0)[20:])**2 / alg._dataE[0, 20:]**2))

        self.assertAlmostEqual(chi2_without_errors, chi2_with_errors * 0.0015**2, places=15)


    def test_fit_neutron_compton_profiles_to_row(self):
        alg = VesuvioAnalysisRoutine()
        alg._instrument_params = np.array([
            [144, 144, 54.1686, -0.41, 11.005, 0.720184],
            [145, 145, 52.3407, -0.53, 11.005, 0.717311],
            [146, 146, 50.7811, -1.14, 11.005, 0.742482]
            ])
        alg._resolution_params = load_resolution(alg._instrument_params)
        alg._masses = np.array([1, 12, 16, 27])

        alg._dataX = np.array([
            [113., 119., 125., 131., 137., 143., 149., 155., 161., 167., 173., 179., 185., 191., 197., 203., 209., 215., 221., 227., 233., 239., 245., 251., 257., 263., 269., 275., 281., 287., 293., 299., 305., 311., 317., 323., 329., 335., 341., 347., 353., 359., 365., 371., 377., 383., 389., 395., 401., 407., 413., 419., 425., 429.],
            [113., 119., 125., 131., 137., 143., 149., 155., 161., 167., 173., 179., 185., 191., 197., 203., 209., 215., 221., 227., 233., 239., 245., 251., 257., 263., 269., 275., 281., 287., 293., 299., 305., 311., 317., 323., 329., 335., 341., 347., 353., 359., 365., 371., 377., 383., 389., 395., 401., 407., 413., 419., 425., 429.],
            [113., 119., 125., 131., 137., 143., 149., 155., 161., 167., 173., 179., 185., 191., 197., 203., 209., 215., 221., 227., 233., 239., 245., 251., 257., 263., 269., 275., 281., 287., 293., 299., 305., 311., 317., 323., 329., 335., 341., 347., 353., 359., 365., 371., 377., 383., 389., 395., 401., 407., 413., 419., 425., 429.],
        ])
        alg._set_kinematic_arrays(alg._dataX)
        alg._set_gaussian_resolution()
        alg._set_lorentzian_resolution()
        alg._set_y_space_arrays()

        synthetic_ncp = np.zeros_like(alg._dataX)
        example_fit_parameters = np.array([
            [7.1, 5.05, 0.02, 0.22, 12.71, 1.0, 0.0, 8.76, -1.1, 0.3, 13.897, 0.64],
            [5, 4, -1, 0.5, 12, -2, 0.8, 8., -1.1, 0.6, 14., 0.5]
        ])
        for i, pars in enumerate(example_fit_parameters):
            alg._row_being_fit = i
            NCP, FSE = alg._neutron_compton_profiles(pars)
            synthetic_ncp[i] = NCP.sum(axis=0)

        alg._dataY = synthetic_ncp.copy()
        # Add noise (except for third row that is masked)
        alg._dataY[:2, :] += 0.002*(np.random.random_sample((2, synthetic_ncp.shape[-1]))-0.5)
        # Mask tof range on second row
        alg._dataY[1, 5:15] = np.zeros(10)
        alg._dataE = np.full_like(alg._dataX, 0.0015, dtype=np.double)

        alg._initial_fit_parameters = np.array([1, 5.0, 0.1, 1, 12.0, 0.1, 1, 8.0, 0.1, 1, 13.0, 0.1])
        alg._initial_fit_bounds = np.array([
            [0, None], [3, 6], [-3, 1],
            [0, None], [9, 13], [-3, 1],
            [0, None], [6, 11], [-3, 1],
            [0, None], [11, 15], [-3, 1],
        ])
        alg._constraints = ()

        # Create arrays for storing results
        ncp_total_array = np.zeros_like(alg._dataY)
        fse_total_array = np.zeros_like(alg._dataY)
        alg._fit_profiles_workspaces = {
            "total": MagicMock(dataY=lambda row: ncp_total_array[row]),
            "1": MagicMock(), "12": MagicMock(), "16": MagicMock(), "27": MagicMock()
        }
        alg._fit_fse_workspaces = {
            "total": MagicMock(dataY=lambda row: fse_total_array[row]),
            "1": MagicMock(), "12": MagicMock(), "16": MagicMock(), "27": MagicMock()
        }
        # Mock methods that are not used
        alg._table_fit_results = MagicMock()
        alg._profiles_table = MagicMock()
        alg._fit_parameters = MagicMock()

        # Fit
        alg._row_being_fit = 0
        alg._fit_neutron_compton_profiles_to_row()
        alg._row_being_fit = 1
        alg._fit_neutron_compton_profiles_to_row()
        alg._row_being_fit = 2
        alg._fit_neutron_compton_profiles_to_row()

        np.testing.assert_allclose(synthetic_ncp, ncp_total_array, atol=1e-3)
        # Check masked row is correctly ignored
        np.testing.assert_allclose(synthetic_ncp[-1], np.zeros(synthetic_ncp.shape[-1]))


    def test_fit_neutron_compton_profiles_to_row_with_masked_tof(self):
        alg = VesuvioAnalysisRoutine()
        alg._masses = np.array([1])
        alg._instrument_params = np.array([[144, 144, 54.1686, -0.41, 11.005, 0.720184]])
        alg._resolution_params = load_resolution(alg._instrument_params)
        alg._initial_fit_parameters = np.array([1, 5.2, 0])
        alg._initial_fit_bounds = np.array([[0, None], [3, 6], [None, None]])
        alg._constraints = ()
        alg._table_fit_results = MagicMock()
        alg._profiles_table = MagicMock()
        alg._fit_parameters = MagicMock()

        alg._dataX = np.arange(113, 430).reshape(1, -1)
        alg._dataE = np.full_like(alg._dataX, 0.0015, dtype=np.double)
        alg._dataY = scipy.special.voigt_profile(alg._dataX - 235, 30, 0) + 0.005*(np.random.random_sample(alg._dataX.shape)-0.5)
        # Mask TOF range
        cut_off_idx = 100
        alg._dataY[:, :cut_off_idx] = 0
        alg._set_kinematic_arrays(alg._dataX)
        alg._set_gaussian_resolution()
        alg._set_lorentzian_resolution()
        alg._set_y_space_arrays()

        # Create mock for storing ncp total result
        ncp_array_masked = np.zeros_like(alg._dataY)
        alg._fit_profiles_workspaces = {"total": MagicMock(dataY=lambda arg: ncp_array_masked), "1": MagicMock()}
        alg._fit_fse_workspaces = {"total": MagicMock(), "1": MagicMock()}

        # Fit ncp
        alg._row_being_fit = 0
        alg._fit_neutron_compton_profiles_to_row()
        fit_parameters_masked = alg._fit_parameters.copy()

        # Now cut range so that zeros are not part of dataY
        alg._dataY = alg._dataY[:, cut_off_idx:].reshape(1, -1)
        alg._dataX = alg._dataX[:, cut_off_idx:].reshape(1, -1)
        alg._dataE = alg._dataE[:, cut_off_idx:].reshape(1, -1)

        # Run methods that depend on dataX
        alg._set_kinematic_arrays(alg._dataX)
        alg._set_gaussian_resolution()
        alg._set_lorentzian_resolution()
        alg._set_y_space_arrays()

        ncp_array_cut = ncp_array_masked[:, cut_off_idx:].reshape(1, -1)
        alg._fit_profiles_workspaces = {"total": MagicMock(dataY=lambda arg: ncp_array_cut), "1": MagicMock()}

        alg._fit_neutron_compton_profiles_to_row()
        fit_parameters_cut = alg._fit_parameters.copy()

        np.testing.assert_allclose(fit_parameters_cut[:, 1:-2], fit_parameters_masked[:, 1:-2], atol=1e-6)
        np.testing.assert_allclose(ncp_array_masked[:, cut_off_idx:], ncp_array_cut, atol=1e-6)


    @patch("mvesuvio.analysis_reduction.scipy.optimize.minimize")
    def test_fit_neutron_compton_profiles_to_row_passes_constraints_to_minimize(self, mock_minimize):
        alg = VesuvioAnalysisRoutine()

        constraints = ({"type": "eq", "fun": lambda par: par[0] - 1.0},)
        alg._constraints = constraints
        alg._initial_fit_parameters = [1.0, 5.0, 0.0]
        alg._initial_fit_bounds = [[0, None], [3, 6], [-1, 1]]
        alg._row_being_fit = 0
        alg._dataY = np.array([[1.0, 1.0, 1.0, 1.0]])
        alg._profiles_table = MagicMock(
            rowCount=MagicMock(return_value=1),
            column=MagicMock(return_value=["1"]),
        )
        alg._instrument_params = np.array([[144, 144, 54.1686, -0.41, 11.005, 0.720184]])

        alg._table_fit_results = MagicMock()
        alg._fit_parameters = np.zeros((1, 5))

        ncp_total = np.zeros(4)
        fse_total = np.zeros(4)
        alg._fit_profiles_workspaces = {
            "1": MagicMock(dataY=lambda row: np.zeros(4)),
            "total": MagicMock(dataY=lambda row: ncp_total),
        }
        alg._fit_fse_workspaces = {
            "1": MagicMock(dataY=lambda row: np.zeros(4)),
            "total": MagicMock(dataY=lambda row: fse_total),
        }

        alg._neutron_compton_profiles = MagicMock(return_value=(np.array([[0.1, 0.2, 0.3, 0.4]]), np.array([[0.0, 0.0, 0.0, 0.0]])))

        mock_minimize.return_value = {"x": np.array([1.2, 5.1, 0.1]), "fun": 0.1}

        alg._fit_neutron_compton_profiles_to_row()

        self.assertTrue(mock_minimize.called)
        self.assertIs(mock_minimize.call_args.kwargs["constraints"], constraints)


    def test_set_gaussian_resolution(self):

        alg = VesuvioAnalysisRoutine()
        alg._instrument_params = np.array([
            [144, 144, 54.1686, -0.41, 11.005, 0.720184],
            [145, 145, 52.3407, -0.53, 11.005, 0.717311],
            ])
        alg._resolution_params = load_resolution(alg._instrument_params)
        alg._masses = np.array([1, 12])

        alg._dataX = np.array([
            [110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5],
            [110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5],
        ])
        alg._set_kinematic_arrays(alg._dataX)
        alg._set_gaussian_resolution()
        expected_gaussian_resolution = np.array(
            [[[3.709857, 3.62982, 3.552662, 3.478251, 3.406461, 3.337172, 3.270274, 3.205661, 3.143232, 3.082894, 3.024556, 2.968136, 2.913551, 2.860728, 2.809592, 2.760078, 2.712118, 2.665653, 2.620622],
            [31.128982, 30.438141, 29.771605, 29.128243, 28.50699, 27.906841, 27.326849, 26.766118, 26.223805, 25.699109, 25.191275, 24.699587, 24.223369, 23.761977, 23.314804, 22.881272, 22.460833, 22.052966, 21.657176]],
            [[3.716323, 3.636666, 3.559864, 3.485785, 3.414305, 3.345307, 3.278679, 3.214318, 3.152124, 3.092005, 3.033872, 2.977641, 2.923233, 2.870573, 2.81959, 2.770216, 2.722386, 2.67604, 2.631121],
            [31.204563, 30.517381, 29.854298, 29.214196, 28.596021, 27.998779, 27.421535, 26.863402, 26.323544, 25.801171, 25.295536, 24.805929, 24.33168, 23.872154, 23.426748, 22.99489, 22.576037, 22.169674, 21.77531, ]]]
        )
        np.testing.assert_allclose(expected_gaussian_resolution, alg._gaussian_resolution, atol=1e-6)


    def test_set_lorentzian_resolution(self):
        alg = VesuvioAnalysisRoutine()
        alg._instrument_params = np.array([
            [144, 144, 54.1686, -0.41, 11.005, 0.720184],
            [145, 145, 52.3407, -0.53, 11.005, 0.717311],
            ])
        alg._resolution_params = load_resolution(alg._instrument_params)
        alg._masses = np.array([1, 12])

        alg._dataX = np.array([
            [110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5],
            [110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5],
        ])
        alg._set_kinematic_arrays(alg._dataX)
        alg._set_lorentzian_resolution()
        expected_lorentzian_resolution = np.array(
            [[[0.313762, 0.308099, 0.302633, 0.297353, 0.292252, 0.287321, 0.282552, 0.277939, 0.273474, 0.269152, 0.264966, 0.26091, 0.256979, 0.253168, 0.249472, 0.245887, 0.242408, 0.239031, 0.235751],
            [2.410374, 2.368446, 2.32816, 2.289439, 2.252211, 2.21641, 2.18197, 2.148833, 2.11694, 2.086238, 2.056676, 2.028205, 2.000781, 1.974359, 1.948898, 1.92436, 1.900708, 1.877906, 1.855921]],
            [[0.317283, 0.311673, 0.306255, 0.301023, 0.295966, 0.291078, 0.28635, 0.281775, 0.277348, 0.273061, 0.268909, 0.264886, 0.260986, 0.257205, 0.253538, 0.24998, 0.246526, 0.243174, 0.239918],
            [2.41149, 2.370058, 2.330248, 2.291987, 2.255202, 2.219827, 2.1858, 2.15306, 2.121553, 2.091224, 2.062023, 2.033904, 2.00682, 1.980729, 1.955591, 1.931366, 1.908019, 1.885515, 1.863821]]]
        )
        np.testing.assert_allclose(expected_lorentzian_resolution, alg._lorentzian_resolution, atol=1e-6)


    def test_get_gaussian_resolution(self):
        alg = VesuvioAnalysisRoutine()
        alg._gaussian_resolution = np.arange(18).reshape(2, 3, 3)
        alg._y_space_arrays = np.arange(18).reshape(2, 3, 3)

        alg._row_being_fit = 0
        centers = np.array([0.5, 3.6, 7.6]).reshape(-1, 1)
        np.testing.assert_allclose(alg._get_gaussian_resolution(centers), np.array([0, 4, 8]).reshape(-1, 1))

        alg._row_being_fit = 1
        centers = np.array([11.3, 9.6, 14.7]).reshape(-1, 1)
        np.testing.assert_allclose(alg._get_gaussian_resolution(centers), np.array([11, 12, 15]).reshape(-1, 1))


    def test_get_lorentzian_resolution(self):
        alg = VesuvioAnalysisRoutine()
        alg._lorentzian_resolution = np.arange(18).reshape(2, 3, 3)
        alg._y_space_arrays = np.arange(18).reshape(2, 3, 3)

        alg._row_being_fit = 0
        centers = np.array([0.5, 3.6, 7.6]).reshape(-1, 1)
        np.testing.assert_allclose(alg._get_lorentzian_resolution(centers), np.array([0, 4, 8]).reshape(-1, 1))

        alg._row_being_fit = 1
        centers = np.array([11.3, 9.6, 14.7]).reshape(-1, 1)
        np.testing.assert_allclose(alg._get_lorentzian_resolution(centers), np.array([11, 12, 15]).reshape(-1, 1))


    def test_set_means_and_std(self):
        alg = VesuvioAnalysisRoutine()
        alg._create_means_table = MagicMock()
        alg._profiles_table = MagicMock(rowCount=MagicMock(return_value=2), column=MagicMock(return_value=['1.0', '12.0']))

        def pick_column(arg):
            table = {
                '1.0 w': [5.6, 5.1, 0, 2, 5.4],
                '12.0 w': [2.1, 1, 0, 2.3, 1.9],
                '1.0 i': [7.8, 7.6, 0, 5, 7.3],
                '12.0 i': [3.1, 2, 0, 3.2, 3.1],
            }
            return table[arg]

        alg._table_fit_results = MagicMock(rowCount=MagicMock(return_value=5), column=MagicMock(side_effect=pick_column))

        alg._set_means_and_std()

        self.assertEqual(alg._table_fit_results.column.call_count, 4)
        self.assertEqual(alg._mean_widths[0], np.mean([5.6, 5.1, 5.4]))
        self.assertEqual(alg._std_widths[0], np.std([5.6, 5.1, 5.4]))
        self.assertEqual(alg._mean_widths[1], np.mean([2.1, 2.3, 1.9]))
        self.assertEqual(alg._std_widths[1], np.std([2.1, 2.3, 1.9]))
        self.assertEqual(alg._mean_intensity_ratios[0], np.nanmean(np.array([7.8, 7.6, np.nan, np.nan, 7.3]) / np.array([7.8+3.1, np.nan, np.nan, np.nan, 7.3+3.1])))
        self.assertEqual(alg._std_intensity_ratios[0], np.nanstd(np.array([7.8, 7.6, np.nan, np.nan, 7.3]) / np.array([7.8+3.1, np.nan, np.nan, np.nan, 7.3+3.1])))
        self.assertEqual(alg._mean_intensity_ratios[1], np.nanmean(np.array([3.1, np.nan, np.nan, 3.2, 3.1]) / np.array([7.8+3.1, np.nan, np.nan, np.nan, 7.3+3.1])))
        self.assertEqual(alg._std_intensity_ratios[1], np.nanstd(np.array([3.1, np.nan, np.nan, 3.2, 3.1]) / np.array([7.8+3.1, np.nan, np.nan, np.nan, 7.3+3.1])))


    def test_create_multiple_scattering_workspaces_adds_hydrogen_for_lowercase_backward_mode(self):
        alg = VesuvioAnalysisRoutine()
        alg._masses = np.array([12.0, 16.0])
        alg._mean_widths = np.array([8.0, 10.0])
        alg._mean_intensity_ratios = np.array([0.7, 0.3])
        alg._mode_running = ScatteringType.BACKWARD
        alg._chosen_index_for_h_ratio = 0
        alg._h_ratio = 2.0
        alg._transmission_guess = 0.85

        # Stop right after the mode-dependent block, nothing later is under test
        with patch("mvesuvio.analysis_reduction.VesuvioThickness", side_effect=StopIteration) as mock_thickness:
            with self.assertRaises(StopIteration):
                alg.create_multiple_scattering_workspaces()

        np.testing.assert_allclose(mock_thickness.call_args.kwargs["Masses"], [12.0, 16.0, 1.0079])


    def test_create_multiple_scattering_workspaces(self):
        unit_test_dir = Path(__file__).parent.parent.parent / "data/analysis/unit"
        ws_input = Load(str(unit_test_dir / "system_test_inputs_bckwd_cropped.nxs"))
        bench_tot_sctr = Load(str(unit_test_dir / "bench_system_test_inputs_bckwd_cropped_tot_sctr.nxs"))
        bench_mltp_sctr = Load(str(unit_test_dir / "bench_system_test_inputs_bckwd_cropped_mltp_sctr.nxs"))

        alg = VesuvioAnalysisRoutine()
        alg._workspace_for_corrections = ws_input
        alg._sample_shape_xml = '''<cuboid id="sample-shape">
            <left-front-bottom-point x="0.05" y="-0.05" z="0.0005" />
            <left-front-top-point x="0.05" y="0.05" z="0.0005"/>
            <left-back-bottom-point x="0.05" y="-0.05" z="-0.0005" />
            <right-front-bottom-point x="-0.05" y="-0.05" z="0.0005" />
            </cuboid>'''
        alg._masses = np.array([12.0, 16.0, 27.0])
        alg._mean_widths = np.array([15.35080, 8.72859, 13.89955])
        alg._mean_intensity_ratios = np.array([0.53110, 0.17667, 0.29223])
        alg._mode_running = "backward"
        alg._chosen_index_for_h_ratio = 0
        alg._h_ratio = 19.0620008206
        alg._transmission_guess = 0.8537
        alg._number_of_events = 1.0e5
        alg._multiple_scattering_order= 2

        alg.create_multiple_scattering_workspaces()

        (result, messages) = CompareWorkspaces(mtd["ws_input_mltp_sctr"], bench_mltp_sctr, Tolerance=1e-5)
        self.assertTrue(result)
        (result, messages) = CompareWorkspaces(mtd["ws_input_tot_sctr"], bench_tot_sctr, Tolerance=1e-5)
        self.assertTrue(result)


    def test_create_gamma_workspaces(self):
        unit_test_dir = Path(__file__).parent.parent.parent / "data/analysis/unit"
        ws_input = Load(str(unit_test_dir / "system_test_inputs_fwd_cropped.nxs"))
        bench_gamma_backgr = Load(str(unit_test_dir / "bench_system_test_inputs_fwd_cropped_gamma_backgr.nxs"))

        alg = VesuvioAnalysisRoutine()
        alg._workspace_for_corrections = ws_input
        alg._masses = np.array([1.0078, 12.0, 16.0, 27.0])
        alg._mean_widths = np.array([5.27454, 15.35080, 8.72859, 13.89955])
        alg._mean_intensity_ratios = np.array([0.91184, 0.06548, 0.00782, 0.01486])

        alg.create_gamma_workspaces()

        (result, messages) = CompareWorkspaces(mtd["ws_input_gamma_backgr"], bench_gamma_backgr, Tolerance=1e-5)
        self.assertTrue(result)


    def test_replace_zeros_with_ncp_for_corrections(self):

        dataY = np.random.random((3, 10))
        dataY[:, 5:8] = 0
        ncp = np.arange(30).reshape((3, 10))

        alg = VesuvioAnalysisRoutine()
        alg._fit_profiles_workspaces = {
            "total": MagicMock(
                extractY=MagicMock(return_value=ncp),
            )
        }
        alg._workspace_for_corrections = MagicMock(
            extractY=MagicMock(return_value=dataY),
            dataY=MagicMock(side_effect=lambda i: dataY[i]),
            getNumberHistograms=MagicMock(return_value=3)
        )

        expected_dataY = dataY.copy()
        expected_dataY[expected_dataY==0] = ncp[expected_dataY==0]

        with patch('mvesuvio.analysis_reduction.SumSpectra') as mock_sum_spectra:
            alg._replace_zeros_with_ncp_for_corrections()
            mock_sum_spectra.assert_called_once()
            np.testing.assert_allclose(dataY, expected_dataY)


    def test_calculate_summed_workspaces(self):

        alg = VesuvioAnalysisRoutine()

        ws_mock = Mock()
        ws_mock.name.side_effect=lambda:"_ws"
        alg._workspace_being_fit = ws_mock

        mock_ncp1 = Mock()
        mock_ncp1.name.side_effect=lambda:"_ncp1"
        mock_ncp2 = Mock()
        mock_ncp2.name.side_effect=lambda:"_ncp2"
        mock_fse1= Mock()
        mock_fse1.name.side_effect=lambda:"_fse1"
        mock_fse2= Mock()
        mock_fse2.name.side_effect=lambda:"_fse2"

        alg._fit_profiles_workspaces = {
            "1": mock_ncp1,
            "2": mock_ncp2
        }
        alg._fit_fse_workspaces = {
            "1": mock_fse1,
            "2": mock_fse2
        }
        with patch('mvesuvio.analysis_reduction.SumSpectra') as mock_sum_spectra:

            alg._calculate_summed_workspaces()

            mock_sum_spectra.assert_has_calls([
                call(InputWorkspace='_ws', OutputWorkspace='_ws_sum'),
                call(InputWorkspace='_ncp1', OutputWorkspace='_ncp1_sum'),
                call(InputWorkspace='_ncp2', OutputWorkspace='_ncp2_sum'),
                call(InputWorkspace='_fse1', OutputWorkspace='_fse1_sum'),
                call(InputWorkspace='_fse2', OutputWorkspace='_fse2_sum')
            ])

    def test_correct_for_multiple_scattering(self):
        mock_ms_ws = MagicMock()
        mock_ms_ws.name.side_effect = lambda: "ws_mlp_sctr"

        alg = VesuvioAnalysisRoutine()
        alg._multiple_scattering_correction = True
        alg._gamma_correction = False
        alg.create_multiple_scattering_workspaces = MagicMock(return_value=mock_ms_ws)
        alg._workspace_for_corrections = MagicMock(name=MagicMock(return_value="ws_for_corrections"))

        with patch('mvesuvio.analysis_reduction.Minus') as mock_minus:

            alg._correct_for_gamma_and_multiple_scattering("ws_to_correct")

            mock_minus.assert_has_calls([
                call(LHSWorkspace='ws_to_correct', RHSWorkspace='ws_mlp_sctr', OutputWorkspace='ws_to_correct')
            ])


    @patch('mvesuvio.analysis_reduction.VesuvioAnalysisRoutine.setPropertyValue')
    @patch('mvesuvio.analysis_reduction.print_table_workspace')
    def test_means_table(self, _mock1, _mock2):

        alg = VesuvioAnalysisRoutine()
        alg._profiles_table = MagicMock(column=MagicMock(return_value=['1', '2', '3']))
        alg._masses = np.array([1, 2, 3])
        alg._mean_widths = [5.1, 10.1, 12.3]
        alg._std_widths = [0.1, 0.2, 0.3]
        alg._mean_intensity_ratios = [0.7, 0.2, 0.3]
        alg._std_intensity_ratios = [0.01, 0.02, 0.03]
        alg._workspace_being_fit = MagicMock(name=MagicMock(return_value="_ws"))

        with patch('mvesuvio.analysis_reduction.CreateEmptyTableWorkspace') as mock_create_table_ws:
            table_mock = Mock(addRow=Mock())
            mock_create_table_ws.return_value = table_mock
            alg._create_means_table()
            table_mock.assert_has_calls([
                call.addColumn(type='str', name='label'),
                call.addColumn(type='float', name='mass'),
                call.addColumn(type='float', name='mean_width'),
                call.addColumn(type='float', name='std_width'),
                call.addColumn(type='float', name='mean_intensity'),
                call.addColumn(type='float', name='std_intensity'),
                call.addRow(['1', 1.0, 5.1, 0.1, 0.7, 0.01]),
                call.addRow(['2', 2.0, 10.1, 0.2, 0.2, 0.02]),
                call.addRow(['3', 3.0, 12.3, 0.3, 0.3, 0.03]),
                call.name()]
            )


    def test_correct_for_gamma_and_multiple_scattering(self):
        mock_gamma_ws = MagicMock()
        mock_gamma_ws.name.side_effect = lambda: "ws_gamma"
        mock_ms_ws = MagicMock()
        mock_ms_ws.name.side_effect = lambda: "ws_mlp_sctr"

        alg = VesuvioAnalysisRoutine()
        alg._gamma_correction = True
        alg.create_gamma_workspaces = MagicMock(return_value=mock_gamma_ws)
        alg._multiple_scattering_correction = True
        alg.create_multiple_scattering_workspaces = MagicMock(return_value=mock_ms_ws)
        alg._workspace_for_corrections = MagicMock(name=MagicMock(return_value="ws_for_corrections"))

        with patch('mvesuvio.analysis_reduction.Minus') as mock_minus:

            alg._correct_for_gamma_and_multiple_scattering("ws_to_correct")

            mock_minus.assert_has_calls([
                call(LHSWorkspace='ws_to_correct', RHSWorkspace='ws_gamma', OutputWorkspace='ws_to_correct'),
                call(LHSWorkspace='ws_to_correct', RHSWorkspace='ws_mlp_sctr', OutputWorkspace='ws_to_correct')
            ])


    def test_correct_for_gamma(self):
        mock_gamma_ws = MagicMock()
        mock_gamma_ws.name.side_effect = lambda: "ws_gamma"

        alg = VesuvioAnalysisRoutine()
        alg._multiple_scattering_correction = False
        alg._gamma_correction = True
        alg.create_gamma_workspaces = MagicMock(return_value=mock_gamma_ws)
        alg._workspace_for_corrections = MagicMock(name=MagicMock(return_value="ws_for_corrections"))

        with patch('mvesuvio.analysis_reduction.Minus') as mock_minus:

            alg._correct_for_gamma_and_multiple_scattering("ws_to_correct")

            mock_minus.assert_has_calls([
                call(LHSWorkspace='ws_to_correct', RHSWorkspace='ws_gamma', OutputWorkspace='ws_to_correct'),
            ])


    def test_run_applies_corrections_between_iterations(self):
        alg = VesuvioAnalysisRoutine()
        alg._profiles_table = Mock(rowCount=Mock(return_value=1))
        alg._name = "input"
        alg._number_of_iterations = 1
        alg._workspace_being_fit = Mock()
        alg._workspace_being_fit.name.return_value = "input"
        alg._workspace_for_corrections = Mock()
        alg._workspace_for_corrections.name.return_value = "corrections"
        alg._update_workspace_data = Mock()
        alg._fit_neutron_compton_profiles = Mock()
        alg._calculate_summed_workspaces = Mock()
        alg._set_means_and_std = Mock()
        alg._replace_zeros_with_ncp_for_corrections = Mock()
        alg._correct_for_gamma_and_multiple_scattering = Mock()
        alg._remask_columns_with_zeros = Mock()
        workspaces = {"input_0": Mock(), "input_1": Mock()}

        with (
            patch("mvesuvio.analysis_reduction.mtd", workspaces),
            patch("mvesuvio.analysis_reduction.CloneWorkspace") as clone_workspace,
            patch("mvesuvio.analysis_reduction.RenameWorkspace") as rename_workspace,
        ):
            result = alg.run()

        self.assertIs(result, alg)
        self.assertEqual(alg._update_workspace_data.call_count, 2)
        self.assertEqual(alg._fit_neutron_compton_profiles.call_count, 2)
        alg._replace_zeros_with_ncp_for_corrections.assert_called_once_with()
        alg._correct_for_gamma_and_multiple_scattering.assert_called_once_with("next_iteration")
        alg._remask_columns_with_zeros.assert_called_once_with("next_iteration")
        rename_workspace.assert_called_once_with(InputWorkspace="next_iteration", OutputWorkspace="input_1")
        clone_workspace.assert_has_calls(
            [
                call(InputWorkspace="input", OutputWorkspace="input_0"),
                call(InputWorkspace="corrections", OutputWorkspace="next_iteration"),
            ]
        )

    def test_run_requires_at_least_one_profile(self):
        alg = VesuvioAnalysisRoutine()
        alg._profiles_table = Mock(rowCount=Mock(return_value=0))

        with self.assertRaisesRegex(AssertionError, "Need at least one profile"):
            alg.run()

    def test_remask_columns_with_zeros(self):
        alg = VesuvioAnalysisRoutine()
        alg._zero_columns_boolean_mask = np.array([False, True, False, True])
        data_y = np.ones((2, 4))
        data_e = np.full((2, 4), 2.0)
        workspace = Mock()
        workspace.getNumberHistograms.return_value = 2
        workspace.dataY.side_effect = lambda row: data_y[row]
        workspace.dataE.side_effect = lambda row: data_e[row]

        with patch("mvesuvio.analysis_reduction.mtd", {"next_iteration": workspace}):
            alg._remask_columns_with_zeros("next_iteration")

        np.testing.assert_array_equal(data_y, [[1, 0, 1, 0], [1, 0, 1, 0]])
        np.testing.assert_array_equal(data_e, [[2, 0, 2, 0], [2, 0, 2, 0]])

if __name__ == "__main__":
    unittest.main()
