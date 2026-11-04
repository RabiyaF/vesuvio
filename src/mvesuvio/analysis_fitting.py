import matplotlib.pyplot as plt
import numpy as np
from mantid.simpleapi import (
    SumSpectra,
    Divide,
    Integration,
    CloneWorkspace,
    DeleteWorkspace,
    CreateWorkspace,
    CreateEmptyTableWorkspace,
    SaveAscii,
    Plus,
    GroupWorkspaces,
    mtd,
    Fit,
)
from scipy import optimize
from scipy import signal
from pathlib import Path
from iminuit import Minuit, cost
from iminuit.util import make_func_code, describe
import jacobi
import time
from mantid.kernel import logger

from mvesuvio.globals import FitModels, Masking
from mvesuvio.util import handle_config
from mvesuvio.util.general_helpers import print_table_workspace, extractWS

try:
    plt.style.use(["ggplot", handle_config.get_plots_config_file()])
except OSError:
    pass
repoPath = Path(__file__).absolute().parent  # Path to the repository

PLOTS_PROJECTION = "mantid"


class FitInYSpace:
    def __init__(self, fi, ws_joy, ws_joy_avg, ws_res, outputs_dir):
        self.ws_joy = ws_joy
        self.ws_joy_avg = ws_joy_avg
        self.ws_resolution = ws_res

        fi.outputs_dir = outputs_dir
        fi.figSavePath = fi.outputs_dir / "figures"
        fi.figSavePath.mkdir(exist_ok=True, parents=True)

        # TODO: Look into fixing this
        # If errors are zero, don't run minos or global fit
        if np.max(ws_joy_avg.extractE()) == 0:
            fi.run_minos = False
            fi.do_global_fit = False

        self.fitting_inputs = fi

    def run(self):
        ws_resolution_sum = SumSpectra(InputWorkspace=self.ws_resolution, OutputWorkspace=self.ws_resolution.name() + "_sum")
        normalise_workspace(ws_resolution_sum)

        wsJoY, wsJoYAvg = self.ws_joy, self.ws_joy_avg

        fitProfileMinuit(self.fitting_inputs, wsJoYAvg, ws_resolution_sum)
        fitProfileMantidFit(self.fitting_inputs, wsJoYAvg, ws_resolution_sum)

        printYSpaceFitResults()

        if self.fitting_inputs.do_global_fit:
            runGlobalFit(wsJoY, self.ws_resolution, self.fitting_inputs)

        save_workspaces(self.fitting_inputs)
        return


def normalise_workspace(ws_name):
    """Updates workspace with the normalised version."""
    Integration(InputWorkspace=ws_name, OutputWorkspace="tmp_int")
    Divide(LHSWorkspace=ws_name, RHSWorkspace="tmp_int", OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_int")


def fitProfileMinuit(yFitIC, wsYSpaceSym, wsRes):
    dataX, dataY, dataE = extractFirstSpectra(wsYSpaceSym)
    resX, resY, resE = extractFirstSpectra(wsRes)
    assert np.all(dataX == resX), "Resolution should operate on the same range as DataX"

    model, defaultPars, sharedPars = selectModelAndPars(yFitIC.fitting_model)

    xDelta, resDense = oddPointsRes(resX, resY)

    def convolvedModel(x, y0, *pars):
        return y0 + signal.convolve(model(x, *pars), resDense, mode="same") * xDelta

    signature = describe(model)[:]  # Build signature of convolved function
    signature[1:1] = ["y0"]  # Add intercept as first fitting parameter after range 'x'

    # Initialize limits as None, constrained later
    convolvedModel._parameters = {key: None for key in signature}
    defaultPars["y0"] = 0  # Add initialization of parameter to dictionary

    # Fit only valid values, ignore cut-offs
    dataXNZ, dataYNZ, dataENZ = selectNonZeros(dataX, dataY, dataE)

    # Fit with Minuit
    if np.all(dataE == 0):  # Choose fitting without weights
        costFun = MyLeastSquares(dataXNZ, dataYNZ, convolvedModel)
    else:
        costFun = cost.LeastSquares(dataXNZ, dataYNZ, dataENZ, convolvedModel)

    m = Minuit(costFun, **defaultPars)

    m.limits["A"] = (0, None)
    if yFitIC.fitting_model == FitModels.DOUBLEWELL:
        m.limits["d"] = (0, None)
        m.limits["R"] = (0, None)

    if yFitIC.fitting_model == FitModels.GAUSS:
        m.simplex()
        m.migrad()

        def constrFunc() -> None:  # No constraint function for gaussian profile
            return

    else:

        def constrFunc(*pars):  # Constrain physical model before convolution
            return model(dataXNZ, *pars[1:])  # First parameter is intercept, not part of model()

        m.simplex()
        m.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))

    # Explicit calculation of Hessian after the fit
    m.hesse()

    # Weighted Chi2
    chi2 = m.fval / (len(dataXNZ) - m.nfit)

    # Best fit and confidence band
    # Calculated for the whole range of dataX, including where zero
    dataYFit, dataYCov = jacobi.propagate(lambda pars: convolvedModel(dataX, *pars), m.values, m.covariance)
    dataYSigma = np.sqrt(np.diag(dataYCov))
    dataYSigma *= chi2  # Weight the confidence band
    Residuals = dataY - dataYFit

    wsYSpaceSym = CloneWorkspace(wsYSpaceSym, OutputWorkspace=wsYSpaceSym.name() + "_minuit_" + yFitIC.fitting_model)
    # Create workspace to store best fit curve and errors on the fit
    wsMinFit = createFitResultsWorkspace(wsYSpaceSym, dataX, dataY, dataE, dataYFit, dataYSigma, Residuals)
    saveMinuitPlot(yFitIC, wsMinFit, m)

    # Calculate correlation matrix
    corrMatrix = m.covariance.correlation()
    corrMatrix *= 100

    # Create correlation tableWorkspace
    createCorrelationTableWorkspace(wsYSpaceSym, m.parameters, corrMatrix)

    # Run Minos
    fitCols = runMinos(m, yFitIC, constrFunc, wsYSpaceSym.name())

    # Create workspace with final fitting parameters and their errors
    createFitParametersTableWorkspace(wsYSpaceSym, *fitCols, chi2)

    # TODO: This is bad
    output_name = wsYSpaceSym.name()
    DeleteWorkspace(output_name)
    GroupWorkspaces([ws_name for ws_name in mtd.getObjectNames() if "_minuit_" in ws_name], OutputWorkspace=output_name)
    return


def extractFirstSpectra(ws):
    dataY = ws.extractY()[0]
    dataX = ws.extractX()[0]
    dataE = ws.extractE()[0]
    return dataX, dataY, dataE


def selectModelAndPars(modelFlag):
    """
    Selects the function to fit.
    Specifies the starting parameters of that function as default parameters.
    The shared parameters are used in the global fit.
    The defaultPars should be in the same order as the signature of the function
    """

    if modelFlag == FitModels.GAUSS:

        def model(x, A, x0, sigma):
            return A / (2 * np.pi) ** 0.5 / sigma * np.exp(-((x - x0) ** 2) / 2 / sigma**2)

        defaultPars = {"A": 1, "x0": 0, "sigma": 5}
        sharedPars = ["sigma"]  # Used only in Global fit

    elif modelFlag == FitModels.GAUSS_CNTR:

        def model(x, A, sigma):
            return A / (2 * np.pi) ** 0.5 / sigma * np.exp(-(x**2) / 2 / sigma**2)

        defaultPars = {"A": 1, "sigma": 5}
        sharedPars = ["sigma"]  # Used only in Global fit

    elif modelFlag == FitModels.GCC4C6:

        def model(x, A, x0, sigma1, c4, c6):
            return (
                A
                * np.exp(-((x - x0) ** 2) / 2 / sigma1**2)
                / (np.sqrt(2 * np.pi * sigma1**2))
                * (
                    1
                    + c4 / 32 * (16 * ((x - x0) / np.sqrt(2) / sigma1) ** 4 - 48 * ((x - x0) / np.sqrt(2) / sigma1) ** 2 + 12)
                    + c6
                    / 384
                    * (
                        64 * ((x - x0) / np.sqrt(2) / sigma1) ** 6
                        - 480 * ((x - x0) / np.sqrt(2) / sigma1) ** 4
                        + 720 * ((x - x0) / np.sqrt(2) / sigma1) ** 2
                        - 120
                    )
                )
            )

        defaultPars = {"A": 1, "x0": 0, "sigma1": 6, "c4": 0, "c6": 0}
        sharedPars = ["sigma1", "c4", "c6"]  # Used only in Global fit

    elif modelFlag == FitModels.GCC4C6_CNTR:

        def model(x, A, sigma1, c4, c6):
            return (
                A
                * np.exp(-(x**2) / 2 / sigma1**2)
                / (np.sqrt(2 * np.pi * sigma1**2))
                * (
                    1
                    + c4 / 32 * (16 * (x / np.sqrt(2) / sigma1) ** 4 - 48 * (x / np.sqrt(2) / sigma1) ** 2 + 12)
                    + c6
                    / 384
                    * (
                        64 * (x / np.sqrt(2) / sigma1) ** 6
                        - 480 * (x / np.sqrt(2) / sigma1) ** 4
                        + 720 * (x / np.sqrt(2) / sigma1) ** 2
                        - 120
                    )
                )
            )

        defaultPars = {"A": 1, "sigma1": 6, "c4": 0, "c6": 0}
        sharedPars = ["sigma1", "c4", "c6"]  # Used only in Global fit

    elif modelFlag == FitModels.GCC4:

        def model(x, A, x0, sigma1, c4):
            return (
                A
                * np.exp(-((x - x0) ** 2) / 2 / sigma1**2)
                / (np.sqrt(2 * np.pi * sigma1**2))
                * (1 + c4 / 32 * (16 * ((x - x0) / np.sqrt(2) / sigma1) ** 4 - 48 * ((x - x0) / np.sqrt(2) / sigma1) ** 2 + 12))
            )

        defaultPars = {"A": 1, "x0": 0, "sigma1": 6, "c4": 0}
        sharedPars = ["sigma1", "c4"]  # Used only in Global fit

    elif modelFlag == FitModels.GCC4_CNTR:

        def model(x, A, sigma1, c4):
            return (
                A
                * np.exp(-(x**2) / 2 / sigma1**2)
                / (np.sqrt(2 * np.pi * sigma1**2))
                * (1 + c4 / 32 * (16 * (x / np.sqrt(2) / sigma1) ** 4 - 48 * (x / np.sqrt(2) / sigma1) ** 2 + 12))
            )

        defaultPars = {"A": 1, "sigma1": 6, "c4": 0}
        sharedPars = ["sigma1", "c4"]  # Used only in Global fit

    elif modelFlag == FitModels.GCC6:

        def model(x, A, x0, sigma1, c6):
            return (
                A
                * np.exp(-((x - x0) ** 2) / 2 / sigma1**2)
                / (np.sqrt(2 * np.pi * sigma1**2))
                * (
                    1
                    + +c6
                    / 384
                    * (
                        64 * ((x - x0) / np.sqrt(2) / sigma1) ** 6
                        - 480 * ((x - x0) / np.sqrt(2) / sigma1) ** 4
                        + 720 * ((x - x0) / np.sqrt(2) / sigma1) ** 2
                        - 120
                    )
                )
            )

        defaultPars = {"A": 1, "x0": 0, "sigma1": 6, "c6": 0}
        sharedPars = ["sigma1", "c6"]  # Used only in Global fit

    elif modelFlag == FitModels.GCC6_CNTR:

        def model(x, A, sigma1, c6):
            return (
                A
                * np.exp(-(x**2) / 2 / sigma1**2)
                / (np.sqrt(2 * np.pi * sigma1**2))
                * (
                    1
                    + +c6
                    / 384
                    * (
                        64 * (x / np.sqrt(2) / sigma1) ** 6
                        - 480 * (x / np.sqrt(2) / sigma1) ** 4
                        + 720 * (x / np.sqrt(2) / sigma1) ** 2
                        - 120
                    )
                )
            )

        defaultPars = {"A": 1, "sigma1": 6, "c6": 0}
        sharedPars = ["sigma1", "c6"]  # Used only in Global fit

    elif modelFlag == FitModels.DOUBLEWELL:

        def model(x, A, d, R, sig1, sig2):
            # h = 2.04
            theta = np.linspace(0, np.pi, 300)[:, np.newaxis]  # 300 points seem like a good estimate for ~10 examples
            y = x[np.newaxis, :]

            sigTH = np.sqrt(sig1**2 * np.cos(theta) ** 2 + sig2**2 * np.sin(theta) ** 2)
            alpha = 2 * (d * sig2 * sig1 * np.sin(theta) / sigTH) ** 2
            beta = (2 * sig1**2 * d * np.cos(theta) / sigTH**2) * y
            denom = 2.506628 * sigTH * (1 + R**2 + 2 * R * np.exp(-2 * d**2 * sig1**2))
            jp = np.exp(-(y**2) / (2 * sigTH**2)) * (1 + R**2 + 2 * R * np.exp(-alpha) * np.cos(beta)) / denom
            jp *= np.sin(theta)

            JBest = np.trapezoid(jp, x=theta, axis=0)
            JBest /= np.abs(np.trapezoid(JBest, x=y))
            JBest *= A
            return JBest

        defaultPars = {
            "A": 1,
            "d": 1,
            "R": 1,
            "sig1": 3,
            "sig2": 5,
        }  # TODO: Starting parameters and bounds?
        sharedPars = ["d", "R", "sig1", "sig2"]  # Only varying parameter is amplitude A

    elif modelFlag == FitModels.GAUSS2D:
        # Anisotropic case
        def model(x, A, sig1, sig2):
            # h = 2.04
            theta = np.linspace(0, np.pi, 300)[:, np.newaxis]
            y = x[np.newaxis, :]

            sigTH = np.sqrt(sig1**2 * np.cos(theta) ** 2 + sig2**2 * np.sin(theta) ** 2)
            jp = np.exp(-(y**2) / (2 * sigTH**2)) / (2.506628 * sigTH)
            jp *= np.sin(theta)

            JBest = np.trapezoid(jp, x=theta, axis=0)
            JBest /= np.abs(np.trapezoid(JBest, x=y))
            JBest *= A
            return JBest

        defaultPars = {"A": 1, "sig1": 3, "sig2": 5}
        sharedPars = ["sig1", "sig2"]

    elif modelFlag == FitModels.GAUSS3D:

        def model(x, A, sig_x, sig_y, sig_z):
            y = x[:, np.newaxis, np.newaxis]
            n_steps = 50  # Low number of integration steps because otherwise too slow
            theta = np.linspace(0, np.pi / 2, n_steps)[np.newaxis, :, np.newaxis]
            phi = np.linspace(0, np.pi / 2, n_steps)[np.newaxis, np.newaxis, :]

            S2_inv = (
                np.sin(theta) ** 2 * np.cos(phi) ** 2 / sig_x**2
                + np.sin(theta) ** 2 * np.sin(phi) ** 2 / sig_y**2
                + np.cos(theta) ** 2 / sig_z**2
            )

            J = np.sin(theta) / S2_inv * np.exp(-(y**2) / 2 * S2_inv)

            J = np.trapezoid(J, x=phi, axis=2)[:, :, np.newaxis]  # Keep shape
            J = np.trapezoid(J, x=theta, axis=1)

            J *= A * 2 / np.pi * 1 / np.sqrt(2 * np.pi) * 1 / (sig_x * sig_y * sig_z)  # Normalisation
            J = J.squeeze()
            return J

        defaultPars = {"A": 1, "sig_x": 5, "sig_y": 5, "sig_z": 5}
        sharedPars = ["sig_x", "sig_y", "sig_z"]

    else:
        options = [fit.value for fit in FitModels]
        raise ValueError(f"Fitting Model not recognized, available options: {', '.join(options)}")

    logger.notice(f"\nShared Parameters: {[key for key in sharedPars]}")
    logger.notice(f"\nUnshared Parameters: {[key for key in defaultPars if key not in sharedPars]}")

    assert all(isinstance(item, str) for item in sharedPars), "Parameters in list must be strings."
    assert describe(model)[-len(sharedPars) :] == sharedPars, (
        "Function signature needs to have shared parameters at the end: model(*unsharedPars, *sharedPars)"
    )

    return model, defaultPars, sharedPars


def selectNonZeros(dataX, dataY, dataE):
    """
    Selects non zero points.
    Uses zeros in dataY becasue dataE can be all zeros in one of the bootstrap types.
    """
    zeroMask = dataY == 0

    dataXNZ = dataX[~zeroMask]
    dataYNZ = dataY[~zeroMask]
    dataENZ = dataE[~zeroMask]
    return dataXNZ, dataYNZ, dataENZ


class MyLeastSquares:
    """
    Generic least-squares cost function without error.
    This structure is required for high compatibility with Minuit.
    """

    errordef = Minuit.LEAST_SQUARES  # for Minuit to compute errors correctly

    def __init__(self, x, y, model):
        self.model = model  # model predicts y for given x
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.func_code = make_func_code(describe(model)[1:])

    def __call__(self, *par):  # we accept a variable number of model parameters
        ym = self.model(self.x, *par)
        return np.sum((self.y - ym) ** 2)

    @property
    def ndata(self):
        return len(self.x)


def createFitResultsWorkspace(wsYSpaceSym, dataX, dataY, dataE, dataYFit, dataYSigma, Residuals):
    """Creates workspace similar to the ones created by Mantid Fit."""

    ws_fit_complete = CreateWorkspace(
        DataX=np.concatenate((dataX, dataX, dataX)),
        DataY=np.concatenate((dataY, dataYFit, Residuals)),
        DataE=np.concatenate((dataE, dataYSigma, np.zeros(len(dataE)))),
        NSpec=3,
        OutputWorkspace=wsYSpaceSym.name() + "_Workspace",
    )
    return ws_fit_complete


def saveMinuitPlot(yFitIC, wsMinuitFit, mObj):
    """Saves figure with Minuit Fit."""

    leg = ""
    for p, v, e in zip(mObj.parameters, mObj.values, mObj.errors):
        leg += f"${p}={v:.2f} \\pm {e:.2f}$\n"

    fig, ax = plt.subplots(subplot_kw={"projection": PLOTS_PROJECTION})
    ax.errorbar(wsMinuitFit, "r-", wkspIndex=1, label=leg, elinewidth=1.5)
    ax.errorbar(wsMinuitFit, "k.", wkspIndex=0, label="Weighted Avg", elinewidth=1.5)
    ax.plot(wsMinuitFit, "b.", wkspIndex=2, label="Residuals")
    ax.set_xlabel("YSpace")
    ax.set_ylabel("Counts")
    ax.set_title("Minuit Fit")
    ax.legend()

    fileName = wsMinuitFit.name() + ".pdf"
    savePath = yFitIC.figSavePath / fileName
    plt.savefig(savePath, bbox_inches="tight")
    plt.close(fig)
    return


def createCorrelationTableWorkspace(wsYSpaceSym, parameters, corrMatrix):
    tableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsYSpaceSym.name() + "_NormalizedCovarianceMatrix")
    tableWS.setTitle("Minuit Fit")
    tableWS.addColumn(type="str", name="Name")
    for p in parameters:
        tableWS.addColumn(type="float", name=p)
    for p, arr in zip(parameters, corrMatrix):
        tableWS.addRow([p] + list(arr))
    print_table_workspace(tableWS)


def runMinos(mObj, yFitIC, constrFunc, wsName):
    """Outputs columns to be displayed in a table workspace"""

    # Extract info from fit before running any MINOS
    parameters = list(mObj.parameters)
    values = list(mObj.values)
    errors = list(mObj.errors)

    # If minos is set not to run, ouput columns with zeros on minos errors
    if not (yFitIC.run_minos):
        minosAutoErr = list(np.zeros((len(parameters), 2)))
        minosManErr = list(np.zeros((len(parameters), 2)))
        return parameters, values, errors, minosAutoErr, minosManErr

    bestFitVals = {}
    bestFitErrs = {}
    for p, v, e in zip(mObj.parameters, mObj.values, mObj.errors):
        bestFitVals[p] = v
        bestFitErrs[p] = e

    if yFitIC.fitting_model == FitModels.GAUSS:  # Case with no positivity constraint, can use automatic minos()
        mObj.minos()
        me = mObj.merrors

        # Build minos errors lists in suitable format
        minosAutoErr = []
        for p in parameters:
            minosAutoErr.append([me[p].lower, me[p].upper])
        minosManErr = list(np.zeros(np.array(minosAutoErr).shape))

        if yFitIC.show_plots:
            plotAutoMinos(mObj, wsName, yFitIC)

    else:  # Case with positivity constraint on function, use manual implementation
        # Changes values of minuit obj m, do not use m below this point
        merrors, fig = runAndPlotManualMinos(mObj, constrFunc, bestFitVals, bestFitErrs, yFitIC.show_plots, yFitIC, wsName)

        # Same as above, but the other way around
        minosManErr = []
        for p in parameters:
            minosManErr.append(merrors[p])
        minosAutoErr = list(np.zeros(np.array(minosManErr).shape))

        if yFitIC.show_plots:
            fig.show()
        else:
            plt.close(fig)
            del fig

    return parameters, values, errors, minosAutoErr, minosManErr


def runAndPlotManualMinos(minuitObj, constrFunc, bestFitVals, bestFitErrs, showPlots, yFitIC, wsName):
    """
    Runs brute implementation of minos algorithm and
    plots the profile for each parameter along the way.
    """
    # Reason for two distinct operations inside the same function is that its easier
    # to build the minos plots for each parameter as they are being calculated.
    logger.notice("\nRunning Minos ... \n")

    # Set format of subplots
    height = 2
    width = int(np.ceil(len(minuitObj.parameters) / 2))
    figsize = (12, 7)
    # Output plot to Mantid
    fig, axs = plt.subplots(
        height,
        width,
        tight_layout=True,
        figsize=figsize,
        subplot_kw={"projection": PLOTS_PROJECTION},
    )
    fig.canvas.manager.set_window_title(wsName + "_minos")

    merrors = {}
    for p, ax in zip(minuitObj.parameters, axs.flat):
        lerr, uerr = runMinosForPar(minuitObj, constrFunc, p, 2, ax, bestFitVals, bestFitErrs, showPlots)
        merrors[p] = np.array([lerr, uerr])

    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:  # If empty list
            ax.set_visible(False)

    # ALl axes share same legend, so set figure legend to first axis
    handle, label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handle, label, loc="lower right")
    savePath = yFitIC.figSavePath / fig.canvas.manager.get_window_title()
    plt.savefig(savePath, bbox_inches="tight")
    return merrors, fig


def runMinosForPar(minuitObj, constrFunc, var: str, bound: int, ax, bestFitVals, bestFitErrs, showPlots):
    resetMinuit(minuitObj, bestFitVals, bestFitErrs)
    # Run Fitting procedures again to be on the safe side and reset to minimum
    minuitObj.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))
    minuitObj.hesse()

    # Extract parameters from minimum
    varVal = minuitObj.values[var]
    varErr = minuitObj.errors[var]
    # Store fval of best fit
    fValsMin = minuitObj.fval  # Used to calculate error bands at the end

    varSpace = buildVarRange(bound, varVal, varErr)

    # Split variable space into right and left side
    lhsVarSpace, rhsVarSpace = np.split(varSpace, 2)
    lhsVarSpace = np.flip(lhsVarSpace)  # Flip to start at minimum

    for minimizer in ("Scipy", "Migrad"):
        resetMinuit(minuitObj, bestFitVals, bestFitErrs)
        rhsMinos = runMinosOnRange(minuitObj, var, rhsVarSpace, minimizer, constrFunc)

        resetMinuit(minuitObj, bestFitVals, bestFitErrs)
        lhsMinos = runMinosOnRange(minuitObj, var, lhsVarSpace, minimizer, constrFunc)

        wholeMinos = np.concatenate((np.flip(lhsMinos), rhsMinos), axis=None)  # Flip left hand side again

        if minimizer == "Scipy":  # Calculate minos errors from constrained scipy
            lerr, uerr = errsFromMinosCurve(varSpace, varVal, wholeMinos, fValsMin, dChi2=1)
            ax.plot(varSpace, wholeMinos, label="fVals Constr Scipy")

        elif minimizer == "Migrad":  # Plot migrad as well to see the difference between constrained and unconstrained
            plotProfile(ax, var, varSpace, wholeMinos, lerr, uerr, fValsMin, varVal, varErr)
        else:
            raise ValueError("Minimizer not recognized.")

    resetMinuit(minuitObj, bestFitVals, bestFitErrs)
    return lerr, uerr


def resetMinuit(minuitObj, bestFitVals, bestFitErrs):
    """Sets Minuit parameters to best fit values and errors."""
    for p in bestFitVals:
        minuitObj.values[p] = bestFitVals[p]
        minuitObj.errors[p] = bestFitErrs[p]
    return


def buildVarRange(bound, varVal, varErr):
    """Range of points over which cost function is evaluated."""
    # Create variable space more dense near the minima using a quadratic density
    limit = (bound * varErr) ** (1 / 2)  # Square root is corrected below
    varSpace = np.linspace(-limit, limit, 30)
    varSpace = varSpace**2 * np.sign(varSpace) + varVal
    assert len(varSpace) % 2 == 0, "Number of points in Minos range needs to be even"
    return varSpace


def runMinosOnRange(minuitObj, var, varRange, minimizer, constrFunc):
    result = np.zeros(varRange.size)
    minuitObj.fixed[var] = True

    # Unconstrained fit over side range
    for i, value in enumerate(varRange):
        minuitObj.values[var] = value  # Fix variable

        if minimizer == "Migrad":
            minuitObj.migrad()  # Fit
        elif minimizer == "Scipy":
            minuitObj.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))

        result[i] = minuitObj.fval  # Store minimum

    minuitObj.fixed[var] = False
    return result


def errsFromMinosCurve(varSpace, varVal, fValsScipy, fValsMin, dChi2=1):
    # Use intenpolation to create dense array of fmin values
    varSpaceDense = np.linspace(np.min(varSpace), np.max(varSpace), 100000)
    fValsScipyDense = np.interp(varSpaceDense, varSpace, fValsScipy)
    # Calculate points of intersection with line delta fmin val = 1
    idxErr = np.argwhere(np.diff(np.sign(fValsScipyDense - fValsMin - 1)))

    if idxErr.size != 2:  # Intersections not found, do not plot error range
        lerr, uerr = 0.0, 0.0
    else:
        lerr, uerr = varSpaceDense[idxErr].flatten() - varVal

        if lerr * uerr >= 0:  # Case where we get either two positive or two negative errors, ill defined profile
            lerr, uerr = 0, 0

    return lerr, uerr


def plotAutoMinos(minuitObj, wsName, yFitIC):
    # Set format of subplots
    height = 2
    width = int(np.ceil(len(minuitObj.parameters) / 2))
    figsize = (12, 7)
    # Output plot to Mantid
    fig, axs = plt.subplots(
        height,
        width,
        tight_layout=True,
        figsize=figsize,
        subplot_kw={"projection": PLOTS_PROJECTION},
    )
    fig.canvas.manager.set_window_title(wsName + "_autominos")

    for p, ax in zip(minuitObj.parameters, axs.flat):
        loc, fvals, status = minuitObj.mnprofile(p, bound=2)

        minfval = minuitObj.fval
        minp = minuitObj.values[p]
        hessp = minuitObj.errors[p]
        lerr = minuitObj.merrors[p].lower
        uerr = minuitObj.merrors[p].upper
        plotProfile(ax, p, loc, fvals, lerr, uerr, minfval, minp, hessp)

    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:  # If empty list
            ax.set_visible(False)

    # ALl axes share same legend, so set figure legend to first axis
    handle, label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handle, label, loc="lower right")
    savePath = yFitIC.figSavePath / fig.canvas.manager.get_window_title()
    plt.savefig(savePath, bbox_inches="tight")
    if yFitIC.show_plots:
        fig.show()
    else:
        plt.close(fig)
        del fig, axs


def plotProfile(ax, var, varSpace, fValsMigrad, lerr, uerr, fValsMin, varVal, varErr):
    """
    Plots likelihood profilef for the Migrad fvals.
    varSpace : x axis
    fValsMigrad : y axis
    """

    ax.set_title(var + f" = {varVal:.3f} {lerr:.3f} {uerr:+.3f}")

    ax.plot(varSpace, fValsMigrad, label="fVals Migrad")

    ax.axvspan(lerr + varVal, uerr + varVal, alpha=0.2, color="red", label="Minos error")
    ax.axvspan(
        varVal - varErr,
        varVal + varErr,
        alpha=0.2,
        color="green",
        label="Hessian Std error",
    )

    ax.axvline(varVal, 0.03, 0.97, color="k", ls="--")
    ax.axhline(fValsMin + 1, 0.03, 0.97, color="k")
    ax.axhline(fValsMin, 0.03, 0.97, color="k")


def createFitParametersTableWorkspace(wsYSpaceSym, parameters, values, errors, minosAutoErr, minosManualErr, chi2):
    # Create Parameters workspace
    tableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsYSpaceSym.name() + "_Parameters")
    tableWS.setTitle("Minuit Fit")
    tableWS.addColumn(type="str", name="Name")
    tableWS.addColumn(type="float", name="Value")
    tableWS.addColumn(type="float", name="Error")
    tableWS.addColumn(type="float", name="Auto Minos Error-")
    tableWS.addColumn(type="float", name="Auto Minos Error+")
    tableWS.addColumn(type="float", name="Manual Minos Error-")
    tableWS.addColumn(type="float", name="Manual Minos Error+")

    for p, v, e, mae, mme in zip(parameters, values, errors, minosAutoErr, minosManualErr):
        tableWS.addRow([p, v, e, mae[0], mae[1], mme[0], mme[1]])

    tableWS.addRow(["Cost function", chi2, 0, 0, 0, 0, 0])
    return


def oddPointsRes(x, res):
    """
    Make a odd grid that ensures a resolution with a single peak at the center.
    """

    assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
    assert x.size == res.size, "x and res need to be the same size!"

    if res.size % 2 == 0:
        dens = res.size + 1  # If even change to odd
    else:
        dens = res.size  # If odd, keep being odd

    xDense = np.linspace(np.min(x), np.max(x), dens)  # Make gridd with odd number of points - peak at center
    xDelta = xDense[1] - xDense[0]

    resDense = np.interp(xDense, x, res)

    return xDelta, resDense


def fitProfileMantidFit(yFitIC, wsYSpaceSym, wsRes):
    logger.notice("\nFitting on the sum of spectra in the West domain ...\n")
    for minimizer in ["Levenberg-Marquardt", "Simplex"]:
        if yFitIC.fitting_model == FitModels.GAUSS:
            function = f"""composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0;
            name=UserFunction,Formula=y0 + A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
            y0=0,A=1,x0=0,sigma=5,   ties=()"""

        elif yFitIC.fitting_model == FitModels.GCC4C6:
            function = f"""
            composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0,X=(),Y=();
            name=UserFunction,Formula=y0 + A*exp( -(x-x0)^2/2./sigma1^2)/(sqrt(2.*3.1415*sigma1^2))
            *(1.+c4/32.*(16.*((x-x0)/sqrt(2)/sigma1)^4-48.*((x-x0)/sqrt(2)/sigma1)^2+12)+c6/384*
            (64*((x-x0)/sqrt(2)/sigma1)^6 - 480*((x-x0)/sqrt(2)/sigma1)^4 + 720*((x-x0)/sqrt(2)/sigma1)^2 - 120)),
            y0=0, A=1,x0=0,sigma1=4.0,c4=0.0,c6=0.0,ties=(),constraints=(0<c4,0<c6)
            """
        elif yFitIC.fitting_model == FitModels.GCC4:
            function = f"""
            composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0,X=(),Y=();
            name=UserFunction,Formula=y0 + A*exp( -(x-x0)^2/2./sigma1^2)/(sqrt(2.*3.1415*sigma1^2))
            *(1.+c4/32.*(16.*((x-x0)/sqrt(2)/sigma1)^4-48.*((x-x0)/sqrt(2)/sigma1)^2+12)),
            y0=0, A=1,x0=0,sigma1=4.0,c4=0.0,ties=()
            """
        elif yFitIC.fitting_model == FitModels.GCC6:
            function = f"""
            composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0,X=(),Y=();
            name=UserFunction,Formula=y0 + A*exp( -(x-x0)^2/2./sigma1^2)/(sqrt(2.*3.1415*sigma1^2))
            *(1.+c6/384*(64*((x-x0)/sqrt(2)/sigma1)^6 - 480*((x-x0)/sqrt(2)/sigma1)^4 + 720*((x-x0)/sqrt(2)/sigma1)^2 - 120)),
            y0=0, A=1,x0=0,sigma1=4.0,c6=0.0,ties=()
            """
        elif (
            (yFitIC.fitting_model == FitModels.DOUBLEWELL)
            | (yFitIC.fitting_model == FitModels.GAUSS2D)
            | (yFitIC.fitting_model == FitModels.GAUSS3D)
            | (yFitIC.fitting_model == FitModels.GAUSS_CNTR)
            | (yFitIC.fitting_model == FitModels.GCC4C6_CNTR)
            | (yFitIC.fitting_model == FitModels.GCC4_CNTR)
            | (yFitIC.fitting_model == FitModels.GCC6_CNTR)
        ):
            logger.warning("Fitting model recognized but not currently implemented in Mantid Fit. Skipping Mantid Fit ...")
            return
        else:
            options = [fit.value for fit in FitModels]
            raise ValueError(f"Fitting Model not recognized, available options: {', '.join(options)}")

        suffix = "lm" if minimizer == "Levenberg-Marquardt" else minimizer.lower()
        outputName = wsYSpaceSym.name() + f"_{suffix}_" + yFitIC.fitting_model
        CloneWorkspace(InputWorkspace=wsYSpaceSym, OutputWorkspace=outputName)

        # Fit produces output workspaces with results
        Fit(
            Function=function,
            InputWorkspace=outputName,
            Output=outputName,
            Minimizer=minimizer,
        )

        # Delete workspace prior to replacing it, just to be sure
        DeleteWorkspace(outputName)
        # Group output workspaces
        ws_to_group = [ws_name for ws_name in mtd.getObjectNames() if f"_{suffix}_" in ws_name]
        GroupWorkspaces(ws_to_group, OutputWorkspace=outputName)
    return


def printYSpaceFitResults():
    for ws_name in mtd.getObjectNames():
        if ws_name.endswith("Parameters"):
            print_table_workspace(mtd[ws_name])


def runGlobalFit(wsYSpace, wsRes, IC):
    logger.notice("\nRunning GLobal Fit ...\n")

    dataX, dataY, dataE, dataRes, instrPars = extractData(wsYSpace, wsRes, IC)
    dataX, dataY, dataE, dataRes, instrPars = takeOutMaskedSpectra(dataX, dataY, dataE, dataRes, instrPars)

    idxList = groupDetectors(instrPars, IC)
    dataX, dataY, dataE, dataRes = avgWeightDetGroups(dataX, dataY, dataE, dataRes, idxList, IC)

    # if IC.do_symmetrisation:
    #     dataY, dataE = weightedSymArr(dataY, dataE)

    model, defaultPars, sharedPars = selectModelAndPars(IC.fitting_model)

    totCost = 0
    for i, (x, y, yerr, res) in enumerate(zip(dataX, dataY, dataE, dataRes)):
        totCost += calcCostFun(model, i, x, y, yerr, res, sharedPars)

    defaultPars["y0"] = 0  # Introduce default parameter for convolved model

    assert len(describe(totCost)) == len(sharedPars) + len(dataY) * (len(defaultPars) - len(sharedPars)), (
        f"Wrong parameters for Global Fit:\n{describe(totCost)}"
    )

    # Minuit Fit with global cost function and local+global parameters
    initPars = minuitInitialParameters(defaultPars, sharedPars, len(dataY))

    logger.notice("\nRunning Global Fit ...\n")
    m = Minuit(totCost, **initPars)

    for i in range(len(dataY)):  # Set limits for unshared parameters
        m.limits["A" + str(i)] = (0, np.inf)

    if IC.fitting_model == FitModels.DOUBLEWELL:
        m.limits["d"] = (0, np.inf)  # Shared parameters
        m.limits["R"] = (0, np.inf)

    t0 = time.time()
    if IC.fitting_model == FitModels.GAUSS:
        m.simplex()
        m.migrad()

    else:
        totSig = describe(totCost)  # This signature has 'x' already removed
        sharedIdxs = [totSig.index(shPar) for shPar in sharedPars]
        nCostFunctions = len(totCost)  # Number of individual cost functions
        x = dataX[0]

        def constr(*pars):
            """
            Constraint for positivity of non Gaussian function.
            Input: All parameters defined in global cost function.
            x is the range for each individual cost fun, defined outside function.
            Builds array with all constraints from individual functions.
            """

            sharedPars = [pars[i] for i in sharedIdxs]  # sigma1, c4, c6 in original GC
            unsharedPars = np.delete(pars, sharedIdxs, None)
            unsharedParsSplit = np.split(unsharedPars, nCostFunctions)  # Splits unshared parameters per individual cost fun

            joinedGC = np.zeros(nCostFunctions * x.size)
            for i, unshParsModel in enumerate(
                unsharedParsSplit
            ):  # Attention to format of unshared and shared parameters when calling model
                joinedGC[i * x.size : (i + 1) * x.size] = model(
                    x, *unshParsModel[1:], *sharedPars
                )  # Intercept is first of unshared parameters

            return joinedGC

        m.simplex()
        m.scipy(constraints=optimize.NonlinearConstraint(constr, 0, np.inf))

    t1 = time.time()
    logger.notice(f"\nTime of fitting: {t1 - t0:.2f} seconds")

    # Explicitly calculate errors
    m.hesse()

    # Number of non zero points (considered in the fit) minus no of parameters
    chi2 = m.fval / (np.sum(dataE != 0) - m.nfit)

    fit_ws_group = save_result_of_global_fit(dataX, dataY, dataE, m, totCost, wsYSpace.name(), IC.fitting_model, chi2)

    if IC.show_plots:
        plot_global_fit(fit_ws_group, IC)

    # Pass into array to store values in variable
    return np.array(m.values), np.array(m.errors)


def extractData(ws, wsRes, ic):
    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()
    dataRes = wsRes.extractY()
    instrPars = loadInstrParsFileIntoArray(ic)
    assert len(instrPars) == len(dataY), "Load of IP file not working correctly, probable issue with indexing."
    return dataX, dataY, dataE, dataRes, instrPars


def loadInstrParsFileIntoArray(ic):
    ipFilesPath = Path(handle_config.read_cached_var("caching.ipfolder"))
    data = np.loadtxt(str(ipFilesPath / ic.instrument_parameters_file), dtype=str)[1:].astype(float)
    spectra = data[:, 0]
    firstSpec, lastSpec = [int(d) for d in ic.detectors.split("-")]
    select_rows = np.where((spectra >= firstSpec) & (spectra <= lastSpec))
    instrPars = data[select_rows]
    return instrPars


def takeOutMaskedSpectra(dataX, dataY, dataE, dataRes, instrPars):
    zerosRowMask = np.all(dataY == 0, axis=1)
    dataY = dataY[~zerosRowMask]
    dataE = dataE[~zerosRowMask]
    dataX = dataX[~zerosRowMask]
    dataRes = dataRes[~zerosRowMask]
    instrPars = instrPars[~zerosRowMask]
    return dataX, dataY, dataE, dataRes, instrPars


# ------- Groupings


def groupDetectors(ipData, yFitIC):
    """
    Uses the method of k-means to find clusters in theta-L1 space.
    Input: instrument parameters to extract L1 and theta of detectors.
    Output: list of group lists containing the idx of spectra.
    """

    checkNGroupsValid(yFitIC, ipData)

    logger.notice(f"\nNumber of gropus: {yFitIC.number_of_global_fit_groups}")

    L1 = ipData[:, -1].copy()
    theta = ipData[:, 2].copy()

    # Normalize  ranges to similar values, needed for clustering
    L1 /= np.sum(L1)
    theta /= np.sum(theta)

    L1 *= 2  # Bigger weight to L1

    points = np.vstack((L1, theta)).T
    assert points.shape == (len(L1), 2), "Wrong shape."
    # Initial centers of groups
    startingIdxs = np.linspace(0, len(points) - 1, yFitIC.number_of_global_fit_groups).astype(int)
    centers = points[startingIdxs, :]  # Centers of cluster groups, NOT fitting parameter

    if False:  # Set to True to investigate problems with groupings
        plotDetsAndInitialCenters(L1, theta, centers)

    clusters = kMeansClustering(points, centers)
    idxList = formIdxList(clusters)

    if yFitIC.show_plots:
        fig, ax = plt.subplots(tight_layout=True, subplot_kw={"projection": PLOTS_PROJECTION})
        fig.canvas.manager.set_window_title("Grouping of detectors")
        plotFinalGroups(ax, ipData, idxList)
        fig.show()
    return idxList


def checkNGroupsValid(yFitIC, ipData):
    """Checks number of groups selected for global fit is valid."""

    nSpectra = len(ipData)  # Number of spectra in the workspace

    if yFitIC.number_of_global_fit_groups == "all":
        yFitIC.number_of_global_fit_groups = nSpectra
    else:
        assert isinstance(yFitIC.number_of_global_fit_groups, int), "Number of global groups needs to be an integer."
        assert yFitIC.number_of_global_fit_groups <= nSpectra, (
            "Number of global groups needs to be less or equal to the no of unmasked spectra."
        )
        assert yFitIC.number_of_global_fit_groups > 0, "NUmber of global groups needs to be bigger than zero"
    return


def kMeansClustering(points, centers):
    """
    Algorithm used to form groups of detectors.
    Works best for spherical groups with similar scaling on x and y axis.
    Fails in some rare cases, solution is to try a different number of groups.
    Returns clusters in the form a int i assigned to each detector.
    Detectors with the same i assigned belong to the same group.
    """

    prevCenters = centers  # Starting centers
    while True:
        clusters = closestCenter(points, prevCenters)  # Form groups by assigning points to their closest center
        centers = calculateCenters(points, clusters)  # Recalculate centers of new groups

        if np.all(centers == prevCenters):
            break

        assert np.isfinite(centers).all(), f"Invalid centers found:\n{centers}\nTry a different number for the groupings."

        prevCenters = centers

    clusters = closestCenter(points, centers)
    return clusters


def closestCenter(points, centers):
    """
    Checks each point and assigns it to closest center.
    Each center is represented by an int i.
    Returns clusters with corresponding centers.
    """

    clusters = np.zeros(len(points))
    for p in range(len(points)):  # Iterate over each point
        distMin = np.inf  # To be replaced in first iteration

        for i in range(len(centers)):  # Assign closest center to point
            dist = pairDistance(points[p], centers[i])

            if dist < distMin:  # Store minimum found
                distMin = dist
                closeCenter = i

        clusters[p] = closeCenter
    return clusters


def pairDistance(p1, p2):
    "Calculates the distance between two points."
    return np.sqrt(np.sum(np.square(p1 - p2)))


def calculateCenters(points, clusters):
    """Calculates centers for the given clusters"""

    nGroups = len(np.unique(clusters))

    centers = np.zeros((nGroups, 2))
    for i in range(nGroups):
        centers[i, :] = np.mean(points[clusters == i, :], axis=0)  # If cluster i is not present, returns nan
    return centers


def formIdxList(clusters):
    """Converts assignment of clusters into a list of indexes."""

    idxList = []
    for i in np.unique(clusters):
        idxs = np.argwhere(clusters == i).flatten()
        idxList.append(list(idxs))

    # Print groupings information
    logger.notice("\nGroups formed successfully:\n")
    groupLen = np.array([len(group) for group in idxList])
    unique, counts = np.unique(groupLen, return_counts=True)
    for length, no in zip(unique, counts):
        logger.notice(f"{no} groups with {length} detectors.")

    return idxList


def plotDetsAndInitialCenters(L1, theta, centers):
    """Used in debugging."""
    fig, ax = plt.subplots(tight_layout=True, subplot_kw={"projection": PLOTS_PROJECTION})
    fig.canvas.manager.set_window_title("Starting centroids for groupings")
    ax.scatter(L1, theta, alpha=0.3, color="r", label="Detectors")
    ax.scatter(centers[:, 0], centers[:, 1], color="k", label="Starting centroids")
    ax.axes.xaxis.set_ticks([])  # Numbers plotted do not correspond to real numbers, so hide them
    ax.axes.yaxis.set_ticks([])
    ax.set_xlabel("L1")
    ax.set_ylabel("Theta")
    ax.legend()
    fig.show()


def plotFinalGroups(ax, ipData, idxList):
    """Plot of groupings of detectors."""

    for i, idxs in enumerate(idxList):
        L1 = ipData[idxs, -1]
        theta = ipData[idxs, 2]
        ax.scatter(L1, theta, label=f"Group {i}")

        dets = ipData[idxs, 0]
        for det, x, y in zip(dets, L1, theta):
            ax.text(x, y, str(int(det)), fontsize=8)

    ax.set_xlabel("L1")
    ax.set_ylabel("Theta")
    ax.legend()
    return


# --------- Weighted Avg of detectors


def avgWeightDetGroups(dataX, dataY, dataE, dataRes, idxList, yFitIC):
    """
    Performs weighted average on each detector group given by the index list.
    The imput arrays do not include masked spectra.
    """
    assert ~np.any(np.all(dataY == 0, axis=1)), (
        f"Input data should not include masked spectra at: {np.argwhere(np.all(dataY == 0, axis=1))}"
    )

    if yFitIC.mask_zeros_with == Masking.NAN:
        return avgGroupsWithBins(dataX, dataY, dataE, dataRes, idxList, yFitIC)

    # Use Default for unmasked or NCP masked
    return avgGroupsOverCols(dataX, dataY, dataE, dataRes, idxList)


def avgGroupsOverCols(dataX, dataY, dataE, dataRes, idxList):
    """
    Averaging used when JoY workspace is already Rebinned and Normalised.
    Selects groups of detectors and performs the weighted average for each group.
    Returns arrays with the group averages.
    """

    wDataX, wDataY, wDataE, wDataRes = initiateZeroArr((len(idxList), len(dataY[0])))

    for i, idxs in enumerate(idxList):
        groupX, groupY, groupE, groupRes = extractArrByIdx(dataX, dataY, dataE, dataRes, idxs)
        assert len(groupY) > 0, "Group with zero detectors found, invalid."

        if len(groupY) == 1:  # Cannot use weight avg in single spec, wrong results
            meanY, meanE = groupY, groupE
            meanRes = groupRes

        else:
            meanY, meanE = weightedAvgArr(groupY, groupE)
            meanRes = np.nanmean(groupRes, axis=0)  # Nans are not present but safeguard

        assert np.all(groupX[0] == np.mean(groupX, axis=0)), "X values should not change with groups"

        for wsData, mean in zip([wDataX, wDataY, wDataE, wDataRes], [groupX[0], meanY, meanE, meanRes]):
            wsData[i] = mean

    assert ~np.any(np.all(wDataY == 0, axis=1)), (
        f"Some avg weights in groups are not being performed:\n{np.argwhere(np.all(wDataY == 0, axis=1))}"
    )
    return wDataX, wDataY, wDataE, wDataRes


def weightedAvgArr(dataYO, dataEO):
    """
    Weighted average over columns of 2D arrays.
    Ignores any zero or NaN value.
    """

    # Run some tests
    assert dataYO.shape == dataEO.shape, "Y and E arrays should have same shape for weighted average."
    assert np.all((dataYO == 0) == (dataEO == 0)), (
        f"Masked zeros should match in DataY and DataE: {np.argwhere((dataYO == 0) != (dataEO == 0))}"
    )
    assert np.all(np.isnan(dataYO) == np.isnan(dataEO)), "Masked nans should match in DataY and DataE."
    assert len(dataYO) > 1, "Weighted average needs more than one element to be performed."

    dataY = dataYO.copy()  # Copy arrays not to change original data
    dataE = dataEO.copy()

    # Ignore invalid data by changing zeros to nans
    # If data is already masked with nans, it remains unaltered
    zerosMask = dataY == 0
    dataY[zerosMask] = np.nan
    dataE[zerosMask] = np.nan

    meanY = np.nansum(dataY / np.square(dataE), axis=0) / np.nansum(1 / np.square(dataE), axis=0)
    meanE = np.sqrt(1 / np.nansum(1 / np.square(dataE), axis=0))

    # Change invalid data back to original masking format with zeros
    nanInfMask = (meanE == np.inf) | (meanE == np.nan) | (meanY == np.nan)
    meanY[nanInfMask] = 0
    meanE[nanInfMask] = 0

    # Test that columns of zeros are left unchanged
    assert np.all((meanY == 0) == (meanE == 0)), "Weighted avg output should have masks in the same DataY and DataE."
    assert np.all((np.all(dataYO == 0, axis=0) | np.all(np.isnan(dataYO), axis=0)) == (meanY == 0)), "Masked cols should be ignored."

    return meanY, meanE


def avgGroupsWithBins(dataX, dataY, dataE, dataRes, idxList, yFitIC):
    """
    Performed only when mask with NaNs and Bins is turned on.
    Selection of groups is done as usual.
    Weighted average uses altered function to account for the unique format.
    Several dataY points correspond to each dataX point.
    """

    # Build range to average over
    rebin_pars = yFitIC.range_for_rebinning_in_y_space
    first, step, last = [float(s) for s in rebin_pars.split(",")]
    meanX = np.arange(first, last, step) + step / 2  # Correction to match Mantid range

    wDataX, wDataY, wDataE, wDataRes = initiateZeroArr((len(idxList), len(meanX)))
    for i, idxs in enumerate(idxList):
        groupX, groupY, groupE, groupRes = extractArrByIdx(dataX, dataY, dataE, dataRes, idxs)

        meanY, meanE = weightedAvgXBinsArr(groupX, groupY, groupE, meanX)

        meanRes = np.nanmean(groupRes, axis=0)  # Nans are not present but safeguard

        for wsData, mean in zip([wDataX, wDataY, wDataE, wDataRes], [meanX, meanY, meanE, meanRes]):
            wsData[i] = mean

    return wDataX, wDataY, wDataE, wDataRes


def weightedAvgXBins(wsXBins, xp):
    """Weighted average on ws where dataY points are grouped per dataX bin centers."""
    dataX, dataY, dataE = extractWS(wsXBins)

    meansY, meansE = weightedAvgXBinsArr(dataX, dataY, dataE, xp)

    wsYSpaceAvg = CreateWorkspace(
        DataX=xp,
        DataY=meansY,
        DataE=meansE,
        NSpec=1,
        OutputWorkspace=wsXBins.name() + "_wavg",
    )
    return wsYSpaceAvg


def weightedAvgXBinsArr(dataX, dataY, dataE, xp):
    """
    Weighted Average on arrays where several dataY points correspond to a single dataX point.
    xp is the range over which to perform the average.
    dataX points can only take values in xp.
    Ignores any zero or NaN value.
    """
    meansY = np.zeros(len(xp))
    meansE = np.zeros(len(xp))

    for i in range(len(xp)):
        # Perform weighted average over all dataY and dataE values with the same xp[i]
        # Change shape to column to match weighted average function
        pointMask = dataX == xp[i]
        allY = dataY[pointMask][:, np.newaxis]
        allE = dataE[pointMask][:, np.newaxis]

        # If no points were found for a given abcissae
        if np.sum(pointMask) == 0:
            mY, mE = 0, 0  # Mask with zeros

        # If one point was found, set to that point
        elif np.sum(pointMask) == 1:
            mY, mE = allY.flatten(), allE.flatten()

        # Weighted avg over all spectra and several points per spectra
        else:
            # Case of bootstrap replica with no errors
            if np.all(dataE == 0):
                mY = avgArr(allY)
                mE = 0

            # Default for most cases
            else:
                mY, mE = weightedAvgArr(allY, allE)  # Outputs masked values as zeros

        # DataY and DataE should never reach NaN, but safeguard in case they do
        if (mE == np.nan) | (mY == np.nan):
            mY, mE = 0, 0

        meansY[i] = mY
        meansE[i] = mE

    return meansY, meansE


def weightedAvgCols(wsYSpace):
    """Returns ws with weighted avg of columns of input ws"""
    dataX, dataY, dataE = extractWS(wsYSpace)
    if np.all(dataE == 0):  # Bootstrap case where errors are not used
        meanY = avgArr(dataY)
        meanE = np.zeros(meanY.shape)
    else:
        meanY, meanE = weightedAvgArr(dataY, dataE)
    wsYSpaceAvg = CreateWorkspace(
        DataX=dataX[0, :],
        DataY=meanY,
        DataE=meanE,
        NSpec=1,
        OutputWorkspace=wsYSpace.name() + "_wavg",
    )
    return wsYSpaceAvg


def avgArr(dataYO):
    """
    Average over columns of 2D dataY.
    Ignores any zero values as being masked.
    """

    assert len(dataYO) > 1, "Averaging needs more than one element."

    dataY = dataYO.copy()
    dataY[dataY == 0] = np.nan
    meanY = np.nanmean(dataY, axis=0)
    meanY[meanY == np.nan] = 0

    assert np.all(np.all(dataYO == 0, axis=0) == (meanY == 0)), "Columns of zeros should give zero."
    return meanY


def initiateZeroArr(shape):
    wDataX = np.zeros(shape)
    wDataY = np.zeros(shape)
    wDataE = np.zeros(shape)
    wDataRes = np.zeros(shape)
    return wDataX, wDataY, wDataE, wDataRes


def extractArrByIdx(dataX, dataY, dataE, dataRes, idxs):
    groupE = dataE[idxs, :]
    groupY = dataY[idxs, :]
    groupX = dataX[idxs, :]
    groupRes = dataRes[idxs, :]
    return groupX, groupY, groupE, groupRes


def calcCostFun(model, i, x, y, yerr, res, sharedPars):
    "Returns cost function for one spectrum i to be summed to total cost function"

    xDelta, resDense = oddPointsRes(x, res)

    def convolvedModel(xrange, y0, *pars):
        """Performs convolution first on high density grid and interpolates to desired x range"""
        return y0 + signal.convolve(model(xrange, *pars), resDense, mode="same") * xDelta

    signature = describe(model)[:]
    signature[1:1] = ["y0"]

    costSig = [key if key in sharedPars else key + str(i) for key in signature]
    convolvedModel._parameters = {key: None for key in costSig}

    # Select only valid data, i.e. when error is not 0 or nan or inf
    nonZeros = (yerr != 0) & (yerr != np.nan) & (yerr != np.inf) & (y != np.nan)
    xNZ = x[nonZeros]
    yNZ = y[nonZeros]
    yerrNZ = yerr[nonZeros]

    costFun = cost.LeastSquares(xNZ, yNZ, yerrNZ, convolvedModel)
    return costFun


def minuitInitialParameters(defaultPars, sharedPars, nSpec):
    """Buids dictionary to initialize Minuit with starting global+local parameters"""

    initPars = {}
    # Populate with initial shared parameters
    for sp in sharedPars:
        initPars[sp] = defaultPars[sp]
    # Add initial unshared parameters
    unsharedPars = [key for key in defaultPars if key not in sharedPars]
    for up in unsharedPars:
        for i in range(nSpec):
            initPars[up + str(i)] = defaultPars[up]
    return initPars


def create_table_for_global_fit_parameters(wsName, model, m, chi2):
    t = CreateEmptyTableWorkspace(OutputWorkspace=wsName + f"_globalfit_{model}_Parameters")
    t.setTitle("Global Fit Parameters")
    t.addColumn(type="str", name="Name")
    t.addColumn(type="float", name="Value")
    t.addColumn(type="float", name="Error")

    logger.notice(f"Value of Chi2/ndof: {chi2:.2f}")
    logger.notice(f"Migrad Minimum valid: {m.valid}")
    for p, v, e in zip(m.parameters, m.values, m.errors):
        t.addRow([p, v, e])

    t.addRow(["Cost function", chi2, 0])
    print_table_workspace(t)


def plot_global_fit(ws_group, yFitIC):
    individual_names = ws_group.getNames()
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(10, int(len(individual_names) / 3)),
        tight_layout=False,
        constrained_layout=False,
        subplot_kw={"projection": PLOTS_PROJECTION},
    )
    fig_title = ws_group.name() + "_results"
    fig.canvas.manager.set_window_title(fig_title)

    offset = 0
    vertical_spacing = 0.02
    for ws_name in individual_names:
        if ws_name.endswith("Parameters") or ws_name.endswith("chi2"):
            continue

        fit_ws = mtd[ws_name]

        if ws_name.endswith("sum"):
            axs[1].errorbar(fit_ws, wkspIndex=0, fmt="k.", markersize=5)
            axs[1].plot(fit_ws, "b.", wkspIndex=2, markersize=5)
            axs[1].plot(fit_ws, "r-", wkspIndex=1, linewidth=2)
            axs[1].set_title("Sum of groups and fits")
            continue

        y_data = fit_ws.extractY()[0, :]
        y_range = y_data.max() - y_data.min()

        temp_ws = fit_ws.clone(StoreInADS=False)
        temp_ws += offset

        axs[0].errorbar(temp_ws, wkspIndex=0, fmt="k.", markersize=3, elinewidth=1)
        axs[0].plot(temp_ws, "r-", wkspIndex=1, linewidth=2)

        x_data = fit_ws.extractX()[0, :]
        x_range = x_data.max() - x_data.min()
        axs[0].text(x_data.max() + x_range * 0.01, offset + y_range / 3, "Group " + ws_name.split("_")[-1], fontsize=9)

        offset += y_data.max() + vertical_spacing

    savePath = yFitIC.figSavePath / fig_title
    plt.savefig(savePath, bbox_inches="tight")
    fig.show()
    return fig  # Return figure for testing purposes


def save_result_of_global_fit(data_x, data_y, data_e, m, total_cost_fun, ws_name, fit_model, chi2):
    global_sum_name = ws_name + "_global_fit_sum"
    individual_names = []

    # Create table with only chi2
    chi2_table_name = ws_name + "_global_fit_" + fit_model + "_chi2"
    chi2_table = CreateEmptyTableWorkspace(OutputWorkspace=chi2_table_name)
    chi2_table.setTitle("Global Fit Chi2")
    chi2_table.addColumn(type="float", name="Chi2")
    chi2_table.addRow([chi2])

    # Create table of parameters
    pars_table_name = ws_name + "_global_fit_" + fit_model + "_Parameters"
    pars_table = CreateEmptyTableWorkspace(OutputWorkspace=pars_table_name)
    pars_table.setTitle("Global Fit Parameters")
    pars_table.addColumn(type="int", name="Group")
    signature = describe(total_cost_fun[0])
    for parameter in signature:
        parameter = parameter[:-1] if parameter.endswith("0") else parameter
        pars_table.addColumn(type="str", name=parameter)

    for i, (x, y, yerr, cost_fun) in enumerate(zip(data_x, data_y, data_e, total_cost_fun)):
        signature = describe(cost_fun)
        values = m.values[signature]

        yfit = cost_fun.model(x, *values)
        res = y - yfit

        CreateWorkspace(
            DataX=np.concatenate([x, x, x]),
            DataY=np.concatenate([y, yfit, res]),
            DataE=np.concatenate([yerr, np.zeros_like(yerr), np.zeros_like(yerr)]),
            Nspec=3,
            OutputWorkspace=ws_name + f"_global_fit_{i}",
            Distribution=True,
        )
        individual_names.append(ws_name + f"_global_fit_{i}")

        if i == 1:
            Plus(ws_name + "_global_fit_0", ws_name + "_global_fit_1", OutputWorkspace=global_sum_name)
        elif i > 1:
            Plus(global_sum_name, ws_name + f"_global_fit_{i}", OutputWorkspace=global_sum_name)

        # Build strings for par values
        errors = m.errors[signature]
        pars_table.addRow([i] + [f" {v:.3f} +/- {e:.3f} " for v, e in zip(values, errors)])

    individual_names.append(global_sum_name)
    individual_names.append(pars_table_name)
    individual_names.append(chi2_table_name)

    return GroupWorkspaces(individual_names, OutputWorkspace=ws_name + "_global_fit_group")


def save_workspaces(yFitIC):
    for ws_name in mtd.getObjectNames():
        save_path = yFitIC.outputs_dir / f"{yFitIC.fitting_model}_fit" / ws_name
        if ws_name.endswith("Parameters") or ws_name.endswith("CovarianceMatrix"):
            save_path.parent.mkdir(exist_ok=True, parents=True)
            SaveAscii(ws_name, str(save_path))
        if ws_name.endswith("Workspace"):
            save_path.parent.mkdir(exist_ok=True, parents=True)
            ws_pars = mtd[ws_name.replace("Workspace", "Parameters")]
            lab = ""
            for p, v, e in zip(ws_pars.column("Name"), ws_pars.column("Value"), ws_pars.column("Error")):
                if p.startswith("Cost"):
                    break
                lab += f"{p.split('.')[-1]}={v:.2f} $\\pm$ {e:.2f}\n"
            SaveAscii(ws_name, str(save_path), LogList=["Weighted Avg", lab, "Residuals"])
