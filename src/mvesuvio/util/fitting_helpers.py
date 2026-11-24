import numpy as np
from mantid.simpleapi import (
    AppendSpectra,
    CloneWorkspace,
    DeleteWorkspace,
    MaskDetectors,
    Rebin,
    RenameWorkspace,
    SumSpectra,
    VesuvioResolution,
    mtd,
)


def isolate_lighest_mass_data(initial_ws, ws_group_ncp, subtract_fse=True):
    # NOTE: Minus() is not used so it does not change dataE.

    ws_ncp_names = [n for n in ws_group_ncp.getNames() if n.endswith("ncp")]
    masses = [float(n.split("_")[-2]) for n in ws_ncp_names if "total" not in n]
    ws_total_ncp_name = next(n for n in ws_ncp_names if n.endswith("_total_ncp"))
    ws_lighest_ncp_name = ws_ncp_names[masses.index(min(masses))]
    ws_lighest_ncp = mtd[ws_lighest_ncp_name]
    ws_total_ncp = mtd[ws_total_ncp_name]
    suffix = "_m0"

    # Main subtraction.
    isolated_data_y = initial_ws.extractY() - (ws_total_ncp.extractY() - ws_lighest_ncp.extractY())

    if subtract_fse:
        suffix += "_-fse"
        ws_lighest_fse = mtd[ws_lighest_ncp_name.replace("ncp", "fse")]

        isolated_data_y -= ws_lighest_fse.extractY()

        # Subtract from fitted ncp.
        ws_lighest_ncp_y = ws_lighest_ncp.extractY()
        ws_lighest_ncp_y -= ws_lighest_fse.extractY()
        ws_lighest_ncp = CloneWorkspace(ws_lighest_ncp, OutputWorkspace=ws_lighest_ncp.name() + "_-fse")
        _write_data_y_into_ws(ws_lighest_ncp_y, ws_lighest_ncp)
        SumSpectra(ws_lighest_ncp.name(), OutputWorkspace=ws_lighest_ncp.name() + "_sum")

    # Preserve masked values.
    isolated_data_y[initial_ws.extractY() == 0] = 0
    ws_lighest_data = CloneWorkspace(initial_ws, OutputWorkspace=initial_ws.name() + suffix)
    _write_data_y_into_ws(isolated_data_y, ws_lighest_data)
    SumSpectra(ws_lighest_data.name(), OutputWorkspace=ws_lighest_data.name() + "_sum")

    return ws_lighest_data, ws_lighest_ncp


def calculate_resolution(mass, ws, rebin_range):
    resolution_name = ws.name() + "_resolution"
    for index in range(ws.getNumberHistograms()):
        VesuvioResolution(Workspace=ws, WorkspaceIndex=index, Mass=mass, OutputWorkspaceYSpace="tmp")
        Rebin(
            InputWorkspace="tmp",
            Params=rebin_range,
            OutputWorkspace="tmp",
        )

        if index == 0:  # Ensures that workspace has desired units.
            RenameWorkspace("tmp", resolution_name)
        else:
            AppendSpectra(resolution_name, "tmp", OutputWorkspace=resolution_name)

    masked_idx = [ws.spectrumInfo().isMasked(i) for i in range(ws.getNumberHistograms())]
    MaskDetectors(resolution_name, WorkspaceIndexList=np.flatnonzero(masked_idx))
    DeleteWorkspace("tmp")
    return mtd[resolution_name]


def _write_data_y_into_ws(data_y, ws):
    for i in range(ws.getNumberHistograms()):
        ws.dataY(i)[:] = data_y[i, :]
