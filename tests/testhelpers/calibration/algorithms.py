# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +

import mantid  # noqa


def create_algorithm(name, **kwargs):
    """Create a named algorithm, set the properties given by the keywords and return the
    algorithm handle WITHOUT executing the algorithm
    Useful keywords:
      - child: Makes algorithm a child algorithm
      - rethrow: Causes exceptions to be rethrown on execution
    Parameters:
        name - The name of the algorithm
        kwargs - A dictionary of property name:value pairs
    @returns The algorithm handle
    """
    # Initialize the whole framework
    import mantid.simpleapi

    if "Version" in kwargs:
        alg = mantid.api.AlgorithmManager.createUnmanaged(name, kwargs["Version"])
        del kwargs["Version"]
    else:
        alg = mantid.api.AlgorithmManager.createUnmanaged(name)
    alg.initialize()
    # Avoid problem that Load needs to set Filename first if it exists
    if name == "Load" and "Filename" in kwargs:
        alg.setPropertyValue("Filename", kwargs["Filename"])
        del kwargs["Filename"]
    if "child" in kwargs:
        alg.setChild(True)
        del kwargs["child"]
        if "OutputWorkspace" in alg:
            alg.setPropertyValue("OutputWorkspace", "UNUSED_NAME_FOR_CHILD")
    if "rethrow" in kwargs:
        alg.setRethrows(True)
        del kwargs["rethrow"]
    alg.setProperties(kwargs)
    return alg
