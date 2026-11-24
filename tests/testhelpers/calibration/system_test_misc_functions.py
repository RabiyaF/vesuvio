from sys import maxsize

import numpy as np

from tests.testhelpers.calibration.system_test_base import TestConstants


def assert_allclose_excluding_bad_detectors(expected_position, position, rtol, default_rtol=TestConstants.DEFAULT_RELATIVE_TOLERANCE):
    np.set_printoptions(threshold=maxsize)
    test_failures = []
    for i, (elem_m, elem_n) in enumerate(zip(expected_position, position)):
        if np.ma.is_masked(elem_m) and np.ma.is_masked(elem_n):  # detector masked
            break
        detector_specific_rtol = rtol.get(i, default_rtol)
        try:
            np.testing.assert_allclose(elem_m, elem_n, detector_specific_rtol, atol=0)
        except AssertionError:
            test_failures.append(
                f"Element {i}: Expected {elem_m}, found {elem_n}. atol "
                f"{abs(elem_n - elem_m)}, rtol {abs(elem_n - elem_m) / elem_n},"
                f"max tol: {detector_specific_rtol}"
            )
    return test_failures
