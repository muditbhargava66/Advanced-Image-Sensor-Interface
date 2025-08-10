"""Type stubs for scikit-image (skimage) library."""

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

# Common type aliases
ArrayLike = Union[NDArray[Any], list, tuple]
ImageArray = NDArray[np.uint8]
FloatArray = NDArray[np.floating[Any]]

def __getattr__(name: str) -> Any: ...

# Submodules
