"""Type stubs for skimage.filters module."""

from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

def gaussian(
    image: NDArray[Any],
    sigma: Union[float, tuple[float, ...]] = ...,
    output: Optional[NDArray[Any]] = ...,
    mode: str = ...,
    cval: float = ...,
    multichannel: Optional[bool] = ...,
    preserve_range: bool = ...,
    truncate: float = ...,
) -> NDArray[Any]: ...
def median(
    image: NDArray[Any],
    disk: Optional[NDArray[Any]] = ...,
    out: Optional[NDArray[Any]] = ...,
    mask: Optional[NDArray[Any]] = ...,
    shift_x: bool = ...,
    shift_y: bool = ...,
) -> NDArray[Any]: ...
def sobel(image: NDArray[Any], mask: Optional[NDArray[Any]] = ...) -> NDArray[np.floating[Any]]: ...
def sobel_h(image: NDArray[Any], mask: Optional[NDArray[Any]] = ...) -> NDArray[np.floating[Any]]: ...
def sobel_v(image: NDArray[Any], mask: Optional[NDArray[Any]] = ...) -> NDArray[np.floating[Any]]: ...
def threshold_otsu(image: NDArray[Any], nbins: int = ...) -> float: ...
def __getattr__(name: str) -> Any: ...
