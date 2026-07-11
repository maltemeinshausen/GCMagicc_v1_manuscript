"""
IPCC AR6 Colormaps - Centralized colormap definitions

This module provides access to IPCC AR6 approved colormaps for climate data visualization.
Based on the official IPCC AR6 Visual Style Guide and colormaps repository:
https://github.com/IPCC-WG1/colormaps/

Usage:
    from .helper_ipcc_colormaps import get_divergent_colormap, get_continuous_colormap

    # For temperature deviations (divergent)
    cmap = get_divergent_colormap('tas')

    # For precipitation (continuous)
    cmap = get_continuous_colormap('pr')
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def create_ipcc_misc_div_colormap():
    """
    Create the IPCC misc_div custom colormap from the official repository.
    This is a custom divergent colormap with 11 colors.
    """
    # IPCC misc_div colormap colors (RGB values 0-255, converted to 0-1)
    misc_div_colors = [
        [8 / 255, 29 / 255, 88 / 255],  # Dark blue
        [35 / 255, 77 / 255, 160 / 255],  # Blue
        [36 / 255, 152 / 255, 192 / 255],  # Light blue
        [115 / 255, 200 / 255, 188 / 255],  # Cyan
        [214 / 255, 239 / 255, 178 / 255],  # Light green
        [254 / 255, 254 / 255, 209 / 255],  # Light yellow
        [254 / 255, 225 / 255, 135 / 255],  # Yellow
        [253 / 255, 170 / 255, 72 / 255],  # Orange
        [252 / 255, 90 / 255, 45 / 255],  # Red-orange
        [211 / 255, 15 / 255, 31 / 255],  # Red
        [128 / 255, 0 / 255, 38 / 255],  # Dark red
    ]

    # Create and return the colormap
    return mcolors.LinearSegmentedColormap.from_list("ipcc_misc_div", misc_div_colors, N=256)


# IPCC AR6 Divergent Colormaps (symmetric around zero)
DIVERGENT_COLORMAPS = {
    "temperature": "RdBu_r",  # Red-Blue reversed for temperature deviations
    "pressure": create_ipcc_misc_div_colormap(),  # Custom IPCC misc_div for pressure deviations
    "wind": "PuOr",  # Purple-Orange for wind deviations
    "precipitation": "BrBG",  # Brown-Green for precipitation deviations
    "other": "RdBu_r",  # Default divergent for other variables
}

# IPCC AR6 Continuous Colormaps (for absolute values)
CONTINUOUS_COLORMAPS = {
    "temperature": "viridis",  # Sequential for temperature magnitudes
    "pressure": "plasma",  # Sequential for pressure magnitudes
    "wind": "inferno",  # Sequential for wind magnitudes
    "precipitation": "cividis",  # Sequential for precipitation magnitudes
    "other": "viridis",  # Default continuous for other variables
}

# Variable type mappings
VARIABLE_TYPES = {
    "tas": "temperature",
    "temp": "temperature",
    "temperature": "temperature",
    "pr": "precipitation",
    "precip": "precipitation",
    "precipitation": "precipitation",
    "sfcWind": "wind",
    "wind": "wind",
    "psl": "pressure",
    "pressure": "pressure",
    "slp": "pressure",
}


def get_variable_type(var_name):
    """
    Determine the variable type for colormap selection.

    Parameters
    ----------
    var_name : str
        Variable name (e.g., 'tas', 'pr', 'psl', 'sfcWind')

    Returns
    -------
    str
        Variable type for colormap selection
    """
    var_lower = var_name.lower()

    # Check exact matches first
    if var_lower in VARIABLE_TYPES:
        return VARIABLE_TYPES[var_lower]

    # Check partial matches
    for key, value in VARIABLE_TYPES.items():
        if key in var_lower or var_lower in key:
            return value

    # Default to 'other' if no match found
    return "other"


def get_divergent_colormap(var_name):
    """
    Get appropriate IPCC AR6 divergent colormap for a variable.

    Parameters
    ----------
    var_name : str
        Variable name (e.g., 'tas', 'pr', 'psl', 'sfcWind')

    Returns
    -------
    str
        Colormap name for matplotlib
    """
    var_type = get_variable_type(var_name)
    return DIVERGENT_COLORMAPS.get(var_type, DIVERGENT_COLORMAPS["other"])


def get_continuous_colormap(var_name):
    """
    Get appropriate IPCC AR6 continuous colormap for a variable.

    Parameters
    ----------
    var_name : str
        Variable name (e.g., 'tas', 'pr', 'psl', 'sfcWind')

    Returns
    -------
    str
        Colormap name for matplotlib
    """
    var_type = get_variable_type(var_name)
    return CONTINUOUS_COLORMAPS.get(var_type, CONTINUOUS_COLORMAPS["other"])


def get_variable_units(var_name):
    """
    Get standard units for climate variables.

    Parameters
    ----------
    var_name : str
        Variable name (e.g., 'tas', 'pr', 'psl', 'sfcWind')

    Returns
    -------
    str
        Unit string for the variable
    """
    units_map = {
        "tas": "K",
        "temp": "K",
        "temperature": "K",
        "pr": "mm day⁻¹",
        "precip": "mm day⁻¹",
        "precipitation": "mm day⁻¹",
        "psl": "hPa",
        "pressure": "hPa",
        "slp": "hPa",
        "sfcWind": "m s⁻¹",
        "wind": "m s⁻¹",
    }

    var_lower = var_name.lower()

    # Check exact matches first
    if var_lower in units_map:
        return units_map[var_lower]

    # Check partial matches
    for key, value in units_map.items():
        if key in var_lower or var_lower in key:
            return value

    # Default to empty string if no match found
    return ""


def create_symmetric_colormap(cmap_name, center=0.0):
    """
    Create a symmetric colormap centered on a specific value.

    Parameters
    ----------
    cmap_name : str
        Base colormap name
    center : float
        Center value for the colormap (default: 0.0)

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Symmetric colormap
    """
    cmap = plt.cm.get_cmap(cmap_name)

    # Create symmetric colormap
    colors = cmap(np.linspace(0, 1, 256))

    # Create new colormap
    new_cmap = mcolors.LinearSegmentedColormap.from_list(f"{cmap_name}_symmetric", colors)

    return new_cmap


def get_colormap_bounds(data, center=None, symmetric=True):
    """
    Calculate appropriate colormap bounds for data.

    Parameters
    ----------
    data : numpy.ndarray
        Data array
    center : float, optional
        Center value for symmetric colormap (default: 0.0)
    symmetric : bool
        Whether to create symmetric bounds around center

    Returns
    -------
    tuple
        (vmin, vmax) bounds for colormap
    """
    if center is None:
        center = 0.0

    if symmetric:
        max_abs_val = np.nanmax(np.abs(data))
        return center - max_abs_val, center + max_abs_val
    else:
        return np.nanmin(data), np.nanmax(data)
