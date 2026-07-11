"""
Global font configuration for all segment modules.
Sets up Carlito font for matplotlib plots including titles, labels, tickmarks, and legends.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Optional

# Global font configuration
_carlito_font = None


def setup_carlito_font() -> Optional[fm.FontProperties]:
    """
    Set up Carlito font for matplotlib plots.
    Returns the font properties object if successful, None otherwise.
    """
    global _carlito_font

    if _carlito_font is not None:
        return _carlito_font

    try:
        # Try to find Carlito font on common locations first.
        candidate_paths = [
            "/usr/share/fonts/google-carlito-fonts/Carlito-Regular.ttf",
            "/System/Library/Fonts/Carlito.ttf",  # macOS
            "/usr/share/fonts/truetype/carlito/Carlito-Regular.ttf",  # Alternative Linux path
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                _carlito_font = fm.FontProperties(fname=path)
                break

        # Fall back to a system-registered Carlito only when it is truly available.
        if _carlito_font is None:
            try:
                names = {f.name for f in fm.fontManager.ttflist}
                if "Carlito" in names:
                    _carlito_font = fm.FontProperties(family="Carlito")
            except Exception:
                _carlito_font = None

        # Apply global font settings. If Carlito is unavailable, avoid forcing it and
        # use DejaVu Sans to prevent repetitive "findfont ... not found" spam.
        plt.rcParams["font.family"] = "sans-serif"
        if _carlito_font is not None:
            sans = [f for f in plt.rcParams.get("font.sans-serif", []) if str(f).lower() != "carlito"]
            plt.rcParams["font.sans-serif"] = ["Carlito"] + sans
        else:
            sans = [f for f in plt.rcParams.get("font.sans-serif", []) if str(f).lower() != "carlito"]
            if "DejaVu Sans" not in sans:
                sans = ["DejaVu Sans"] + sans
            plt.rcParams["font.sans-serif"] = sans

        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams["axes.labelsize"] = 10
        plt.rcParams["xtick.labelsize"] = 9
        plt.rcParams["ytick.labelsize"] = 9
        plt.rcParams["legend.fontsize"] = 7
        plt.rcParams["figure.titlesize"] = 14

        return _carlito_font

    except Exception as e:
        print(f"Warning: Could not set up Carlito font: {e}")
        return None


def get_carlito_font() -> Optional[fm.FontProperties]:
    """Get the Carlito font properties object."""
    if _carlito_font is None:
        return setup_carlito_font()
    return _carlito_font


def add_bold_title(fig, title: str, fontsize: int = 16, x: float = 0.05, y: float = 0.95):
    """
    Add a bold title to the top-left corner of a figure.

    Args:
        fig: matplotlib figure object
        title: title text
        fontsize: font size for the title
        x: x position (0-1, left to right)
        y: y position (0-1, bottom to top)
    """
    carlito_font = get_carlito_font()

    if carlito_font:
        fig.text(
            x,
            y,
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontproperties=carlito_font,
            ha="left",
            va="top",
            transform=fig.transFigure,
        )
    else:
        # Fallback to default font
        fig.text(
            x,
            y,
            title,
            fontsize=fontsize,
            fontweight="bold",
            ha="left",
            va="top",
            transform=fig.transFigure,
        )


# Variable lookup dictionary for names and units
VARIABLE_INFO = {
    "tas": {"name": "Temperature", "unit": "degC"},
    "pr": {"name": "Precipitation", "unit": "mm/day"},
    "psl": {"name": "Sea Level Pressure", "unit": "hPa"},
    "ua": {"name": "Zonal Wind", "unit": "m/s"},
    "va": {"name": "Meridional Wind", "unit": "m/s"},
    "zg": {"name": "Geopotential Height", "unit": "m"},
    "hus": {"name": "Specific Humidity", "unit": "kg/kg"},
    "clt": {"name": "Total Cloud Cover", "unit": "%"},
    "clw": {"name": "Liquid Water Path", "unit": "kg/m^2"},
    "cli": {"name": "Ice Water Path", "unit": "kg/m^2"},
    "ts": {"name": "Surface Temperature", "unit": "degC"},
    "tos": {"name": "Sea Surface Temperature", "unit": "degC"},
    "sst": {"name": "Sea Surface Temperature", "unit": "degC"},
    "sic": {"name": "Sea Ice Concentration", "unit": "%"},
    "sit": {"name": "Sea Ice Thickness", "unit": "m"},
    "snd": {"name": "Snow Depth", "unit": "m"},
    "mrlsl": {"name": "Land Sea Mask", "unit": ""},
    "orog": {"name": "Orography", "unit": "m"},
    "sftlf": {"name": "Land Fraction", "unit": "%"},
    "sftof": {"name": "Ocean Fraction", "unit": "%"},
}


def get_variable_info(var: str) -> dict:
    """Get variable name and unit for a given variable code."""
    return VARIABLE_INFO.get(var.lower(), {"name": var.upper(), "unit": ""})


def add_timestamp(fig, fontsize: int = 5):
    """Add current timestamp to bottom-right corner of figure."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%d %b %Y, %I:%M%p").lower()
    fig.text(
        0.98,
        0.02,
        timestamp,
        transform=fig.transFigure,
        fontsize=fontsize,
        color="grey",
        ha="right",
        va="bottom",
    )


# Recipe-specific titles
SEGMENT_TITLES = {
    "BiasMaps": "Bias Maps",
    "IndicatorFrequencies": "Climate Indicator Frequencies",
    "SPEI40": "Standardized Precipitation Evapotranspiration Index (40-Month)",
    "SPI6": "Standardized Precipitation Index (6-Month)",
    "GlobalTimeseries": "Global-Mean Timeseries",
    "ENSOTeleconnections": "ENSO Teleconnection Patterns",
    "PalmerDroughtSeverityIndexProxy": "Palmer Drought Severity Index Proxy",
    "DroughtSeverityIndex": "Drought Severity Index",
    "StandPrecipIndex": "Standardized Precipitation Index",
    "DrySpellFrequency": "Dry Spell Frequency",
    "ConsecutiveDryMonths": "Consecutive Dry Months",
    "PCMDITrends": "PCMDI Trend",
    "PCMDIZonalMeans": "PCMDI Zonal Means",
    "PCMDIVariability": "PCMDI Variability",
    "PCMDITemperatureRange": "PCMDI Temperature Range",
    "PCMDISeasonalCycle": "PCMDI Seasonal Cycle",
    "PCMDITeleconnections": "PCMDI Teleconnections",
    "PCMDIPrecipExtremes": "PCMDI Precipitation Extremes",
    "PCMDIHumidityCoupling": "PCMDI Humidity Coupling",
    "PCMDIClimatology": "PCMDI Climatology",
    "JointEvolution": "Joint Evolution",
    "Histograms": "Histogram",
    "CLIVARENSOCharacteristics": "CLIVAR ENSO Characteristics",
    "SimpleExampleRecipe": "Simple Example",
}


def get_segment_title(segment_name: str) -> str:
    """Get the appropriate title for a given segment."""
    return SEGMENT_TITLES.get(segment_name, f"{segment_name}")
