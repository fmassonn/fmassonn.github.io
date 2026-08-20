"""
Download, process and plot ERA5 2-m air temperature at point locations.

Author
------
François Massonnet

Original version
----------------
3 October 2022

Major rewrite
-------------
August 2026, assisted by ChatGPT

Description
-----------
The script retrieves hourly ERA5 2-m air temperature for one or several
point locations, stores the data in a local cache, computes daily and
climatological statistics, optionally appends recent KMI/IRM observations
for Uccle, and produces CSV files and diagnostic figures.

ERA5 data source
----------------
Copernicus Climate Data Store:
    reanalysis-era5-single-levels-timeseries

The time-series product is optimized for retrieving long ERA5 series
at a single geographical point.

ERA5 CSV downloads are delivered by the CDS as ZIP archives. The CSV file
inside the archive is read directly into pandas without being extracted
permanently to disk.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

import cdsapi
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import Normalize


# ============================================================================
# Configuration
# ============================================================================

ERA5_DATASET = "reanalysis-era5-single-levels-timeseries"
ERA5_VARIABLE = "2m_temperature"

START_DATE = date(1940, 1, 1)

# ERA5 is normally available with a latency of about five days.
ERA5_LAG_DAYS = 5

CLIMATOLOGY_START = 1991
CLIMATOLOGY_END = 2020

# Re-download this recent period on every run.
REFRESH_DAYS = 120

CLIMATOLOGY_SMOOTHING_DAYS = 61

KMI_URL = (
    "https://www.meteo.be/resources/"
    "climatology/uccle_month/Uccle_observations.txt"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = BASE_DIR / "figures"

for directory in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Locations
# ============================================================================


@dataclass(frozen=True)
class Location:
    """A geographical point for ERA5 extraction."""

    name: str
    latitude: float
    longitude: float
    add_kmi_latest: bool = False


LOCATIONS = [
    Location(
        name="Bruxelles",
        latitude=50.85,
        longitude=4.35,
        add_kmi_latest=True,
    ),

    # Example:
    #
    # Location(
    #     name="Sigy-le-Châtel",
    #     latitude=46.56,
    #     longitude=4.57,
    # ),
]


# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ============================================================================
# ERA5 download
# ============================================================================


def download_era5(
    location: Location,
    start_date: date,
    end_date: date,
    target: Path,
) -> None:
    """
    Download hourly ERA5 2-m temperature for one point.

    The CDS returns CSV data inside a ZIP archive.

    Parameters
    ----------
    location
        Geographic location.

    start_date, end_date
        Requested date range, inclusive.

    target
        Destination ZIP file.
    """

    request = {
        "variable": [ERA5_VARIABLE],
        "date": [
            start_date.isoformat(),
            end_date.isoformat(),
        ],
        "location": {
            "longitude": location.longitude,
            "latitude": location.latitude,
        },
        "data_format": "csv",
    }

    logger.info(
        "Downloading ERA5 for %s: %s -> %s",
        location.name,
        start_date,
        end_date,
    )

    client = cdsapi.Client()

    client.retrieve(
        ERA5_DATASET,
        request,
        str(target),
    )


# ============================================================================
# ERA5 CSV reader
# ============================================================================


def read_era5_csv_zip(path: Path) -> pd.DataFrame:
    """
    Read ERA5 hourly 2-m temperature from a CDS ZIP archive.

    Expected CSV columns include:

        valid_time
        latitude
        longitude
        t2m

    ``t2m`` is stored in kelvins and converted here to °C.

    Parameters
    ----------
    path
        Path to ZIP archive returned by the CDS.

    Returns
    -------
    pandas.DataFrame
        UTC DatetimeIndex and one column named ``temperature`` in °C.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"ERA5 archive does not exist: {path}"
        )

    if not zipfile.is_zipfile(path):
        raise ValueError(
            f"Downloaded ERA5 file is not a valid ZIP archive: {path}"
        )

    with zipfile.ZipFile(path, "r") as archive:

        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise ValueError(
                f"No CSV file found in ERA5 archive {path}."
            )

        if len(csv_files) > 1:
            logger.warning(
                "Several CSV files found in ERA5 archive: %s. "
                "Using %s.",
                csv_files,
                csv_files[0],
            )

        csv_name = csv_files[0]

        logger.info(
            "Reading ERA5 CSV: %s",
            csv_name,
        )

        with archive.open(csv_name) as csv_file:
            df = pd.read_csv(csv_file)

    required_columns = {
        "valid_time",
        "t2m",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Unexpected ERA5 CSV format.\n"
            f"Missing columns: {sorted(missing_columns)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["valid_time"] = pd.to_datetime(
        df["valid_time"],
        utc=True,
        errors="coerce",
    )

    if df["valid_time"].isna().any():
        n_invalid = df["valid_time"].isna().sum()

        raise ValueError(
            f"{n_invalid} ERA5 timestamps could not be parsed."
        )

    temperature_kelvin = pd.to_numeric(
        df["t2m"],
        errors="coerce",
    )

    if temperature_kelvin.isna().any():
        logger.warning(
            "%d ERA5 temperature values could not be converted to numbers.",
            temperature_kelvin.isna().sum(),
        )

    df["temperature"] = (
        temperature_kelvin - 273.15
    )

    result = (
        df[["valid_time", "temperature"]]
        .set_index("valid_time")
        .sort_index()
    )

    result.index.name = "time"

    return result


# ============================================================================
# ERA5 cache
# ============================================================================


def cache_path(location: Location) -> Path:
    """Return local Parquet cache path for a location."""

    safe_name = location.name.replace(" ", "_")

    return (
        DATA_DIR
        / f"ERA5_T2M_{safe_name}.parquet"
    )


def update_era5_cache(
    location: Location,
) -> pd.DataFrame:
    """
    Update local ERA5 cache and return the complete hourly series.

    First execution
    ---------------
    Download START_DATE -> latest expected ERA5 date.

    Subsequent executions
    ---------------------
    Re-download the latest REFRESH_DAYS and overwrite that overlap
    in the cache.
    """

    path = cache_path(location)

    latest_era5_date = (
        date.today()
        - timedelta(days=ERA5_LAG_DAYS)
    )

    # ------------------------------------------------------------------
    # Existing cache
    # ------------------------------------------------------------------

    if path.exists():

        logger.info(
            "Reading ERA5 cache %s",
            path,
        )

        cached = pd.read_parquet(path)

        if cached.index.tz is None:
            cached.index = (
                cached.index.tz_localize("UTC")
            )

        last_cached_date = (
            cached.index.max().date()
        )

        download_start = max(
            START_DATE,
            last_cached_date
            - timedelta(days=REFRESH_DAYS),
        )

    # ------------------------------------------------------------------
    # First execution
    # ------------------------------------------------------------------

    else:

        logger.info(
            "No ERA5 cache found for %s",
            location.name,
        )

        cached = pd.DataFrame(
            columns=["temperature"],
            index=pd.DatetimeIndex(
                [],
                tz="UTC",
                name="time",
            ),
        )

        download_start = START_DATE

    # ------------------------------------------------------------------
    # Anything to download?
    # ------------------------------------------------------------------

    if download_start > latest_era5_date:

        logger.info(
            "ERA5 cache already contains all currently expected data."
        )

        check_hourly_data(cached)

        return cached

    # ------------------------------------------------------------------
    # Temporary ZIP archive
    # ------------------------------------------------------------------

    with NamedTemporaryFile(
        suffix=".zip",
        delete=False,
        dir=DATA_DIR,
    ) as handle:

        temporary_path = Path(handle.name)

    try:

        download_era5(
            location=location,
            start_date=download_start,
            end_date=latest_era5_date,
            target=temporary_path,
        )

        new_data = read_era5_csv_zip(
            temporary_path
        )

    finally:

        temporary_path.unlink(
            missing_ok=True
        )

    # ------------------------------------------------------------------
    # Merge cache + freshly downloaded data
    # ------------------------------------------------------------------

    if cached.empty:
        combined = new_data.copy()

    else:
        combined = pd.concat(
            [
                cached,
                new_data,
            ]
        )

    combined = (
        combined
        .loc[
            ~combined.index.duplicated(
                keep="last"
            )
        ]
        .sort_index()
    )

    # ------------------------------------------------------------------
    # Quality control
    # ------------------------------------------------------------------

    check_hourly_data(combined)

    # ------------------------------------------------------------------
    # Save cache
    # ------------------------------------------------------------------

    combined.to_parquet(path)

    logger.info(
        "ERA5 cache contains %d hourly values (%s -> %s)",
        len(combined),
        combined.index.min(),
        combined.index.max(),
    )

    return combined


# ============================================================================
# Quality control
# ============================================================================


def check_hourly_data(
    df: pd.DataFrame,
) -> None:
    """
    Perform basic QC checks on the hourly ERA5 series.
    """

    if df.empty:
        raise ValueError(
            "ERA5 dataset is empty."
        )

    if df.index.has_duplicates:
        raise ValueError(
            "Duplicate timestamps found in ERA5 data."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "ERA5 timestamps are not sorted."
        )

    expected = pd.date_range(
        df.index.min(),
        df.index.max(),
        freq="1h",
        tz="UTC",
    )

    missing_times = expected.difference(
        df.index
    )

    if len(missing_times):

        logger.warning(
            "%d hourly timestamps are missing from the ERA5 series.",
            len(missing_times),
        )

    suspicious = (
        df["temperature"].notna()
        & ~df["temperature"].between(-90, 60)
    )

    if suspicious.any():

        logger.warning(
            "%d physically suspicious temperature values detected.",
            suspicious.sum(),
        )


# ============================================================================
# Daily statistics
# ============================================================================


def compute_daily_statistics(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily ERA5 mean, minimum and maximum temperatures.

    Days are defined from 00:00 to 23:00 UTC.
    """

    logger.info(
        "Computing daily statistics."
    )

    daily = (
        hourly["temperature"]
        .resample("1D")
        .agg(
            mean="mean",
            min="min",
            max="max",
            count="count",
        )
    )

    incomplete = (
        daily["count"] != 24
    )

    if incomplete.any():

        logger.warning(
            "%d incomplete ERA5 days detected.",
            incomplete.sum(),
        )

    daily["source"] = "ERA5"

    return daily


# ============================================================================
# KMI / IRM observations
# ============================================================================


def download_kmi_uccle() -> pd.DataFrame:
    """
    Download recent daily observations from KMI/IRM Uccle.

    Returns columns:
        mean
        min
        max
        count
        source
    """

    logger.info(
        "Downloading recent KMI/IRM observations for Uccle."
    )

    response = requests.get(
        KMI_URL,
        timeout=30,
    )

    response.raise_for_status()

    rows = []

    for line in response.text.splitlines():

        fields = line.split()

        if not fields:
            continue

        try:

            timestamp = pd.to_datetime(
                fields[0],
                format="%d-%m-%Y",
                utc=True,
            )

        except (
            ValueError,
            IndexError,
        ):
            continue

        try:

            tmax = float(fields[1])
            tmin = float(fields[2])
            tmean = float(fields[3])

        except (
            ValueError,
            IndexError,
        ):
            continue

        rows.append(
            {
                "time": timestamp,
                "mean": tmean,
                "min": tmin,
                "max": tmax,
                "count": np.nan,
                "source": "KMI",
            }
        )

    if not rows:

        raise ValueError(
            "No KMI observations could be parsed."
        )

    return (
        pd.DataFrame(rows)
        .set_index("time")
        .sort_index()
    )


def append_recent_kmi(
    daily: pd.DataFrame,
    location: Location,
) -> pd.DataFrame:
    """
    Append KMI observations after the latest ERA5 day.

    ERA5 remains authoritative wherever both sources overlap.
    """

    if not location.add_kmi_latest:
        return daily

    try:

        kmi = download_kmi_uccle()

    except requests.RequestException as exc:

        logger.warning(
            "Could not download KMI observations: %s",
            exc,
        )

        return daily

    except ValueError as exc:

        logger.warning(
            "Could not parse KMI observations: %s",
            exc,
        )

        return daily

    last_era5 = daily.index.max()

    kmi = kmi.loc[
        kmi.index > last_era5
    ]

    if kmi.empty:

        logger.info(
            "No KMI observations newer than ERA5."
        )

        return daily

    logger.info(
        "Adding %d recent KMI observation days.",
        len(kmi),
    )

    return (
        pd.concat([daily, kmi])
        .sort_index()
    )


# ============================================================================
# Climatology
# ============================================================================


def calendar_day(
    index: pd.DatetimeIndex,
) -> pd.Index:
    """
    Return calendar-day labels such as ``01-31`` or ``12-25``.
    """

    return index.strftime("%m-%d")


def compute_daily_climatology(
    daily: pd.DataFrame,
    year_start: int = CLIMATOLOGY_START,
    year_end: int = CLIMATOLOGY_END,
    smoothing_days: int = CLIMATOLOGY_SMOOTHING_DAYS,
) -> pd.DataFrame:
    """
    Compute raw and circularly smoothed 365-day temperature climatology.

    February 29 is excluded from the climatology but retained in the
    original daily time series.
    """

    logger.info(
        "Computing daily climatology %d-%d.",
        year_start,
        year_end,
    )

    reference = daily.loc[
        (daily.index.year >= year_start)
        & (daily.index.year <= year_end)
        & (daily["source"] == "ERA5")
    ].copy()

    reference = reference.loc[
        ~(
            (reference.index.month == 2)
            & (reference.index.day == 29)
        )
    ]

    reference["calendar_day"] = (
        calendar_day(reference.index)
    )

    climatology = (
        reference
        .groupby("calendar_day")["mean"]
        .mean()
        .rename("climatology")
        .to_frame()
    )

    calendar = pd.date_range(
        "2001-01-01",
        "2001-12-31",
        freq="1D",
        tz="UTC",
    )

    labels = calendar.strftime("%m-%d")

    climatology = (
        climatology.reindex(labels)
    )

    if climatology["climatology"].isna().any():

        missing = climatology.index[
            climatology["climatology"].isna()
        ].tolist()

        raise ValueError(
            "Climatology contains missing calendar days: "
            f"{missing}"
        )

    values = (
        climatology["climatology"]
        .to_numpy()
    )

    climatology["smoothed"] = (
        circular_rolling_mean(
            values,
            smoothing_days,
        )
    )

    climatology["reference_date"] = calendar

    return climatology


def circular_rolling_mean(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Compute centered circular moving average.
    """

    if window % 2 == 0:
        raise ValueError(
            "Smoothing window must be odd."
        )

    if window > len(values):
        raise ValueError(
            "Smoothing window cannot exceed the length of the series."
        )

    half = window // 2

    padded = np.concatenate(
        [
            values[-half:],
            values,
            values[:half],
        ]
    )

    smoothed = (
        pd.Series(padded)
        .rolling(
            window=window,
            center=True,
        )
        .mean()
        .to_numpy()
    )

    return smoothed[
        half:-half
    ]


def add_climatology_to_daily(
    daily: pd.DataFrame,
    climatology: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach climatological mean and anomaly to every daily value.
    """

    result = daily.copy()

    lookup = (
        climatology["smoothed"]
        .to_dict()
    )

    result["calendar_day"] = (
        calendar_day(result.index)
    )

    result["climatology"] = (
        result["calendar_day"]
        .map(lookup)
    )

    feb29 = (
        (result.index.month == 2)
        & (result.index.day == 29)
    )

    if feb29.any():

        feb29_clim = np.mean(
            [
                lookup["02-28"],
                lookup["03-01"],
            ]
        )

        result.loc[
            feb29,
            "climatology",
        ] = feb29_clim

    result["anomaly"] = (
        result["mean"]
        - result["climatology"]
    )

    return result


# ============================================================================
# Historical daily records
# ============================================================================


def add_previous_records(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute previous max/min record for each calendar day.

    Example
    -------
    ``previous_record_max`` for 15 August 2026 is the largest daily
    maximum previously observed on any earlier 15 August.
    """

    logger.info(
        "Computing historical daily records."
    )

    result = daily.copy()

    result["calendar_day"] = (
        calendar_day(result.index)
    )

    result["previous_record_max"] = (
        result
        .groupby("calendar_day")["max"]
        .transform(
            lambda x:
                x.expanding()
                .max()
                .shift(1)
        )
    )

    result["previous_record_min"] = (
        result
        .groupby("calendar_day")["min"]
        .transform(
            lambda x:
                x.expanding()
                .min()
                .shift(1)
        )
    )

    result["record_high"] = (
        result["max"]
        > result["previous_record_max"]
    )

    result["record_low"] = (
        result["min"]
        < result["previous_record_min"]
    )

    return result


# ============================================================================
# Output files
# ============================================================================


def write_csv_files(
    location: Location,
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    """
    Write hourly and daily temperature statistics to CSV.
    """

    safe_name = (
        location.name.replace(" ", "_")
    )

    hourly_out = (
        OUTPUT_DIR
        / f"hourly_T2M_{safe_name}.csv.gz"
    )

    hourly.round(2).to_csv(
        hourly_out,
        compression="gzip",
    )

    daily_out = (
        OUTPUT_DIR
        / f"daily_statistics_T2M_{safe_name}.csv"
    )

    columns = [
        "mean",
        "min",
        "max",
        "source",
        "climatology",
        "anomaly",
    ]

    daily[columns].round(2).to_csv(
        daily_out
    )

    logger.info(
        "Written %s",
        hourly_out,
    )

    logger.info(
        "Written %s",
        daily_out,
    )


# ============================================================================
# Figure utilities
# ============================================================================


def format_date_axis(
    ax: plt.Axes,
) -> None:
    """
    Apply common formatting to a date axis.
    """

    ax.xaxis.set_major_locator(
        mdates.MonthLocator()
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%d %b %y"
        )
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.grid(
        alpha=0.3,
        linewidth=0.7,
    )

    ax.set_axisbelow(True)


# ============================================================================
# Climatology figure
# ============================================================================


def plot_climatology(
    location: Location,
    climatology: pd.DataFrame,
) -> None:
    """
    Plot raw and smoothed annual temperature cycle.
    """

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        climatology["reference_date"],
        climatology["climatology"],
        lw=1,
        label="Daily climatology",
    )

    ax.plot(
        climatology["reference_date"],
        climatology["smoothed"],
        lw=2,
        label=(
            f"{CLIMATOLOGY_SMOOTHING_DAYS}-day smooth"
        ),
    )

    ax.xaxis.set_major_locator(
        mdates.MonthLocator()
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b")
    )

    ax.grid(
        alpha=0.3
    )

    ax.set_ylabel(
        "Temperature (°C)"
    )

    ax.set_title(
        f"ERA5 daily 2-m temperature climatology – {location.name}\n"
        f"{CLIMATOLOGY_START}–{CLIMATOLOGY_END}"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        FIGURE_DIR
        / f"climatology_{location.name}.png"
    )

    fig.savefig(
        path,
        dpi=200,
    )

    plt.close(fig)

    logger.info(
        "Written %s",
        path,
    )


# ============================================================================
# Generic temperature-period figure
# ============================================================================


def plot_temperature_period(
    location: Location,
    daily: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    title: str,
    output_path: Path,
    xlim_end: pd.Timestamp | None = None,
) -> None:
    """
    Plot daily mean temperature over a specified period.

    The climatological seasonal cycle is drawn as a black dashed line.

    Daily anomalies are represented as bars:
        blue = colder than climatology
        red  = warmer than climatology

    ERA5 bars are filled.
    KMI bars are outlined.
    """

    subset = daily.loc[
        (daily.index >= start)
        & (daily.index <= end)
    ].copy()

    if subset.empty:

        logger.warning(
            "No daily data available between %s and %s.",
            start,
            end,
        )

        return

    if xlim_end is None:
        xlim_end = end

    fig, ax = plt.subplots(
        figsize=(10, 4.5)
    )

    # ------------------------------------------------------------------
    # Climatological seasonal cycle
    # ------------------------------------------------------------------

    ax.plot(
        subset.index,
        subset["climatology"],
        linestyle="--",
        linewidth=1.2,
        color="black",
        label=(
            f"Climatology "
            f"({CLIMATOLOGY_START}–{CLIMATOLOGY_END})"
        ),
        zorder=3,
    )

    # ------------------------------------------------------------------
    # Daily anomalies
    # ------------------------------------------------------------------

    norm = Normalize(
        vmin=-10,
        vmax=10,
        clip=True,
    )

    cmap = plt.get_cmap(
        "RdBu_r"
    )

    for timestamp, row in subset.iterrows():

        anomaly = row["anomaly"]
        climatology = row["climatology"]

        if (
            pd.isna(anomaly)
            or pd.isna(climatology)
        ):
            continue

        color = cmap(
            norm(anomaly)
        )

        if row["source"] == "ERA5":

            ax.bar(
                timestamp,
                anomaly,
                bottom=climatology,
                width=1.0,
                color=color,
                linewidth=0,
                zorder=2,
            )

        else:

            ax.bar(
                timestamp,
                anomaly,
                bottom=climatology,
                width=1.0,
                facecolor="none",
                edgecolor=color,
                linewidth=0.8,
                linestyle="--",
                zorder=2,
            )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    format_date_axis(ax)

    ax.axhline(
        0,
        color="black",
        linewidth=0.6,
    )

    ax.set_xlim(
        start,
        xlim_end,
    )

    ax.set_ylabel(
        "Temperature (°C)"
    )

    ax.set_title(
        title
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
    )

    plt.close(fig)

    logger.info(
        "Written %s",
        output_path,
    )


# ============================================================================
# Recent temperature figure
# ============================================================================


def plot_recent_temperature(
    location: Location,
    daily: pd.DataFrame,
    days: int = 365,
) -> None:
    """
    Plot the most recent ``days`` days using the standard temperature style.
    """

    end = daily.index.max()

    start = (
        end
        - pd.Timedelta(days=days)
    )

    output_path = (
        FIGURE_DIR
        / f"T2M_{location.name}_last365d.png"
    )

    title = (
        f"Daily mean 2-m air temperature\n"
        f"{location.name}"
    )

    plot_temperature_period(
        location=location,
        daily=daily,
        start=start,
        end=end,
        xlim_end=(
            end
            + pd.Timedelta(days=5)
        ),
        title=title,
        output_path=output_path,
    )


# ============================================================================
# Historical annual temperature figure
# ============================================================================


def plot_annual_year(
    location: Location,
    daily: pd.DataFrame,
    year: int,
) -> None:
    """
    Plot one calendar year using exactly the same graphical style
    as the recent 365-day figure.
    """

    start = pd.Timestamp(
        year=year,
        month=1,
        day=1,
        tz="UTC",
    )

    end = pd.Timestamp(
        year=year,
        month=12,
        day=31,
        tz="UTC",
    )

    output_path = (
        FIGURE_DIR
        / f"T2M_{location.name}_{year}.png"
    )

    title = (
        f"Daily mean 2-m air temperature\n"
        f"{location.name}, {year}"
    )

    plot_temperature_period(
        location=location,
        daily=daily,
        start=start,
        end=end,
        title=title,
        output_path=output_path,
    )


# ============================================================================
# Recent minimum / maximum figure
# ============================================================================


def plot_recent_minmax(
    location: Location,
    daily: pd.DataFrame,
    days: int = 365,
) -> None:
    """
    Plot recent daily min/max temperatures and historical records.
    """

    end = daily.index.max()

    start = (
        end
        - pd.Timedelta(days=days)
    )

    subset = daily.loc[
        daily.index >= start
    ].copy()

    fig, ax = plt.subplots(
        figsize=(10, 4.5)
    )

    # Historical record envelope.
    ax.fill_between(
        subset.index,
        subset["previous_record_min"],
        subset["previous_record_max"],
        alpha=0.18,
        label="Previous historical min–max",
    )

    # Current daily range.
    ax.vlines(
        subset.index,
        subset["min"],
        subset["max"],
        linewidth=1,
        alpha=0.7,
        label="Daily min–max",
    )

    ax.plot(
        subset.index,
        subset["mean"],
        linewidth=0.8,
        label="Daily mean",
    )

    highs = subset.loc[
        subset["record_high"]
    ]

    lows = subset.loc[
        subset["record_low"]
    ]

    ax.scatter(
        highs.index,
        highs["max"],
        marker="*",
        s=30,
        zorder=5,
        label="New record high",
    )

    ax.scatter(
        lows.index,
        lows["min"],
        marker="*",
        s=30,
        zorder=5,
        label="New record low",
    )

    format_date_axis(ax)

    ax.axhline(
        0,
        color="black",
        linewidth=0.6,
    )

    ax.set_xlim(
        start,
        end + pd.Timedelta(days=5),
    )

    ax.set_ylabel(
        "Temperature (°C)"
    )

    ax.set_title(
        f"Daily minimum and maximum 2-m temperature\n"
        f"{location.name}"
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    path = (
        FIGURE_DIR
        / f"T2M_MinMax_{location.name}_last365d.png"
    )

    fig.savefig(
        path,
        dpi=300,
    )

    plt.close(fig)

    logger.info(
        "Written %s",
        path,
    )


# ============================================================================
# Main processing chain
# ============================================================================


def process_location(
    location: Location,
    make_historical_figures: bool = False,
) -> None:
    """
    Run complete processing chain for one location.
    """

    logger.info(
        "=" * 70
    )

    logger.info(
        "Processing %s",
        location.name,
    )

    logger.info(
        "=" * 70
    )

    # ERA5 hourly data.
    hourly = update_era5_cache(
        location
    )

    # Daily statistics.
    daily = compute_daily_statistics(
        hourly
    )

    # Recent KMI/IRM observations.
    daily = append_recent_kmi(
        daily=daily,
        location=location,
    )

    # Climatology.
    climatology = compute_daily_climatology(
        daily
    )

    daily = add_climatology_to_daily(
        daily,
        climatology,
    )

    # Historical records.
    daily = add_previous_records(
        daily
    )

    # CSV outputs.
    write_csv_files(
        location=location,
        hourly=hourly,
        daily=daily,
    )

    # Figures.
    plot_climatology(
        location,
        climatology,
    )

    plot_recent_temperature(
        location,
        daily,
    )

    plot_recent_minmax(
        location,
        daily,
    )

    # Historical annual figures.
    if make_historical_figures:

        first_year = (
            daily.index.year.min()
        )

        last_year = (
            daily.index.year.max()
        )

        for year in range(
            first_year,
            last_year + 1,
        ):

            logger.info(
                "Producing annual figure %d",
                year,
            )

            plot_annual_year(
                location,
                daily,
                year,
            )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """
    Run the analysis for all configured locations.
    """

    for location in LOCATIONS:

        try:

            process_location(
                location,
                make_historical_figures=True,
            )

        except Exception:

            logger.exception(
                "Processing failed for %s",
                location.name,
            )


if __name__ == "__main__":
    main()