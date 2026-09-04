"""
Download, process and plot ERA5 total precipitation at point locations.

Author
------
François Massonnet

Original precipitation version
------------------------------
August 2026, assisted by ChatGPT

Description
-----------
The script retrieves hourly ERA5 total precipitation for one or several
point locations, stores the data in a local cache, computes daily totals
and climatological statistics, and produces CSV files and diagnostic
figures.

ERA5 data source
----------------
Copernicus Climate Data Store:
    reanalysis-era5-single-levels-timeseries

The time-series product is optimized for retrieving long ERA5 series
at a single geographical point.

ERA5 total precipitation is provided in metres of water equivalent
accumulated over the hour ending at the validity time. It is converted
here to millimetres.

For daily totals, the hourly timestamps are shifted backward by one hour
before resampling. This ensures that, for a given day, the accumulation
contains:

    01 UTC ... 23 UTC of that day
    + 00 UTC of the following day

as prescribed for ERA5 accumulated precipitation.
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
from matplotlib.colors import Normalize


# ============================================================================
# Configuration
# ============================================================================

ERA5_DATASET = "reanalysis-era5-single-levels-timeseries"
ERA5_VARIABLE = "total_precipitation"

START_DATE = date(1940, 1, 1)

# ERA5 is normally available with a latency of about five days.
ERA5_LAG_DAYS = 5

CLIMATOLOGY_START = 1991
CLIMATOLOGY_END = 2020

# Re-download this recent period on every run.
REFRESH_DAYS = 120

CLIMATOLOGY_SMOOTHING_DAYS = 61

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


LOCATIONS = [
    Location(
        name="Bruxelles",
        latitude=50.85,
        longitude=4.35,
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
    Download hourly ERA5 total precipitation for one point.

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
        "Downloading ERA5 precipitation for %s: %s -> %s",
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
    Read hourly ERA5 total precipitation from a CDS ZIP archive.

    Expected CSV columns include:

        valid_time
        latitude
        longitude
        tp

    ``tp`` is stored in metres of water equivalent and converted here
    to millimetres.

    Returns
    -------
    pandas.DataFrame
        UTC DatetimeIndex and one column named ``precipitation`` in mm.
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
        "tp",
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

    precipitation_m = pd.to_numeric(
        df["tp"],
        errors="coerce",
    )

    if precipitation_m.isna().any():

        logger.warning(
            "%d ERA5 precipitation values could not be converted to numbers.",
            precipitation_m.isna().sum(),
        )

    # m -> mm
    df["precipitation"] = (
        precipitation_m * 1000.0
    )

    # Tiny negative values can occasionally result from numerical
    # precision / encoding. Values extremely close to zero are set to zero.
    tiny_negative = (
        (df["precipitation"] < 0)
        & (df["precipitation"] > -1e-6)
    )

    df.loc[
        tiny_negative,
        "precipitation",
    ] = 0.0

    result = (
        df[["valid_time", "precipitation"]]
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
        / f"ERA5_PRECIP_{safe_name}.parquet"
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
            columns=["precipitation"],
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
    Perform basic QC checks on the hourly ERA5 precipitation series.
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

    negative = (
        df["precipitation"] < 0
    )

    if negative.any():

        logger.warning(
            "%d negative hourly precipitation values detected.",
            negative.sum(),
        )

    # This is deliberately generous: values larger than 100 mm/h
    # deserve inspection but are not automatically discarded.
    suspicious_high = (
        df["precipitation"] > 100
    )

    if suspicious_high.any():

        logger.warning(
            "%d hourly precipitation values exceed 100 mm/h.",
            suspicious_high.sum(),
        )


# ============================================================================
# Daily statistics
# ============================================================================


def compute_daily_statistics(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily ERA5 precipitation totals.

    ERA5 total precipitation at timestamp T represents the accumulation
    over the hour ending at T.

    Therefore, precipitation for calendar day D is:

        D 01 UTC
        ...
        D 23 UTC
        D+1 00 UTC

    Before resampling, timestamps are shifted backward by one hour so
    that the 00 UTC accumulation is attributed to the preceding day.

    Days with fewer than 24 hourly values are considered incomplete and
    their total is set to NaN.
    """

    logger.info(
        "Computing daily precipitation totals."
    )

    shifted = hourly.copy()

    shifted.index = (
        shifted.index
        - pd.Timedelta(hours=1)
    )

    daily = (
        shifted["precipitation"]
        .resample("1D")
        .agg(
            total="sum",
            count="count",
        )
    )

    # Ignore the artificial previous day created by shifting
    # START_DATE 00 UTC backward by one hour.
    start_timestamp = pd.Timestamp(
        START_DATE,
        tz="UTC",
    )

    daily = daily.loc[
        daily.index >= start_timestamp
    ]

    incomplete = (
        daily["count"] != 24
    )

    if incomplete.any():

        logger.warning(
            "%d incomplete ERA5 precipitation days detected.",
            incomplete.sum(),
        )

        # A partial daily sum would be misleading.
        daily.loc[
            incomplete,
            "total",
        ] = np.nan

    daily["source"] = "ERA5"

    return daily


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
    Compute raw and circularly smoothed 365-day precipitation climatology.

    February 29 is excluded from the climatology but retained in the
    original daily time series.
    """

    logger.info(
        "Computing daily precipitation climatology %d-%d.",
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
        .groupby("calendar_day")["total"]
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
    Attach climatological precipitation and anomaly to every daily value.
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
        result["total"]
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
    Compute previous precipitation record for each calendar day.

    Example
    -------
    ``previous_record_max`` for 15 August 2026 is the largest daily
    precipitation total previously observed on any earlier 15 August.
    """

    logger.info(
        "Computing historical daily precipitation records."
    )

    result = daily.copy()

    result["calendar_day"] = (
        calendar_day(result.index)
    )

    result["previous_record_max"] = (
        result
        .groupby("calendar_day")["total"]
        .transform(
            lambda x:
                x.expanding()
                .max()
                .shift(1)
        )
    )

    result["record_high"] = (
        result["total"]
        > result["previous_record_max"]
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
    Write hourly and daily precipitation statistics to CSV.
    """

    safe_name = (
        location.name.replace(" ", "_")
    )

    hourly_out = (
        OUTPUT_DIR
        / f"hourly_PRECIP_{safe_name}.csv.gz"
    )

    hourly.round(3).to_csv(
        hourly_out,
        compression="gzip",
    )

    daily_out = (
        OUTPUT_DIR
        / f"daily_statistics_PRECIP_{safe_name}.csv"
    )

    columns = [
        "total",
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
    Plot raw and smoothed annual precipitation cycle.
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
        "Precipitation (mm/day)"
    )

    ax.set_ylim(
        bottom=0
    )

    ax.set_title(
        f"ERA5 daily precipitation climatology – {location.name}\n"
        f"{CLIMATOLOGY_START}–{CLIMATOLOGY_END}"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        FIGURE_DIR
        / f"climatology_PRECIP_{location.name}.png"
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
# Generic precipitation-period figure
# ============================================================================


def plot_precipitation_period(
    location: Location,
    daily: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    title: str,
    output_path: Path,
    xlim_end: pd.Timestamp | None = None,
) -> None:
    """
    Plot daily precipitation over a specified period.

    The climatological seasonal cycle is drawn as a black dashed line.

    Daily anomalies are represented as bars:

        blue = drier than climatology
        red  = wetter than climatology

    The upper/lower end of each bar is the actual daily precipitation.
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

        ax.bar(
            timestamp,
            anomaly,
            bottom=climatology,
            width=1.0,
            color=color,
            linewidth=0,
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

    ax.set_ylim(
        bottom=0
    )

    ax.set_ylabel(
        "Daily precipitation (mm)"
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
# Recent precipitation figure
# ============================================================================


def plot_recent_precipitation(
    location: Location,
    daily: pd.DataFrame,
    days: int = 365,
) -> None:
    """
    Plot the most recent ``days`` days using the standard precipitation style.
    """

    valid = daily["total"].dropna()

    if valid.empty:
        return

    end = valid.index.max()

    start = (
        end
        - pd.Timedelta(days=days)
    )

    output_path = (
        FIGURE_DIR
        / f"PRECIP_{location.name}_last365d.png"
    )

    title = (
        f"Daily total precipitation\n"
        f"{location.name}"
    )

    plot_precipitation_period(
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
# Historical annual precipitation figure
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
        / f"PRECIP_{location.name}_{year}.png"
    )

    title = (
        f"Daily total precipitation\n"
        f"{location.name}, {year}"
    )

    plot_precipitation_period(
        location=location,
        daily=daily,
        start=start,
        end=end,
        title=title,
        output_path=output_path,
    )


# ============================================================================
# Recent precipitation records figure
# ============================================================================


def plot_recent_records(
    location: Location,
    daily: pd.DataFrame,
    days: int = 365,
) -> None:
    """
    Plot recent daily precipitation and previous calendar-day records.
    """

    valid = daily["total"].dropna()

    if valid.empty:
        return

    end = valid.index.max()

    start = (
        end
        - pd.Timedelta(days=days)
    )

    subset = daily.loc[
        (daily.index >= start)
        & (daily.index <= end)
    ].copy()

    fig, ax = plt.subplots(
        figsize=(10, 4.5)
    )

    # Previous historical record.
    ax.plot(
        subset.index,
        subset["previous_record_max"],
        linewidth=1,
        alpha=0.8,
        label="Previous daily record",
    )

    # Current daily totals.
    ax.vlines(
        subset.index,
        0,
        subset["total"],
        linewidth=1,
        alpha=0.7,
        label="Daily precipitation",
    )

    records = subset.loc[
        subset["record_high"]
        & subset["total"].notna()
    ]

    ax.scatter(
        records.index,
        records["total"],
        marker="*",
        s=40,
        zorder=5,
        label="New daily record",
    )

    format_date_axis(ax)

    ax.set_xlim(
        start,
        end + pd.Timedelta(days=5),
    )

    ax.set_ylim(
        bottom=0
    )

    ax.set_ylabel(
        "Daily precipitation (mm)"
    )

    ax.set_title(
        f"Daily precipitation and historical records\n"
        f"{location.name}"
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    path = (
        FIGURE_DIR
        / f"PRECIP_Records_{location.name}_last365d.png"
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

    # Daily precipitation totals.
    daily = compute_daily_statistics(
        hourly
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

    plot_recent_precipitation(
        location,
        daily,
    )

    plot_recent_records(
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
