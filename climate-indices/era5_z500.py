"""
Download, process and plot ERA5 500-hPa geopotential height at point locations.

Author
------
François Massonnet

Adapted in September 2026 with assistance from ChatGPT from the ERA5 2-m
temperature script.

Description
-----------
The script retrieves ERA5 geopotential at 500 hPa for one or several point
locations, stores the data in a local cache, converts geopotential (m2 s-2) to
geopotential height (gpm), computes daily means and a 1991-2020 daily
climatology, and produces CSV files and diagnostic figures.

ERA5 data source
----------------
Copernicus Climate Data Store:
    reanalysis-era5-pressure-levels-timeseries

The time-series product is optimized for retrieving long ERA5 series at a
single geographical point. At pressure levels it currently provides values at
00, 06, 12 and 18 UTC.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
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

ERA5_DATASET = "reanalysis-era5-pressure-levels-timeseries"
ERA5_VARIABLE = "geopotential"
PRESSURE_LEVEL = "500"

# Standard gravity used to convert geopotential (m2 s-2) to geopotential
# height (m, conventionally reported as geopotential metres or gpm).
G0 = 9.80665

START_DATE = date(1940, 1, 1)

# ERA5 is normally available with a latency of about five days.
ERA5_LAG_DAYS = 5

CLIMATOLOGY_START = 1991
CLIMATOLOGY_END = 2020

# Re-download this recent period on every run so that ERA5T values can be
# replaced by consolidated ERA5 values when needed.
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
    """Download ERA5 500-hPa geopotential for one point."""

    request = {
        "variable": [ERA5_VARIABLE],
        "pressure_level": [PRESSURE_LEVEL],
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
        "Downloading ERA5 Z500 for %s: %s -> %s",
        location.name,
        start_date,
        end_date,
    )

    client = cdsapi.Client()
    client.retrieve(ERA5_DATASET, request, str(target))


# ============================================================================
# ERA5 CSV reader
# ============================================================================


def read_era5_csv_zip(path: Path) -> pd.DataFrame:
    """
    Read ERA5 500-hPa geopotential from a CDS ZIP archive.

    ERA5 geopotential ``z`` is stored in m2 s-2. It is converted here to
    geopotential height by division by standard gravity, G0 = 9.80665 m s-2.
    The resulting unit is metres, conventionally called geopotential metres
    (gpm).
    """

    if not path.exists():
        raise FileNotFoundError(f"ERA5 archive does not exist: {path}")

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
            raise ValueError(f"No CSV file found in ERA5 archive {path}.")

        if len(csv_files) > 1:
            logger.warning(
                "Several CSV files found in ERA5 archive: %s. Using %s.",
                csv_files,
                csv_files[0],
            )

        csv_name = csv_files[0]
        logger.info("Reading ERA5 CSV: %s", csv_name)

        with archive.open(csv_name) as csv_file:
            df = pd.read_csv(csv_file)

    required_columns = {"valid_time", "z"}
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
        raise ValueError(f"{n_invalid} ERA5 timestamps could not be parsed.")

    geopotential = pd.to_numeric(df["z"], errors="coerce")

    if geopotential.isna().any():
        logger.warning(
            "%d ERA5 geopotential values could not be converted to numbers.",
            geopotential.isna().sum(),
        )

    df["z500"] = geopotential / G0

    result = (
        df[["valid_time", "z500"]]
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
    return DATA_DIR / f"ERA5_Z500_{safe_name}.parquet"



def update_era5_cache(location: Location) -> pd.DataFrame:
    """Update local ERA5 cache and return the complete 6-hourly series."""

    path = cache_path(location)
    latest_era5_date = date.today() - timedelta(days=ERA5_LAG_DAYS)

    if path.exists():
        logger.info("Reading ERA5 cache %s", path)
        cached = pd.read_parquet(path)

        if cached.index.tz is None:
            cached.index = cached.index.tz_localize("UTC")

        last_cached_date = cached.index.max().date()
        download_start = max(
            START_DATE,
            last_cached_date - timedelta(days=REFRESH_DAYS),
        )

    else:
        logger.info("No ERA5 cache found for %s", location.name)
        cached = pd.DataFrame(
            columns=["z500"],
            index=pd.DatetimeIndex([], tz="UTC", name="time"),
        )
        download_start = START_DATE

    if download_start > latest_era5_date:
        logger.info("ERA5 cache already contains all currently expected data.")
        check_6hourly_data(cached)
        return cached

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
        new_data = read_era5_csv_zip(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    if cached.empty:
        combined = new_data.copy()
    else:
        combined = pd.concat([cached, new_data])

    combined = (
        combined.loc[~combined.index.duplicated(keep="last")]
        .sort_index()
    )

    check_6hourly_data(combined)
    combined.to_parquet(path)

    logger.info(
        "ERA5 cache contains %d 6-hourly values (%s -> %s)",
        len(combined),
        combined.index.min(),
        combined.index.max(),
    )

    return combined


# ============================================================================
# Quality control
# ============================================================================


def check_6hourly_data(df: pd.DataFrame) -> None:
    """Perform basic QC checks on the 6-hourly ERA5 series."""

    if df.empty:
        raise ValueError("ERA5 dataset is empty.")

    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps found in ERA5 data.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("ERA5 timestamps are not sorted.")

    expected = pd.date_range(
        df.index.min(),
        df.index.max(),
        freq="6h",
        tz="UTC",
    )
    missing_times = expected.difference(df.index)

    if len(missing_times):
        logger.warning(
            "%d 6-hourly timestamps are missing from the ERA5 series.",
            len(missing_times),
        )

    # Broad physical sanity range for 500-hPa geopotential height.
    suspicious = df["z500"].notna() & ~df["z500"].between(4500, 6500)

    if suspicious.any():
        logger.warning(
            "%d physically suspicious Z500 values detected.",
            suspicious.sum(),
        )


# ============================================================================
# Daily statistics
# ============================================================================


def compute_daily_statistics(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily mean 500-hPa geopotential height.

    The time-series product supplies four analyses per day: 00, 06, 12 and
    18 UTC. Days are defined in UTC.
    """

    logger.info("Computing daily statistics.")

    daily = hourly["z500"].resample("1D").agg(
        mean="mean",
        count="count",
    )

    incomplete = daily["count"] != 4
    if incomplete.any():
        logger.warning(
            "%d incomplete ERA5 days detected.",
            incomplete.sum(),
        )

    return daily


# ============================================================================
# Climatology
# ============================================================================


def calendar_day(index: pd.DatetimeIndex) -> pd.Index:
    """Return calendar-day labels such as ``01-31`` or ``12-25``."""
    return index.strftime("%m-%d")



def circular_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute centered circular moving average."""

    if window % 2 == 0:
        raise ValueError("Smoothing window must be odd.")

    if window > len(values):
        raise ValueError(
            "Smoothing window cannot exceed the length of the series."
        )

    half = window // 2
    padded = np.concatenate([values[-half:], values, values[:half]])

    smoothed = (
        pd.Series(padded)
        .rolling(window=window, center=True)
        .mean()
        .to_numpy()
    )

    return smoothed[half:-half]



def compute_daily_climatology(
    daily: pd.DataFrame,
    year_start: int = CLIMATOLOGY_START,
    year_end: int = CLIMATOLOGY_END,
    smoothing_days: int = CLIMATOLOGY_SMOOTHING_DAYS,
) -> pd.DataFrame:
    """Compute raw and circularly smoothed 365-day Z500 climatology."""

    logger.info(
        "Computing daily climatology %d-%d.",
        year_start,
        year_end,
    )

    reference = daily.loc[
        (daily.index.year >= year_start)
        & (daily.index.year <= year_end)
    ].copy()

    reference = reference.loc[
        ~(
            (reference.index.month == 2)
            & (reference.index.day == 29)
        )
    ]

    reference["calendar_day"] = calendar_day(reference.index)

    climatology = (
        reference.groupby("calendar_day")["mean"]
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
    climatology = climatology.reindex(labels)

    if climatology["climatology"].isna().any():
        missing = climatology.index[
            climatology["climatology"].isna()
        ].tolist()
        raise ValueError(
            "Climatology contains missing calendar days: "
            f"{missing}"
        )

    climatology["smoothed"] = circular_rolling_mean(
        climatology["climatology"].to_numpy(),
        smoothing_days,
    )
    climatology["reference_date"] = calendar

    return climatology



def add_climatology_to_daily(
    daily: pd.DataFrame,
    climatology: pd.DataFrame,
) -> pd.DataFrame:
    """Attach climatological mean and anomaly to every daily value."""

    result = daily.copy()
    lookup = climatology["smoothed"].to_dict()

    result["calendar_day"] = calendar_day(result.index)
    result["climatology"] = result["calendar_day"].map(lookup)

    feb29 = (
        (result.index.month == 2)
        & (result.index.day == 29)
    )

    if feb29.any():
        result.loc[feb29, "climatology"] = np.mean(
            [lookup["02-28"], lookup["03-01"]]
        )

    result["anomaly"] = result["mean"] - result["climatology"]

    return result


# ============================================================================
# Output files
# ============================================================================


def write_csv_files(
    location: Location,
    six_hourly: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    """Write 6-hourly and daily Z500 data to CSV."""

    safe_name = location.name.replace(" ", "_")

    six_hourly_out = (
        OUTPUT_DIR / f"6hourly_Z500_{safe_name}.csv.gz"
    )
    six_hourly.round(2).to_csv(
        six_hourly_out,
        compression="gzip",
    )

    daily_out = OUTPUT_DIR / f"daily_Z500_{safe_name}.csv"
    columns = ["mean", "climatology", "anomaly"]
    daily[columns].round(2).to_csv(daily_out)

    logger.info("Written %s", six_hourly_out)
    logger.info("Written %s", daily_out)


# ============================================================================
# Figure utilities
# ============================================================================


def format_date_axis(ax: plt.Axes) -> None:
    """Apply common formatting to a date axis."""

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, linewidth=0.7)
    ax.set_axisbelow(True)


# ============================================================================
# Climatology figure
# ============================================================================


def plot_climatology(
    location: Location,
    climatology: pd.DataFrame,
) -> None:
    """Plot raw and smoothed annual Z500 cycle."""

    fig, ax = plt.subplots(figsize=(8, 4))

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
        label=f"{CLIMATOLOGY_SMOOTHING_DAYS}-day smooth",
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(alpha=0.3)
    ax.set_ylabel("500-hPa geopotential height (gpm)")
    ax.set_title(
        f"ERA5 daily Z500 climatology – {location.name}\n"
        f"{CLIMATOLOGY_START}–{CLIMATOLOGY_END}"
    )
    ax.legend()

    fig.tight_layout()

    path = FIGURE_DIR / f"climatology_Z500_{location.name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    logger.info("Written %s", path)


# ============================================================================
# Generic Z500-period figure
# ============================================================================


def plot_z500_period(
    location: Location,
    daily: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    title: str,
    output_path: Path,
    xlim_end: pd.Timestamp | None = None,
) -> None:
    """Plot daily mean Z500 and anomalies over a specified period."""

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

    fig, ax = plt.subplots(figsize=(10, 4.5))

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

    # A +/- 250 gpm range captures most daily Z500 anomalies over Brussels
    # while retaining contrast for synoptic variability.
    norm = Normalize(vmin=-250, vmax=250, clip=True)
    cmap = plt.get_cmap("RdBu_r")

    for timestamp, row in subset.iterrows():
        anomaly = row["anomaly"]
        climatology = row["climatology"]

        if pd.isna(anomaly) or pd.isna(climatology):
            continue

        ax.bar(
            timestamp,
            anomaly,
            bottom=climatology,
            width=1.0,
            color=cmap(norm(anomaly)),
            linewidth=0,
            zorder=2,
        )

    format_date_axis(ax)
    ax.set_xlim(start, xlim_end)
    ax.set_ylabel("500-hPa geopotential height (gpm)")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    logger.info("Written %s", output_path)


# ============================================================================
# Recent figure
# ============================================================================


def plot_recent_z500(
    location: Location,
    daily: pd.DataFrame,
    days: int = 365,
) -> None:
    """Plot the most recent ``days`` days."""

    end = daily.index.max()
    start = end - pd.Timedelta(days=days)

    output_path = (
        FIGURE_DIR / f"Z500_{location.name}_last365d.png"
    )

    creation_date = datetime.now().strftime("%d/%m/%Y")
    title = (
        "Daily mean 500-hPa geopotential height\n"
        f"{location.name} — Figure created on {creation_date}"
    )

    plot_z500_period(
        location=location,
        daily=daily,
        start=start,
        end=end,
        xlim_end=end + pd.Timedelta(days=5),
        title=title,
        output_path=output_path,
    )


# ============================================================================
# Historical annual figures
# ============================================================================


def plot_annual_year(
    location: Location,
    daily: pd.DataFrame,
    year: int,
) -> None:
    """Plot one calendar year using the same style as the recent figure."""

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = pd.Timestamp(year=year, month=12, day=31, tz="UTC")

    output_path = FIGURE_DIR / f"Z500_{location.name}_{year}.png"
    title = (
        "Daily mean 500-hPa geopotential height\n"
        f"{location.name}, {year}"
    )

    plot_z500_period(
        location=location,
        daily=daily,
        start=start,
        end=end,
        title=title,
        output_path=output_path,
    )


# ============================================================================
# Main processing chain
# ============================================================================


def process_location(
    location: Location,
    make_historical_figures: bool = False,
) -> None:
    """Run complete processing chain for one location."""

    logger.info("=" * 70)
    logger.info("Processing %s", location.name)
    logger.info("=" * 70)

    six_hourly = update_era5_cache(location)
    daily = compute_daily_statistics(six_hourly)

    climatology = compute_daily_climatology(daily)
    daily = add_climatology_to_daily(daily, climatology)

    write_csv_files(
        location=location,
        six_hourly=six_hourly,
        daily=daily,
    )

    plot_climatology(location, climatology)
    plot_recent_z500(location, daily)

    if make_historical_figures:
        first_year = daily.index.year.min()
        last_year = daily.index.year.max()

        for year in range(first_year, last_year + 1):
            logger.info("Producing annual figure %d", year)
            plot_annual_year(location, daily, year)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run the analysis for all configured locations."""

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
