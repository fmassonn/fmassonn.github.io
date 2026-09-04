"""
Download, process and plot ERA5 500-hPa geopotential height at point locations.

Author
------
François Massonnet

Adapted from the ERA5 2-m temperature script in September 2026,
with assistance from ChatGPT.

Description
-----------
The script retrieves ERA5 geopotential time series on pressure levels for
one or several point locations, keeps only the 500-hPa level, converts
geopotential (m2 s-2) to geopotential height (m), stores the data in a local
cache, computes daily and climatological statistics, and produces CSV files
and diagnostic figures.

ERA5 data source
----------------
Copernicus Climate Data Store:
    reanalysis-era5-pressure-levels-timeseries

The time-series product is optimized for long records at a single point.
It is 6-hourly (00, 06, 12, 18 UTC) and returns 13 pressure levels.  The CSV
reader below explicitly filters pressureLevel == 500 hPa.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta, datetime
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
PRESSURE_LEVEL = 500.0  # hPa

# Standard gravity used to convert geopotential Phi [m2 s-2]
# to geopotential height Z = Phi / g0 [m].
STANDARD_GRAVITY = 9.80665

START_DATE = date(1940, 1, 1)

# ERA5 is normally available with a latency of about five days.
ERA5_LAG_DAYS = 5

CLIMATOLOGY_START = 1991
CLIMATOLOGY_END = 2020

# Re-download this recent period on every run, so preliminary ERA5T values
# can be replaced by consolidated ERA5 values when necessary.
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
    """Download ERA5 pressure-level geopotential time series for one point."""

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
        "Downloading ERA5 pressure-level geopotential for %s: %s -> %s",
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


def read_era5_csv(path: Path) -> pd.DataFrame:
    """
    Read ERA5 geopotential from a CDS CSV response and retain 500 hPa only.

    Depending on the CDS backend, the response may be either a plain CSV
    file or a ZIP archive containing a CSV file.

    Expected CSV columns include:
        valid_time
        pressureLevel
        z
        latitude
        longitude

    ``z`` is geopotential in m2 s-2. It is converted to geopotential height
    in metres using Z = z / 9.80665.

    Returns
    -------
    pandas.DataFrame
        UTC DatetimeIndex and one column named ``z500`` in metres.
    """

    if not path.exists():
        raise FileNotFoundError(f"ERA5 download does not exist: {path}")

    if zipfile.is_zipfile(path):
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
            logger.info("Reading ERA5 CSV from ZIP archive: %s", csv_name)

            with archive.open(csv_name) as csv_file:
                df = pd.read_csv(csv_file)
    else:
        logger.info("Reading ERA5 response as a plain CSV file.")
        df = pd.read_csv(path)

    required_columns = {"valid_time", "pressureLevel", "z"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Unexpected ERA5 CSV format.\n"
            f"Missing columns: {sorted(missing_columns)}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Pressure level may be read as either integer or float.
    pressure = pd.to_numeric(df["pressureLevel"], errors="coerce")
    available = sorted(pressure.dropna().unique())
    df = df.loc[np.isclose(pressure, PRESSURE_LEVEL)].copy()

    if df.empty:
        raise ValueError(
            f"No {PRESSURE_LEVEL:g}-hPa values found. "
            f"Available pressure levels: {available}"
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

    df["z500"] = geopotential / STANDARD_GRAVITY

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
        check_six_hourly_data(cached)
        return cached

    with NamedTemporaryFile(
        suffix=".download",
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
        new_data = read_era5_csv(temporary_path)
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

    check_six_hourly_data(combined)
    combined.to_parquet(path)

    logger.info(
        "ERA5 cache contains %d six-hourly values (%s -> %s)",
        len(combined),
        combined.index.min(),
        combined.index.max(),
    )

    return combined


# ============================================================================
# Quality control
# ============================================================================


def check_six_hourly_data(df: pd.DataFrame) -> None:
    """Perform basic QC checks on the 6-hourly ERA5 Z500 series."""

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
            "%d six-hourly timestamps are missing from the ERA5 series.",
            len(missing_times),
        )

    # Broad sanity limits for 500-hPa geopotential height.
    suspicious = df["z500"].notna() & ~df["z500"].between(4000, 6500)

    if suspicious.any():
        logger.warning(
            "%d physically suspicious Z500 values detected.",
            suspicious.sum(),
        )


# ============================================================================
# Daily statistics
# ============================================================================


def compute_daily_statistics(six_hourly: pd.DataFrame) -> pd.DataFrame:
    """Compute daily mean, minimum and maximum 500-hPa height."""

    logger.info("Computing daily statistics.")

    daily = (
        six_hourly["z500"]
        .resample("1D")
        .agg(
            mean="mean",
            min="min",
            max="max",
            count="count",
        )
    )

    incomplete = daily["count"] != 4
    if incomplete.any():
        logger.warning(
            "%d incomplete ERA5 days detected (expected 4 values/day).",
            incomplete.sum(),
        )

    return daily


# ============================================================================
# Climatology
# ============================================================================


def calendar_day(index: pd.DatetimeIndex) -> pd.Index:
    return index.strftime("%m-%d")



def circular_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute centered circular moving average."""

    if window % 2 == 0:
        raise ValueError("Smoothing window must be odd.")
    if window > len(values):
        raise ValueError("Smoothing window cannot exceed series length.")

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

    logger.info("Computing daily climatology %d-%d.", year_start, year_end)

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
    """Attach climatological Z500 and anomaly to every daily value."""

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
# Historical daily records
# ============================================================================


def add_previous_records(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute previous daily max/min records for each calendar day."""

    logger.info("Computing historical daily records.")

    result = daily.copy()
    result["calendar_day"] = calendar_day(result.index)

    result["previous_record_max"] = (
        result.groupby("calendar_day")["max"]
        .transform(lambda x: x.expanding().max().shift(1))
    )
    result["previous_record_min"] = (
        result.groupby("calendar_day")["min"]
        .transform(lambda x: x.expanding().min().shift(1))
    )

    result["record_high"] = result["max"] > result["previous_record_max"]
    result["record_low"] = result["min"] < result["previous_record_min"]

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

    six_hourly_out = OUTPUT_DIR / f"six_hourly_Z500_{safe_name}.csv.gz"
    daily_out = OUTPUT_DIR / f"daily_statistics_Z500_{safe_name}.csv"

    six_hourly.round(2).to_csv(
        six_hourly_out,
        compression="gzip",
    )

    columns = [
        "mean",
        "min",
        "max",
        "climatology",
        "anomaly",
    ]
    daily[columns].round(2).to_csv(daily_out)

    logger.info("Written %s", six_hourly_out)
    logger.info("Written %s", daily_out)


# ============================================================================
# Figure utilities
# ============================================================================


def format_date_axis(ax: plt.Axes) -> None:
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
    ax.set_ylabel("500-hPa geopotential height (m)")
    ax.set_title(
        f"ERA5 daily 500-hPa geopotential height climatology – {location.name}\n"
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
    """Plot daily mean Z500 and its anomaly relative to climatology."""

    subset = daily.loc[
        (daily.index >= start)
        & (daily.index <= end)
    ].copy()

    if subset.empty:
        logger.warning("No daily data available between %s and %s.", start, end)
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
        label=f"Climatology ({CLIMATOLOGY_START}–{CLIMATOLOGY_END})",
        zorder=3,
    )

    # A +/- 250 m range gives useful colour contrast for synoptic Z500 anomalies.
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
    ax.set_ylabel("500-hPa geopotential height (m)")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    logger.info("Written %s", output_path)


# ============================================================================
# Recent Z500 figure
# ============================================================================


def plot_recent_z500(
    location: Location,
    daily: pd.DataFrame,
    days: int = 365,
) -> None:
    end = daily.index.max()
    start = end - pd.Timedelta(days=days)

    output_path = FIGURE_DIR / f"Z500_{location.name}_last365d.png"
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
# Historical annual Z500 figure
# ============================================================================


def plot_annual_year(
    location: Location,
    daily: pd.DataFrame,
    year: int,
) -> None:
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
# Recent minimum / maximum figure
# ============================================================================


def plot_recent_minmax(
    location: Location,
    daily: pd.DataFrame,
    days: int = 365,
) -> None:
    end = daily.index.max()
    start = end - pd.Timedelta(days=days)
    subset = daily.loc[daily.index >= start].copy()

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.fill_between(
        subset.index,
        subset["previous_record_min"],
        subset["previous_record_max"],
        alpha=0.18,
        label="Previous historical min–max",
    )
    ax.vlines(
        subset.index,
        subset["min"],
        subset["max"],
        linewidth=1,
        alpha=0.7,
        label="Daily 6-hourly min–max",
    )
    ax.plot(
        subset.index,
        subset["mean"],
        linewidth=0.8,
        label="Daily mean",
    )

    highs = subset.loc[subset["record_high"]]
    lows = subset.loc[subset["record_low"]]

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
    ax.set_xlim(start, end + pd.Timedelta(days=5))
    ax.set_ylabel("500-hPa geopotential height (m)")
    ax.set_title(
        "Daily minimum and maximum 500-hPa geopotential height\n"
        f"{location.name}"
    )
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = FIGURE_DIR / f"Z500_MinMax_{location.name}_last365d.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    logger.info("Written %s", path)


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
    daily = add_previous_records(daily)

    write_csv_files(
        location=location,
        six_hourly=six_hourly,
        daily=daily,
    )

    plot_climatology(location, climatology)
    plot_recent_z500(location, daily)
    plot_recent_minmax(location, daily)

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
    for location in LOCATIONS:
        try:
            process_location(
                location,
                make_historical_figures=True,
            )
        except Exception:
            logger.exception("Processing failed for %s", location.name)


if __name__ == "__main__":
    main()
