"""Data loaders for the IEMP Cuban Sound Systems corpus."""

import os

import pandas as pd
import numpy as np
from rhythmic_segments import RhythmicSegments


def get_data_directory(subdirectory=None):
    """Resolve the shared CSS data directory from ``SHARED_DATA_DIR``.

    Parameters
    ----------
    subdirectory:
        Optional subfolder to append to the base path.
    """
    from dotenv import load_dotenv

    load_dotenv()

    data_dir = os.getenv("SHARED_DATA_DIR")
    if subdirectory:
        data_dir = os.path.join(data_dir, subdirectory)

    if not os.path.exists(data_dir):
        raise Exception(f"Data directory does not exist: {data_dir}")

    return data_dir


DATA_DIR = get_data_directory()

ALL_INSTRUMENTS = [
    "Clave",
    "Bass ",
    "Guitar",
    "Tres",
    "Bongo",
    "Bongos",
    "Bell",
    "Cajon",
    "Conga",
    "Trumpet",
]


def load_metre_data(song_id, data_dir=DATA_DIR):
    """Load the raw metre annotations for a given song id.

    Parameters
    ----------
    song_id:
        Identifier used in the CSS file naming scheme.
    data_dir:
        Base directory where the ``raw`` folder lives.
    """
    fn = f'CSS_{song_id.replace("_", "")}_Metre.csv'
    dir = f"{data_dir}/raw/" + song_id
    return pd.read_csv(f"{dir}/{fn}", index_col=0)


def load_metres(metadata, data_dir=DATA_DIR):
    """Compute mean cycle and pulse durations for each song in the metadata.

    Parameters
    ----------
    metadata:
        DataFrame containing song metadata indexed by song id.
    data_dir:
        Base data directory containing the raw metre CSV files.
    """
    metres = []
    for song_id, entry in metadata.iterrows():
        metre = load_metre_data(song_id, data_dir=data_dir)
        cycle_durations = np.diff(metre["Time"])
        metres.append(
            dict(
                song_id=song_id,
                cycle_mean=cycle_durations.mean(),
                pulse_mean=cycle_durations.mean() / 16,
            )
        )
    metres = pd.DataFrame(metres).set_index("song_id", drop=True)
    return metres


def load_metadata(path="metadata.csv", data_dir=DATA_DIR):
    """Load CSS metadata and join per-song metre stats.

    Parameters
    ----------
    path:
        Path to the metadata CSV (relative or absolute).
    data_dir:
        Base data directory containing the raw CSS files.
    """
    metadata = pd.read_csv(path, index_col=0)
    metadata["dir"] = f"{data_dir}/raw/" + metadata.index
    metres = load_metres(metadata, data_dir=data_dir)
    return metadata.join(metres)


def load_segments(song_num, data_dir=DATA_DIR):
    """Load rhythmic segments for all instruments in a single song.

    Parameters
    ----------
    song_num:
        CSS song number (1-based).
    data_dir:
        Base directory containing the ``raw`` CSS files.
    """
    df = pd.read_csv(
        f"{data_dir}/raw/Song_{song_num}/CSS_Song{song_num}_Onsets_Raw.csv"
    )

    if song_num == 3:
        # In 2790 the clave jumps back: 464.918789 -> 418.514076
        df = df.iloc[:2790, :]

    rss = []
    for instrument in ALL_INSTRUMENTS:

        if instrument not in df.columns:
            continue
        try:
            rss.append(
                RhythmicSegments.from_events(
                    df[instrument],
                    length=2,
                    drop_nan=True,
                    meta_constants=dict(
                        song_num=song_num, instrument=instrument.strip().lower()
                    ),
                )
            )
        except:
            print(f"skipping song {song_num}, {instrument}")

    return RhythmicSegments.concat(*rss)


def load_rhythmic_segments(data_dir=DATA_DIR):
    """Load rhythmic segments for all CSS songs and concatenate them.

    Parameters
    ----------
    data_dir:
        Base directory containing the ``raw`` CSS files.
    """
    return RhythmicSegments.concat(
        *[load_segments(i, data_dir=data_dir) for i in range(1, 6)]
    )
