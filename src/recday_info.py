# =============================
# recday_info.py
# =============================
"""
recday_info.py
--------
day and session metadata loaders.
Designed for safe loading, clear failures, and compatibility with the lab's
text descriptor files.

Key behaviors:
- Read per-tetrode descriptor files and optionally a combined descriptor.
- Return pandas DataFrames for downstream analysis.
"""

from typing import Optional, Tuple, List
import pandas as pd
from pathlib import Path
import numpy as np


def load_units(
    baseblock: str,
    par: Optional[dict] = None,
    each_trode_ext: str = ".des.",
    all_trode_ext: str = ".desf",
) -> Optional[pd.DataFrame]:
    """
    Load unit (descriptor) files for a single recording baseblock.

    The function tries the following (in order):
    1. Read per-tetrode descriptor files (baseblock + each_trode_ext + trode_index)
    2. Read a combined descriptor file (baseblock + all_trode_ext) and prefer its
       contents when present.

    Parameters
    ----------
    baseblock : str
        Path prefix for session files (e.g., '/path/to/msm04-160721/msm04-160721').
    par : dict, optional
        Optional parameters (e.g., trode_ch) to determine number of tetrodes.
    each_trode_ext : str, optional
        Extension pattern for per-tetrode files.
    all_trode_ext : str, optional
        Extension pattern for combined descriptor file.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with a column 'des' containing descriptors, or None on fatal
        failure. The DataFrame index is shifted by +2 to match lab numbering.
    """
    try:
        trode_index = range(1, 1 + (len(par["trode_ch"]) if par and "trode_ch" in par else 4))
        units_list = []
        for t in trode_index:
            pth = f"{baseblock}{each_trode_ext}{t}"
            try:
                df = pd.read_csv(pth, header=None, names=["des"])
                units_list.append(df)
            except FileNotFoundError:
                units_list.append(pd.DataFrame({"des": []}))

        if units_list:
            units = pd.concat(units_list, keys=trode_index, names=["trode", "trode_unit"]).reset_index()
        else:
            units = pd.DataFrame()

        try:
            combined = pd.read_csv(f"{baseblock}{all_trode_ext}", header=None, names=["des"])
            # If combined descriptor exists, use it (compatibility with legacy workflows)
            units["des"] = combined["des"].values
        except FileNotFoundError:
            # No combined file: keep per-tetrode descriptors
            pass

        # Shift indexes to match original lab convention (they started at 2)
        units.index = units.index + 2
        return units

    except Exception as e:
        print(f"Failed to load units for {baseblock}: {e}")
        return None


def get_mouse_info_single(eachdir: str, par_loader: Optional[callable] = None) -> Tuple[str, str, str, Optional[dict], pd.DataFrame]:
    """
    Gather basic info for a single mouse/session directory.

    This helper does not change the process working directory. It returns a
    tuple (ipath, bsnm, baseblock, par, desen) where `par` and `desen` may be
    None/empty if loaders are not available or files are missing.
    """
    ipath = str(Path(eachdir))
    bsnm = Path(ipath).name
    baseblock = str(Path(ipath) / bsnm)

    par = None
    desen = pd.DataFrame()
    try:
        if par_loader is not None:
            par = par_loader(baseblock)
    except Exception:
        par = None

    try:
        if par_loader is not None and hasattr(par_loader, 'LoadStages'):
            desen = par_loader.LoadStages(baseblock)
        else:
            desen = pd.DataFrame()
    except Exception:
        desen = pd.DataFrame()

    return ipath, bsnm, baseblock, par, desen


def mouse_info_data_share(mpath: str, ipath: str) -> Tuple[List[str], List[str], List[dict], List[pd.DataFrame], List[object]]:
    """
    Load shared metadata arrays saved as .npy files for distribution.

    Expected files in `mpath/ipath/`: mouseID, allBaseblock, allPar, alldesen, units
    Each is loaded with `allow_pickle=True` to preserve object structures.

    Returns
    -------
    Tuple
        (mouseID, allBaseblock, allPar, alldesen, units)
    """
    base = Path(mpath) / ipath
    mouseID = list(np.load(str(base / 'mouseID'), allow_pickle=True))
    allBaseblock = list(np.load(str(base / 'allBaseblock'), allow_pickle=True))
    allPar = list(np.load(str(base / 'allPar'), allow_pickle=True))
    alldesen = list(np.load(str(base / 'alldesen'), allow_pickle=True))
    units = list(np.load(str(base / 'units'), allow_pickle=True))
    return mouseID, allBaseblock, allPar, alldesen, units


def get_all_mouse_db_info(
    database: List[str],
    SF: bool = True,
    each_trode_ext: str = '.des.',
    all_trode_ext: str = '.desf',
    par_loader: Optional[callable] = None,
):
    """
    Collect metadata (units, stages, params) over a list of directories.

    Parameters
    ----------
    database : list of str
        List of session directories to process.
    SF : bool
        If True and a stage loader is available, call `update_desen` (caller
        must provide this function in the namespace). This keeps behavior
        compatible with the legacy code.
    par_loader : callable, optional
        Optional loader (e.g., vbf.LoadPar) used to populate `par` and
        optionally `LoadStages`.

    Returns
    -------
    Tuple
        (mouseID, allBaseblock, allPar, alldesen, units_list)
    """
    units_list = []
    mouseID = []
    alldesen = []
    allBaseblock = []
    allPar = []

    for eachdir in database:
        ipath = str(Path(eachdir))
        bsnm = Path(ipath).name
        baseblock = str(Path(ipath) / bsnm)
        allBaseblock.append(baseblock)

        par = None
        try:
            if par_loader is not None:
                par = par_loader(baseblock)
        except Exception:
            par = None
        allPar.append(par)

        try:
            units_df = load_units(baseblock, par=par, each_trode_ext=each_trode_ext, all_trode_ext=all_trode_ext)
        except Exception:
            units_df = None
        units_list.append(units_df)

        try:
            if par_loader is not None and hasattr(par_loader, 'LoadStages'):
                des_df = par_loader.LoadStages(baseblock)
            else:
                des_df = pd.DataFrame()
        except Exception:
            des_df = pd.DataFrame()

        if SF and not des_df.empty:
            # `update_desen` is a legacy helper expected in the caller namespace.
            try:
                des_df = update_desen(des_df)
            except Exception:
                pass

        alldesen.append(des_df)
        mouseID.append(bsnm)

    return mouseID, allBaseblock, allPar, alldesen, units_list

