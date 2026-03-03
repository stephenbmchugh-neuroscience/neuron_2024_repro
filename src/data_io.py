# =============================
# data_io.py
# =============================
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import datetime
import os
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------
# Database and summary data utilities
# ---------------------------------------------------------------------

def data_loader(
    BASE_DIR: str,
    path_to_data: str,
    ctype_list: List[str],
    event_list: List[str] = ['ds', 'swr', 'ds1', 'ds2'],
    file_ext: str = '.npy'
) -> Dict[str, Dict[str, Any]]:
    """
    Load summary data for multiple event types.

    Parameters
    ----------
    BASE_DIR : str
        Base directory containing the dataset.
    path_to_data : str
        Relative path from BASE_DIR to the data folder.
    ctype_list : List[str]
        List of content/type identifiers passed to `get_summary_data`.
    event_list : List[str], optional
        List of event subdirectories to process
        (e.g., ['ds', 'swr', 'ds1', 'ds2']).
    file_ext : str, optional
        File extension of the data files, by default '.npy'.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Nested dictionary structured as:
            {
                event_key: {
                    content_type: loaded_data
                }
            }

    Notes
    -----
    For each event in `event_list`, this function calls
    `get_summary_data` on the corresponding subdirectory.
    """
    path_to_dir = str(Path(BASE_DIR, path_to_data))

    all_summary_data: Dict[str, Dict[str, Any]] = {}

    for event_key in event_list:
        ipath = f"{path_to_dir}/{event_key}"
        temp_dat = get_summary_data(
            ipath, ctype_list, file_ext, pprint=False
        )
        all_summary_data[event_key] = temp_dat

    return all_summary_data

def get_summary_data(
    ipath: str,
    ext_pt1: List[str],
    ext_pt2: str,
    pprint: bool = True
) -> Dict[str, Any]:
    """
    Collect summary data files from a directory based on file type extensions.

    Parameters
    ----------
    ipath : str
        Path to the directory containing the data files.
    ext_pt1 : List[str]
        List of file type identifiers (e.g., ['type1', 'type2']).
        Each entry is used to construct a file extension pattern.
    ext_pt2 : str
        File extension suffix (e.g., '.npy').
    pprint : bool, optional
        Whether to print file information during loading, by default True.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping each file type (from ext_pt1) to the
        corresponding loaded data returned by `get_files`.

    Notes
    -----
    This function assumes the existence of a `get_files` function
    with signature similar to:
        get_files(path, extension, rev=False, npy=True, pprint=True)
    """
    data_files: Dict[str, Any] = {}
    fid: List[str] = []

    for ftype in ext_pt1:
        extt1 = "_" + ftype + ext_pt2
        data_files[ftype], fid = get_files(
            ipath, extt1, rev=False, npy=True, pprint=pprint
        )

    return data_files

def get_files(
    path: str,
    extt: str,
    rev: bool = False,
    npy: bool = True,
    pprint: bool = True
) -> Tuple[List[npt.NDArray], List[str]]:
    """
    Load files from a directory matching a given extension pattern.

    Parameters
    ----------
    path : str
        Directory path to search for files.
    extt : str
        File extension pattern to match (e.g., '_type.npy').
        Only files ending with this string are loaded.
    rev : bool, optional
        If True, sort files in reverse order, by default False.
    npy : bool, optional
        If True, load files using `numpy.load`. If False,
        load using `numpy.loadtxt`, by default True.
    pprint : bool, optional
        If True, print matched filenames during loading,
        by default True.

    Returns
    -------
    Tuple[List[numpy.ndarray], List[str]]
        A tuple containing:
        - List of loaded NumPy arrays.
        - List of corresponding file names.

    Notes
    -----
    - Uses `os.walk` to traverse the directory (non-recursive behavior
      since only filenames from the given `path` are loaded).
    - `allow_pickle=True` is enabled when loading `.npy` files.
    """
    iarray: List[npt.NDArray] = []
    fileID: List[str] = []

    for root, dirs, files in os.walk(path):
        for filen in sorted(files, reverse=rev):
            if filen.endswith(extt):
                if pprint:
                    print(f"{extt} files are: {filen}")

                fileID.append(filen)
                fullpath = os.path.join(root, filen)

                if npy:
                    iarray.append(np.load(fullpath, allow_pickle=True))
                else:
                    iarray.append(np.loadtxt(fullpath))

    return iarray, fileID

def read_db(
    filename: str,
    ipath: str,
    mpath: str = "/mnt/smchugh",
    omit: bool = False,
    printpath: bool = True,
) -> List[str]:
    """
    Read a database file listing relative paths and return resolved paths.

    The expected format of the database file is one path per line. Paths may
    be relative to a mount point (`mpath`) or absolute. The function does not
    change the current working directory and will return full paths (unless
    `omit=True`).

    Parameters
    ----------
    filename : str
        Name of the database file (e.g., 'day_group.db').
    ipath : str
        Directory under `mpath` containing the database file. May be absolute
        or relative.
    mpath : str, optional
        Root mount path to prepend to relative entries in the db file. Defaults
        to '/mnt/smchugh'.
    omit : bool, optional
        If True, entries are returned exactly as they appear in the file and
        not joined to `mpath`.
    printpath : bool, optional
        If True, the resolved path to the database file is printed.

    Returns
    -------
    List[str]
        List of resolved path strings.

    Raises
    ------
    FileNotFoundError
        If the database file cannot be opened.
    """
    db_path = Path(mpath) / ipath.strip("/") / filename
    if printpath:
        print(f"Reading DB file: {str(db_path)}")

    out: List[str] = []
    with db_path.open("r") as fh:
        for line in fh:
            entry = line.strip()
            out.append(entry if omit else str(Path(mpath) / entry))
    return out


def write_to_text(
    output_dir: str,
    fname: str,
    odata: Iterable,
    extension: str,
    fmt: str = "%d",
) -> str:
    """
    Save an iterable (list/ndarray) to a text file.

    This helper ensures the destination directory exists and returns the
    absolute path of the saved file. Useful for saving interval lists,
    timestamps, or small summary tables.

    Parameters
    ----------
    output_dir : str
        Directory to save the file. Will be created if missing.
    fname : str
        Base filename (without extension).
    odata : Iterable
        Data to write. Will be converted to a numpy array for saving.
    extension : str
        Extension including the leading dot (e.g. '.txt').
    fmt : str, optional
        Format string for np.savetxt (default '%d').

    Returns
    -------
    str
        Full path to the saved file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fullpath = out_dir / f"{fname}{extension}"
    np.savetxt(str(fullpath), np.asarray(list(odata)), fmt=fmt)
    return str(fullpath)


def save_npy_data(
    mpath: str,
    output_path: str,
    fname: str,
    odata: object,
)	 -> str:
    """
    Save a Python object to a .npy file under `mpath/output_path/`.

    Parameters
    ----------
    mpath : str
        Root path to prepend (e.g. lab mount).
    output_path : str
        Subdirectory under the mount to save to.
    fname : str
        Filename including extension (e.g. 'mouseID.npy').
    odata : object
        Object to save; must be numpy-serializable.

    Returns
    -------
    str
        Full path to the saved .npy file.
    """
    out_dir = Path(mpath) / output_path
    out_dir.mkdir(parents=True, exist_ok=True)
    fullpath = out_dir / fname
    with fullpath.open("wb") as fh:
        np.save(fh, odata)
    return str(fullpath)


def savefig(
    fig,
    opath: str,
    ftitle: str,
    ext: str = ".svg",
) -> str:
    """
    Save a matplotlib figure to disk and return the saved path.

    The function optionally performs a legacy path replacement to support
    older lab directory conventions. It prints a timestamped confirmation so
    automated workflows write a brief log when saving figures.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    opath : str
        Target directory path.
    ftitle : str
        Base filename (no extension).
    ext : str, optional
        Extension including leading dot (default '.svg').
    dl : bool, optional
        If True, apply legacy mount replacement for compatibility.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """

    out_dir = Path(opath)
    out_dir.mkdir(parents=True, exist_ok=True)

    fullpath = out_dir / f"{ftitle}{ext}"
    fig.savefig(str(fullpath))

    now = datetime.datetime.now()
    print(f"Saved {fullpath} at {now.strftime('%Y-%m-%d %H:%M:%S')}")

    return str(fullpath)


def get_database(
    dbname: str,
    ipath: str,
    mpath: str = "/mnt/smchugh",
    old_mpath: str = "/mnt/smchugh/",
    new_mpath: str = "",
    update: bool = True,
    pprint: bool = True,
) -> List[str]:
    """
    Read a named database file and optionally replace mount prefixes.

    This is a thin convenience wrapper over `read_db` that supports path
    substitution (useful when copying a repo between systems with different
    mounts).

    Parameters
    ----------
    dbname, ipath, mpath : see `read_db`.
    old_mpath : str
        String to replace in returned paths when `update` is True.
    new_mpath : str
        Replacement for `old_mpath`.
    update : bool
        Whether to perform the path substitution.
    pprint : bool
        If True, print the resulting list of paths.

    Returns
    -------
    list of str
        Resolved (and possibly updated) database entries.
    """
    db_entries = read_db(dbname, ipath, mpath=mpath, omit=False, printpath=True)
    if update and new_mpath:
        db_entries = [entry.replace(old_mpath, new_mpath) for entry in db_entries]
    if pprint:
        print()
        for idx, val in enumerate(db_entries):
            print(idx, val)
    return db_entries


def get_database_dl(
    day_type: str,
    group_type: str,
    old_mpath: str = '/lfpd4/SF/',
    mpath: str = '/Dupret_Lab/',
    update: bool = True,
    pprint: bool = True,
) -> List[str]:
    """
    Convenience wrapper to fetch Dupret_Lab-style databases.

    Parameters
    ----------
    day_type : str
        e.g., 'day' or 'night'
    group_type : str
        group identifier used in filename construction.

    Returns
    -------
    list of str
        Database entries.
    """
    new_mpath = '/Dupret_Lab/merged/smchugh_merged/'
    ipath = '/analysis/smchugh_analysis/databases'
    dbname = f"{day_type}_{group_type}.db"
    return get_database(dbname, ipath, mpath=mpath, old_mpath=old_mpath, new_mpath=new_mpath, update=update, pprint=pprint)
    
    

