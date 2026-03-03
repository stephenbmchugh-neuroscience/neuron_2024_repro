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


def get_mouse_info_single(BASE_DIR: str, eachdir: str):
    """
    Gather basic info for a single mouse/session directory.

    This helper does not change the process working directory. It returns a
    tuple (ipath, bsnm, baseblock, par, desen) where `par` and `desen` may be
    None/empty if loaders are not available or files are missing.
    """
    import vBaseFunctions3 as vbf
    
    ipath = str(Path(BASE_DIR,eachdir))
    bsnm = Path(ipath).name
    baseblock = str(Path(ipath) / bsnm)

    par = vbf.LoadPar(baseblock)
    desen = vbf.LoadStages(baseblock)
    desen = update_desen(desen)
    units = load_units(baseblock,par,each_trode_ext= ".des.",all_trode_ext=".des")

    return ipath, bsnm, baseblock, par, desen, units


def update_desen(df:pd.DataFrame,nospaces=True):
    '''
    clear white space for desen entries
    '''
    if df['filebase'].iloc[0].startswith('sm'):
        df['filebase'] = 'm' + df['filebase']
    if nospaces:
        df['desen'] = df['desen'].str.replace(" ", "")
    return df

def get_sessions(desenDict,sleepbox=False,sleeponly=False):
    '''

    '''
    if sleepbox:
        sessions = [x for x in desenDict['desen'].values]
    else:
        sessions = [x for x in desenDict['desen'].values if not x.startswith('sb')]
        sessions = [x for x in oSess if not x.startswith('ss')]
    if sleeponly:
        sessions = [x for x in desenDict['desen'].values if x.startswith('sb')]

    return sessions

def get_descode(df,sesstype,debug=False,exact=False):
    '''
    helper file to get session_level filenames from the desen dataframe
    '''
    descode = df[df['desen'].str.contains(sesstype)==True]
    if exact:
        descode = df[df['desen'].str.strip()==sesstype]
    if debug:	
        print(descode)

    return descode
       
def get_cell_inds(ctype,units,exact=True):
    '''

    '''
    if exact:
        cell_inds = [index+2 for index, value in enumerate(units['des'].values)
                          if ctype == value]
    else:
        cell_inds = [index+2 for index, value in enumerate(units['des'].values)
                          if ctype in value]

    return cell_inds
########################################################################################
def get_cell_inds_one_mouse(units,ctype_list=['pdg','p3','p1'],exact=True):
    '''
    '''

    origIDs = {}
    for cindx,ctype in enumerate(ctype_list):
        origIDs[ctype] = get_cell_inds(ctype,units,exact=exact)

    return origIDs
    
def multi_ctype_IDs(clist,origIDs,mouse=None):
    '''
    get the unit_ids for all cells in clist using the dictioary origIDs
    '''
    odata = []
    for cindx,ctype in enumerate(clist):
        if mouse is not None:
            odata += origIDs[ctype][mouse]
        else:
            odata += origIDs[ctype]
        
    return odata
    
def display_unit_df(units,ctype_list=['pdg','p3','p1']):
    '''

    '''
    for ctype in ctype_list:
        display(units[units['des'] == ctype])

