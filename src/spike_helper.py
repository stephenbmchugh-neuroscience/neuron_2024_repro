# =============================
# spik_helper.py
# =============================
"""
spike_helper.py
--------
Spike, interval, and firing-rate utilities.
Pure numerical logic where possible; functions that require lab-specific
I/O are documented to indicate external dependencies.
"""

from typing import Iterable, List, Tuple, Optional
import os
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import smBaseFunctions3 as sbf
import vBaseFunctions3 as vbf


def generate_ifr_one_day(mouse,baseblock,desen,sessions,ctype_list,origIDs,nms,binwidth,ext_list):
    '''
    
    '''
    ##########################################################################################################
    cluID_list = sbf.multi_ctype_IDs(ctype_list,origIDs,mouse)
    n_bins = int(np.sum(nms) / binwidth)
    ##########################################################################################################
    ifr_dict = {}
    for ext_indx,ext in enumerate(ext_list):
        n_cells = len(cluID_list)
        output_array = np.array([]).reshape(n_bins,n_cells,0)
        for sindx,sess in enumerate(sessions):
            ### Select session
            sessionLabel = sbf.get_descode(desen,sess)
            fname = baseblock + '_' + str(sessionLabel['filebase'].index[0])
            ##############################################################################################
            res,clu = vbf.LoadSpikeTimes(fname,trode=None,MinCluId=2,res2eeg=(20000./20000))
            ipath = os.path.split(baseblock)[0] + '/'
            reftemp = get_pulsetimes(ipath,desen,sess,ext,tconv=None,debug=False)
            refTimes = reftemp['begin'].values
            if len(refTimes) > 1:
                refEdges = generate_edges(refTimes,binwidth=binwidth,nmsBefore=nms[0],nmsAfter=nms[1])
                temp_mat = generate_3Difr_matrix(res,clu,cluID_list,refEdges)
                output_array = np.concatenate((output_array,temp_mat),axis=-1)
            else:
                print(sess,'not enough', ext, 'pulses')
        okey = ext[1:].split('_',3)[0]
        ######################################################################################################
        print(okey.upper(),output_array.shape)
        ifr_dict[okey] = output_array
    ##########################################################################################################
    return ifr_dict
##############################################################################################################

def get_tconv(spk_sr: float = 20000.0, lfp_sr: float = 1250.0, trk_sr: float = 39.0625) -> dict:
    """
    Return a dictionary of conversion factors between common sampling rates.

    The returned dictionary contains keys that map conversion names to
    multiplicative factors. For example, `spk_ms` is the duration of one spike
    sample in milliseconds (1000 / spk_sr).

    Parameters
    ----------
    spk_sr : float
        Spike sampling rate in Hz.
    lfp_sr : float
        LFP sampling rate in Hz.
    trk_sr : float
        Tracking sampling rate in Hz.

    Returns
    -------
    dict
        Mapping of conversion strings to numeric factors.
    """
    tconv: dict = {}
    tconv['lfp_spk'] = (spk_sr / lfp_sr)
    tconv['lfp_trk'] = (trk_sr / lfp_sr)
    tconv['lfp_lfp'] = (lfp_sr / lfp_sr)
    tconv['trk_spk'] = (spk_sr / trk_sr)
    tconv['trk_lfp'] = (lfp_sr / trk_sr)
    tconv['spk_lfp'] = (lfp_sr / spk_sr)
    tconv['spk_trk'] = (trk_sr / spk_sr)
    tconv['spk_spk'] = (spk_sr / spk_sr)
    tconv['spk_ms'] = (1000.0 / spk_sr)
    tconv['ms_spk'] = (spk_sr * (1.0 / 1000.0))
    tconv['spk_sr'] = spk_sr
    tconv['lfp_sr'] = lfp_sr
    tconv['trk_sr'] = trk_sr
    return tconv

def get_pulsetimes(ipath,df,sesstype,ext='.audio_pulse',tconv=None,debug=False):
    '''
    input: path to the pulse file, desen(df), session string, pulse extension,
    output: returns a dataframe
    '''

    session = sbf.get_descode(df,sesstype)
    tempind = session.index[0]
    fname = ipath + '/' + str(df['filebase'][tempind]) + ext
    if debug:
        print(ipath,ext)
        print(fname)
    try:
        pulsetimes = pd.read_csv(fname, sep='\s+', header=None, names=['begin', 'end'])
    except:
        print('{} does not exist'.format(fname))
        pulsetimes = pd.DataFrame(columns=['begin', 'end'])
    ## Convert times (tconv) depending on whether spikes (none),lfps(1250/20000) or trk (39.0625/20000)
    if tconv is not None:
        pulsetimes['begin'] = (np.floor(pulsetimes['begin'] * tconv)).astype(int)
        pulsetimes['end'] = (np.floor(pulsetimes['end'] * tconv)).astype(int)    
    pulsetimes.index += 1

    return pulsetimes

def generate_fixed_intervals(
    width: float,
    maxT: float,
    minT: float = 0,
    tconv: float = 1.0,
) -> np.ndarray:
    """
    Create contiguous fixed-width integer intervals between `minT` and `maxT`.

    Parameters
    ----------
    width : float
        Width of intervals in the same units as `tconv` (e.g., seconds).
    maxT : float
        Maximum time (exclusive).
    minT : float
        Minimum time (inclusive).
    tconv : float
        Multiplicative factor to convert units to integer bins.

    Returns
    -------
    np.ndarray
        Array of shape (n_intervals, 2) with integer interval edges.
    """
    tmin = int(minT * tconv)
    tmax = int(maxT * tconv)
    width_int = max(int(width * tconv), 1)

    edges = np.arange(tmin, tmax + 1, width_int)
    if edges.size < 2:
        return np.empty((0, 2), dtype=int)

    intervals = np.stack([edges[:-1], edges[1:]], axis=1).astype(int)
    return intervals


def generate_edges(itimes: Iterable[int], binwidth: float, nmsBefore: float, nmsAfter: float, sr: float = 20000.0) -> List[np.ndarray]:
    """
    Generate arrays of sample indices (edges) around event times.

    Each returned element is an array of integer sample indices spanning
    [time - nmsBefore, time + nmsAfter] with step `binwidth` (converted into
    sample units using `sr`).

    Parameters
    ----------
    itimes : iterable of int
        Reference event times in samples.
    binwidth : float
        Bin width in milliseconds.
    nmsBefore : float
        Milliseconds before the reference time to include.
    nmsAfter : float
        Milliseconds after the reference time to include.
    sr : float
        Sampling rate (Hz) used to convert ms to samples.

    Returns
    -------
    List[np.ndarray]
        One array of sample indices per event.
    """
    oedges: List[np.ndarray] = []
    eachms = sr / 1000.0

    for val in itimes:
        startbin = int(val - eachms * nmsBefore)
        endbin = int(val + eachms * nmsAfter)
        if startbin > 0 and endbin > startbin:
            step = max(int(np.round(binwidth * eachms)), 1)
            oedges.append(np.arange(startbin, endbin + 1, step, dtype=int))
    return oedges


def pulse_edges(df: pd.DataFrame, nbins: int = 150, binwidth: int = 100, sr: float = 20000.0, fw: bool = False) -> List[np.ndarray]:
    """
    Create per-trial edge arrays for pulse intervals defined by a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'begin' and 'end' columns with sample indices.
    nbins : int
        Number of sample points to produce when `fw` is True (linear spacing).
    binwidth : int
        Bin width in ms used when `fw` is False.
    sr : float
        Sampling rate in Hz used to convert ms to samples.
    fw : bool
        If True, produce `nbins` equally-spaced indices between begin and end.

    Returns
    -------
    List[np.ndarray]
        List of sample-index arrays (one per row of df).
    """
    eachms = sr / 1000.0
    oedges: List[np.ndarray] = []

    for _, row in df.iterrows():
        startbin = int(row['begin'])
        endbin = int(row['end'])
        if fw:
            tempdat = np.round(np.linspace(startbin, endbin + 1, nbins)).astype(int)
        else:
            step = max(int(binwidth * eachms), 1)
            tempdat = np.arange(startbin, endbin + 1, step, dtype=int)
        oedges.append(tempdat)
    return oedges


def generate_ifr_matrix(res: np.ndarray, clu: np.ndarray, cluids: Iterable[int], edges: np.ndarray) -> np.ndarray:
    """
    Compute spike count histogram (IFR-like) for each cluster across a single set of bin edges.

    Parameters
    ----------
    res : np.ndarray
        Spike timestamps (sample units).
    clu : np.ndarray
        Cluster ID per spike (same length as res).
    cluids : iterable
        Cluster IDs to include (order determines column order).
    edges : array-like
        Bin edges (length n_bins+1) used for np.histogram.

    Returns
    -------
    np.ndarray
        2D array with shape (n_bins, n_cells) containing counts.
    """
    n_bins = len(edges) - 1
    n_cells = len(cluids)
    omatrix = np.zeros((n_bins, n_cells), dtype=float)

    for idx, val in enumerate(cluids):
        idata = res[clu == val]
        temp, _ = np.histogram(idata, bins=edges)
        omatrix[:, idx] = temp
    return omatrix


def generate_3Difr_matrix(res: np.ndarray, clu: np.ndarray, cluids: Iterable[int], edges_list: List[np.ndarray]) -> np.ndarray:
    """
    Build a 3D IFR matrix across multiple trials/windows.

    The output shape is (n_timebins, n_cells, n_trials) where n_timebins is
    derived from the first element of edges_list (len(edges)-1).
    """
    if len(edges_list) == 0:
        return np.zeros((0, len(cluids), 0))

    try:
        n_timebins = len(edges_list[0]) - 1
    except Exception:
        n_timebins = 0

    omatrix3d = np.zeros((n_timebins, len(cluids), len(edges_list)), dtype=float)

    for trial_idx, edges in enumerate(edges_list):
        omatrix3d[:, :, trial_idx] = generate_ifr_matrix(res, clu, cluids, edges)

    return omatrix3d


def calc_meanFR(res: np.ndarray, clu: np.ndarray, cluid_inds: Iterable[int], itimes: pd.DataFrame, sr: float = 20000.0) -> np.ndarray:
    """
    Compute per-cluster total spikes and mean firing rate over a collection of intervals.

    Returns an array of shape (n_cells, 4) where columns are:
      [cluID, spike_count_total, total_duration_samples, firing_rate_Hz]
    """
    intervalFR = np.zeros((len(cluid_inds), 4), dtype=float)

    sstart = itimes['begin'].values.astype(int)
    eend = itimes['end'].values.astype(int)

    for nn, clu_id in enumerate(cluid_inds):
        tempdat = res[clu == clu_id]
        tempCount = 0
        tempDur = 0
        for start, end in zip(sstart, eend):
            eachInterval = int(end - start)
            tempDur += eachInterval
            tempCount += int(np.sum((tempdat > start) & (tempdat < end)))

        intervalFR[nn, 0] = clu_id
        intervalFR[nn, 1] = tempCount
        intervalFR[nn, 2] = tempDur
        intervalFR[nn, 3] = (tempCount / tempDur) * sr if tempDur > 0 else np.nan

    return intervalFR


def calc_spk_counts(res: np.ndarray, clu: np.ndarray, cluid_inds: Iterable[int], itimes: pd.DataFrame, nTrials: int, sr: float = 20000.0) -> np.ndarray:
    """
    Compute per-cluster, per-trial spike counts and firing rates.

    Returns an array shaped (n_cells, 4, nTrials) where axis=1 contains
    [cluID, spike_count, duration_samples, firing_rate_Hz] for each trial.
    """
    intervalFR = np.zeros((len(cluid_inds), 4, nTrials), dtype=float)

    sstart = itimes['begin'].values.astype(int)
    eend = itimes['end'].values.astype(int)

    for cidx, cluID in enumerate(cluid_inds):
        tempdat = res[clu == cluID]
        for t_idx in range(min(nTrials, len(sstart))):
            start = sstart[t_idx]
            end = eend[t_idx]
            tempDur = int(end - start)
            tempCount = int(np.sum((tempdat > start) & (tempdat < end)))

            intervalFR[cidx, 0, t_idx] = cluID
            intervalFR[cidx, 1, t_idx] = tempCount
            intervalFR[cidx, 2, t_idx] = tempDur
            intervalFR[cidx, 3, t_idx] = (tempCount / tempDur) * sr if tempDur > 0 else np.nan

    return intervalFR


def calc_meanFR_eachTrial(res: np.ndarray, clu: np.ndarray, cluid_inds: Iterable[int], itimes: pd.DataFrame, nTrials: int = 20, sr: float = 20000.0) -> np.ndarray:
    """
    Compute an (n_cells x (nTrials+1)) matrix where column 0 is cluID and
    subsequent columns are firing rates per trial (Hz) rounded to 3 decimals.
    """
    intervalFR = np.zeros((len(cluid_inds), nTrials + 1), dtype=float)

    sstart = itimes['begin'].values.astype(int)
    eend = itimes['end'].values.astype(int)

    for nn, clu_id in enumerate(cluid_inds):
        tempdat = res[clu == clu_id]
        intervalFR[nn, 0] = clu_id
        for idx in range(min(nTrials, len(sstart))):
            eachInterval = int(eend[idx] - sstart[idx])
            spkCount = int(np.sum((tempdat > sstart[idx]) & (tempdat < eend[idx])))
            intervalFR[nn, idx + 1] = np.round((spkCount / (eachInterval / sr)), 3) if eachInterval > 0 else np.nan

    return intervalFR


def random_add(arr: np.ndarray, int_to_add: int = 1) -> np.ndarray:
    """
    Randomly add `int_to_add` to approximately half of elements in the
    first half of `arr`.

    NOTE: This function is nondeterministic; please seed numpy.random in
    tests to achieve reproducible results.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    mask = np.random.choice([False, True], size=len(arr) // 2, replace=True)
    result = arr.copy()
    result[: len(arr) // 2][mask] += int_to_add
    return result


def clean_clu(spikes: np.ndarray, thresh: int = 3, int_to_add: int = 0) -> np.ndarray:
    """
    Clean spike times by downsampling indices and removing spikes with short ISI.

    The original lab pipeline downsampled spike indices by 20 and removed
    entries where the inter-spike-interval (in the downsampled units) was
    less than `thresh`. This implementation mirrors that behavior and
    documents it for reproducibility.
    """
    x = (spikes / 20).astype(int)
    isi = np.diff(x)
    all_isi_inds = np.arange(x.shape[0])
    bad_isi_inds = np.squeeze(np.argwhere(isi < thresh) + 1) if isi.size > 0 else np.array([], dtype=int)
    good_isi_inds = np.setdiff1d(all_isi_inds, bad_isi_inds, assume_unique=False)
    return spikes[good_isi_inds]


def autocorrelation(
    baseblock: str,
    desen: pd.DataFrame,
    sessions: List[str],
    cluID: int,
    binwidth: float = 1.0,
    nms: Tuple[float, float] = (40.0, 40.0),
    clean_ref: bool = False,
    thresh: int = 3,
) -> np.ndarray:
    """
    Compute autocorrelation IFRs across multiple sessions for a reference cluster.

    Notes
    -----
    This function depends on external lab loaders:
    - `get_descode(desen, sess)` which maps a session label into filebase/index
    - `vbf.LoadSpikeTimes(fname, ...)` which returns (res, clu)

    The function aggregates trial IFRs into a 3D array (timebins x 1 x trials).
    """
    tbins = int(2 * (nms[0] / binwidth))
    oMat = np.zeros((tbins, 1, 0), dtype=float)

    for sindx, sess in enumerate(sessions):
        sessionLabel = get_descode(desen, sess)
        fname = f"{baseblock}_{int(sessionLabel['filebase'].index[0])}"
        res, clu = vbf.LoadSpikeTimes(fname, trode=None, MinCluId=2, res2eeg=(20000.0 / 20000.0))
        refSpikes = res[clu == cluID]
        if clean_ref:
            refSpikes = clean_clu(refSpikes, thresh=thresh)
        refEdges = generate_edges(refSpikes, binwidth=binwidth, nmsBefore=nms[0], nmsAfter=nms[1])
        try:
            tempMat = generate_3Difr_matrix(res, clu, [cluID], refEdges)
            oMat = np.concatenate((oMat, tempMat), axis=-1)
        except Exception:
            # session had no usable spikes or bins
            continue
    return oMat


def crosscorrelation(
    baseblock: str,
    desen: pd.DataFrame,
    sessions: List[str],
    refcluID: int,
    targcluID_list: List[int],
    binwidth: float = 1.0,
    nms: Tuple[float, float] = (40.0, 40.0),
) -> np.ndarray:
    """
    Compute cross-correlation IFRs between a reference cluster and target clusters across sessions.

    Returns
    -------
    np.ndarray
        3D array with shape (timebins, n_targets, total_trials)
    """
    tbins = int(2 * (nms[0] / binwidth))
    nCells = len(targcluID_list)
    oMat = np.zeros((tbins, nCells, 0), dtype=float)

    for sindx, sess in enumerate(sessions):
        sessionLabel = get_descode(desen, sess)
        fname = f"{baseblock}_{int(sessionLabel['filebase'].index[0])}"
        res, clu = vbf.LoadSpikeTimes(fname, trode=None, MinCluId=2, res2eeg=(20000.0 / 20000.0))
        refSpikes = res[clu == refcluID]
        refEdges = generate_edges(refSpikes, binwidth=binwidth, nmsBefore=nms[0], nmsAfter=nms[1])
        try:
            tempMat = generate_3Difr_matrix(res, clu, targcluID_list, refEdges)
            oMat = np.concatenate((oMat, tempMat), axis=-1)
        except Exception:
            continue
    return oMat


def pulsecorrelation(
    baseblock: str,
    desen: pd.DataFrame,
    sessions: List[str],
    cluID,
    binwidth: float = 1.0,
    nms: Tuple[float, float] = (40.0, 40.0),
    ext: str = '.light_pulse',
    min_dur: Optional[float] = None,
    max_dur: Optional[float] = None,
    samp_rate: float = 20000.0,
    single_clu: bool = True,
) -> np.ndarray:
    """
    Compute pulse-locked IFR matrices across sessions for one or more clusters.

    Parameters
    ----------
    cluID : int or iterable
        Single cluster ID or list of IDs to process.
    ext : str
        Event file extension used by lab (`.light_pulse` etc.).

    Notes
    -----
    Relies on `get_pulsetimes(ipath, desen, sess, ext, tconv=None, debug=False)`
    to return a DataFrame with 'begin' and 'end' columns (sample indices).
    """
    tbins = int(2 * (nms[0] / binwidth))
    oMat = np.zeros((tbins, 1 if single_clu else len(cluID), 0), dtype=float)

    cluID_list = [cluID] if single_clu else list(cluID)

    for sindx, sess in enumerate(sessions):
        sessionLabel = get_descode(desen, sess)
        fname = f"{baseblock}_{int(sessionLabel['filebase'].index[0])}"
        res, clu = vbf.LoadSpikeTimes(fname, trode=None, MinCluId=2, res2eeg=(20000.0 / 20000.0))
        ipath = str(Path(baseblock).parent) + '/'
        reftemp = get_pulsetimes(ipath, desen, sess, ext, tconv=None, debug=False)
        duration = [(y - x) for x, y in zip(reftemp['begin'].values, reftemp['end'].values)]
        refTimes = np.unique(reftemp['begin'].values)
        if min_dur is not None:
            refTimes = [x for (x, y) in zip(refTimes, duration) if y > min_dur * samp_rate]
            min_pulses = 1
        if max_dur is not None:
            refTimes = [x for (x, y) in zip(refTimes, duration) if y < max_dur * samp_rate]
        else:
            min_pulses = 1
        if len(refTimes) > min_pulses:
            refEdges = generate_edges(refTimes, binwidth=binwidth, nmsBefore=nms[0], nmsAfter=nms[1])
            tempMat = generate_3Difr_matrix(res, clu, cluID_list, refEdges)
            oMat = np.concatenate((oMat, tempMat), axis=-1)
    return oMat


def pulsecorrelation_multi(
    baseblock: str,
    desen: pd.DataFrame,
    sessions: List[str],
    cluID_list: List[int],
    binwidth: float = 1.0,
    nms: Tuple[float, float] = (40.0, 40.0),
    ext: str = '.light_pulse',
    min_dur: Optional[float] = None,
    max_dur: Optional[float] = None,
    samp_rate: float = 20000.0,
) -> np.ndarray:
    """
    Multi-cluster pulse correlation across sessions returning (tbins, n_clusters, total_trials).
    """
    tbins = int(2 * (nms[0] / binwidth))
    oMat = np.zeros((tbins, len(cluID_list), 0), dtype=float)

    for sindx, sess in enumerate(sessions):
        sessionLabel = get_descode(desen, sess)
        fname = f"{baseblock}_{int(sessionLabel['filebase'].index[0])}"
        res, clu = vbf.LoadSpikeTimes(fname, trode=None, MinCluId=2, res2eeg=(20000.0 / 20000.0))
        ipath = str(Path(baseblock).parent) + '/'
        reftemp = get_pulsetimes(ipath, desen, sess, ext, tconv=None, debug=False)
        duration = [(y - x) for x, y in zip(reftemp['begin'].values, reftemp['end'].values)]
        refTimes = reftemp['begin'].values
        if min_dur is not None:
            refTimes = [x for (x, y) in zip(refTimes, duration) if y > min_dur * samp_rate]
            min_pulses = 1
        if max_dur is not None:
            refTimes = [x for (x, y) in zip(refTimes, duration) if y < max_dur * samp_rate]
        else:
            min_pulses = 0
        if len(refTimes) > min_pulses:
            refEdges = generate_edges(refTimes, binwidth=binwidth, nmsBefore=nms[0], nmsAfter=nms[1])
            tempMat = generate_3Difr_matrix(res, clu, cluID_list, refEdges)
            oMat = np.concatenate((oMat, tempMat), axis=-1)
    return oMat


def gen_prob_2d(idata: np.ndarray) -> np.ndarray:
    """
    Normalize a (time x trials) count matrix into a 1D probability distribution over time.

    Returns zeros if the input sum is zero to avoid division by zero.
    """
    total = np.nansum(idata)
    if total == 0:
        return np.zeros(idata.shape[0])
    return np.nansum(idata, axis=1) / total


def zscore_1dList(idata: Iterable[float]) -> List[float]:
    """
    Z-score a 1D iterable, returning a concrete Python list (useful for JSON
    serialization in reporting).
    """
    arr = np.asarray(idata, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if sd == 0:
        return [0.0 for _ in arr]
    return [float((x - mu) / sd) for x in arr]


def z_test_model(idata: np.ndarray, isem: np.ndarray, combomat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform pairwise z-tests for pairs defined in `combomat`.

    Parameters
    ----------
    idata : np.ndarray
        Array of means per condition.
    isem : np.ndarray
        Array of SEMs per condition.
    combomat : np.ndarray
        Shape (n_pairs, 2) with indices to compare.

    Returns
    -------
    (z_values, p_values)
    """
    z = np.zeros((combomat.shape[0],), dtype=float)
    p = np.zeros((combomat.shape[0],), dtype=float)
    for indx, pair in enumerate(combomat):
        indd1 = int(pair[0])
        indd2 = int(pair[1])
        denom = (isem[indd1] + isem[indd2]) / 2.0
        z[indx] = (idata[indd1] - idata[indd2]) / denom if denom != 0 else np.nan
        p[indx] = stats.norm.sf(abs(z[indx])) * 2.0 if not np.isnan(z[indx]) else np.nan
    return z, p


def hist2(data1: np.ndarray, data2: np.ndarray, nbins: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Compute 2D histogram and return counts, x-centers, y-centers, and edges.

    Returns
    -------
    hist, xBinCenters, yBinCenters, [xedges, yedges]
    """
    hist, xedges, yedges = np.histogram2d(data1, data2, bins=nbins)
    # compute bin centers from edges
    xBinCenters = 0.5 * (xedges[:-1] + xedges[1:])
    yBinCenters = 0.5 * (yedges[:-1] + yedges[1:])
    Edges = [xedges, yedges]
    return hist, xBinCenters, yBinCenters, Edges

