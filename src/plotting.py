# =============================
# plotting.py
# =============================
"""
plotting.py
-----------
Matplotlib plotting helpers used to reproduce figures.
Some plotting functions accept or return (fig, ax) so the caller may further
customize or save using `data_io.savefig`.
"""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as Ticker
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import smBaseFunctions3 as sbf
from my_mpl_defaults import *
from analysis import bin_array


def set_plot_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def figure_chooser(figure_panel: int) -> List[str]:
    """
    Return the list of event types corresponding to a given figure panel.

    Parameters
    ----------
    figure_panel : int
        The figure panel identifier. 
        - 1 returns ['ds', 'swr']
        - 2 returns ['ds1', 'ds2']

    Returns
    -------
    List[str]
        A list of event type strings associated with the specified panel.

    Raises
    ------
    ValueError
        If an unsupported figure_panel value is provided.
    """
    if figure_panel == 1:
        event_type_list = ['ds', 'swr']
    elif figure_panel == 2:
        event_type_list = ['ds1', 'ds2']
    else:
        raise ValueError(f"Unsupported figure_panel: {figure_panel}")
    return event_type_list

def cm2inch(*tupl: float) -> Tuple[float, ...]:
    """Convert centimeters to inches for matplotlib figsize arguments.

    Example
    -------
    >>> cm2inch(8, 6)
    (3.1496..., 2.3622...)
    """
    return tuple(x / 2.54 for x in tupl)


def get_fontdict() -> dict:
    """
    Configure matplotlib to use a reproducible sans-serif font and return a
    small font dictionary for text annotations.
    """
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = "Arial"
    return {"family": "sans-serif", "color": "k", "weight": "normal", "size": 10}


def plot_ac(
    iMat: np.ndarray,
    refCluID: int,
    units: List[Optional[pd.DataFrame]],
    mouse: int,
    nms: Tuple[float, float] = (40.0, 40.0),
    binwidth: float = 1.0,
    baseline: Optional[float] = None,
    figsize: Tuple[float, float] = (8.0, 8.0),
    xlim: Optional[Tuple[float, float]] = None,
    ac: bool = True,
    prob: bool = False,
    zscore: bool = False,
    ax: Optional[object] = None,
):
    """
    Plot an auto- or cross-correlogram as a bar plot.

    Parameters
    ----------
    iMat : np.ndarray
        IFR matrix (timebins, n_cells?, trials).
    refCluID : int
        Cluster id used for labeling.
    units : list
        List of unit DataFrames used to obtain descriptive labels.
    mouse : int
        Index into units to select which unit DataFrame to use.
    nms : tuple
        (ms_before, ms_after) for x-axis.
    binwidth : float
        Bin width in ms.
    baseline : float or None
        If provided and zscore is True, baseline window (ms) used to compute z.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    idata = np.nansum(iMat, axis=-1)

    if prob:
        denom = np.nansum(iMat, axis=(0, -1))
        idata = idata / denom if denom != 0 else idata

    if zscore:
        midpoint = int(nms[0] / binwidth)
        startpoint = 0 if baseline is None else max(0, midpoint - int((1000.0 / binwidth) * baseline))
        baseline_vals = idata[startpoint:midpoint, 0] if idata.ndim >= 2 else idata[startpoint:midpoint]
        mean_baseline = np.nanmean(baseline_vals)
        std_baseline = np.nanstd(baseline_vals)
        idata = (idata - mean_baseline) / std_baseline if std_baseline != 0 else idata

    zero_ind = int(nms[0] / binwidth)
    if ac and idata.ndim >= 2:
        idata[zero_ind] = 0

    xpts = np.arange(-nms[0], nms[1], binwidth)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=cm2inch(*figsize))
    else:
        fig = ax.figure

    ax.bar(xpts, np.squeeze(idata), width=binwidth, color='k', align='edge')
    if xlim is None:
        ax.set_xlim(-nms[0], nms[1])
    else:
        ax.set_xlim(-xlim[0], xlim[1])

    ax.set_ylabel('Counts' if not prob and not zscore else ('Probability' if prob else 'Z-score'))
    ax.set_xlabel('Time (ms)')

    label = ''
    try:
        label = units[mouse]['des'].iloc[refCluID]
    except Exception:
        label = str(refCluID)
    ax.set_title(f"{refCluID}_{label}")

    return fig, ax


def plot_group_pulse(
    iMat: np.ndarray,
    nms: Tuple[float, float] = (40.0, 40.0),
    binwidth: float = 1.0,
    barcol: str = 'k',
    figsize: Tuple[float, float] = (8.0, 8.0),
    xlim: Optional[Tuple[float, float]] = None,
    av: str = 'median',
    savgol: bool = False,
    npts: int = 3,
    ax: Optional[object] = None,
):
    """
    Plot a group-averaged pulse response as a bar chart.

    Parameters
    ----------
    iMat : np.ndarray
        Input IFR matrix (time x cells x trials) or (time x trials).
    av : 'mean' or 'median'
        Aggregation across trials.
    savgol : bool
        If True, apply Savitzky-Golay smoothing with window length `npts`.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    if av == 'mean':
        idata = np.nanmean(iMat, axis=-1)
    else:
        idata = np.nanmedian(iMat, axis=-1)

    idata = np.squeeze(idata)

    if savgol and npts > 1:
        wl = min(npts, idata.shape[0] - (1 - (idata.shape[0] % 2)))
        if wl % 2 == 0:
            wl += 1
        try:
            idata = savgol_filter(idata, wl, 1)
        except Exception:
            pass

    xpts = np.arange(-nms[0], nms[1], binwidth)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=cm2inch(*figsize))
    else:
        fig = ax.figure

    ax.bar(xpts, idata, color=barcol, align='edge', width=binwidth)
    if xlim is None:
        ax.set_xlim(-nms[0], nms[1])
    else:
        ax.set_xlim(-xlim[0], xlim[1])

    ax.set_ylabel('Z-score')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Group mean')

    return fig, ax


def plot_group_average(
    lightC2: np.ndarray,
    group_type: str,
    ctype: str,
    ylim: Tuple[float, float] = (-0.8, 0.8),
    ytick_width: float = 0.4,
    figsize: Tuple[float, float] = (10.0, 6.0),
    fscale: Tuple[int, int] = (14, 16),
    pulse_col: str = '#ffff00',
    box: bool = True,
    box_col: str = '#ffcc00ff',
    err_bar: bool = False,
    ax: Optional[object] = None,
):
    """
    High-level wrapper for plotting group-averaged responses with sensible defaults.

    Parameters
    ----------
    lightC2 : np.ndarray
        Data to plot (time x ...)
    group_type : str
        Domain-specific preset name, e.g. 'chr2'.
    ctype : str
        Cell-type name for title annotation.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.reset_orig()
    except Exception:
        pass

    if group_type == 'chr2':
        nms = (80.0, 80.0)
        binwidth = nms[0] / 40.0
        pulse_width = 5
        midpoint = int(nms[0] / binwidth)
        baseline = None
        xlab = 'Time (ms)'
        ylab = 'Z-score'
    else:
        nms = (15000.0, 15000.0)
        binwidth = 250.0
        pulse_width = 15000
        midpoint = int(nms[0] / binwidth)
        baseline = 15
        xlab = 'Time (ms)'
        ylab = 'Z-score FR'

    try:
        idata = np.nanmean(lightC2, axis=-1)
    except Exception:
        idata = lightC2

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=cm2inch(*figsize))
    else:
        fig = ax.figure

    xpts = np.arange(-nms[0], nms[1], binwidth)
    ax.plot(xpts, np.squeeze(idata), linewidth=0.5)
    ax.set_xlim(-nms[0], nms[1])
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f"Group: {group_type} | Cell type: {ctype}")

    return fig, ax
    
################################################################################################################
def plot_event_raster(actmat,
                      nms=(200,200),
                      binwidth=1,
                      fwd=1.8,fht=2.8,
                      col='k',
                      x_maj_loc=200,
                      y_maj_loc=10,
                      yt_fmt='.0f',
                      despine_x=False,
                      despine_y=False,
                      lw=0.5,
                      pad=3,
                      size_scale=2,
                      fscale=(6,7),
                      ylab='Event #',
                      xlab='Time (ms)'):
    '''
    
    '''
    actmat_ = actmat.T
    spk2d = [np.where(actmat_[cid,:]) for cid in range(actmat_.shape[0])]
    
    colors_ =  [col for _ in range(len(spk2d))]
    lengths_ = [.9 for  _ in range(len(spk2d))]
    idx_ = np.arange(len(spk2d))
    spk2d = [sbf.flatten(spk2d[i]) for i in idx_]
    ######################################################################################
    fwd = fwd * size_scale
    fht = fht * size_scale
    fig,ax = plt.subplots(1,1,figsize=cm2inch(fwd,fht))

    ax.eventplot(spk2d,color=colors_,linewidths=lw*1.5,linelengths=lengths_)

    ax.set_xlim(0,actmat_.shape[1])
    ax.set_ylim(0,actmat_.shape[0]);

    ax.xaxis.set_major_locator(Ticker.MultipleLocator(x_maj_loc))
    ax.yaxis.set_major_locator(Ticker.MultipleLocator(y_maj_loc))

    ax = sbf.adjust_plot_pub(ax,
                             xlab=xlab,
                             ylab=ylab,
                             nms=nms,
                             binwidth=binwidth,
                             raster=True,
                             fscale=fscale,
                             grid=False,
                             pad=pad)
    
    if despine_x:
        sns.despine(bottom=True)
        ax.set_xticks([])
    if despine_y:
        sns.despine(bottom=True,left=True)
        ax.set_yticks([])
        
    return fig,ax
#######################################################################################################################################
def plot_psth(actmat,
              z_score=True,
              nms=(200,200),
              binwidth=1,
              zrange=(0,200),
              fwd=3,
              fht=1.5,
              col='k',
              x_maj_loc=200,
              y_maj_loc=4,
              yt_fmt='.0f',
              set_lims=True,
              ylim=(-2,8),
              xlab='Time (ms)',
              ylab='Firing rate\n(z-score)',
              size_scale=2,
              fscale=(5,6,7),
              pad=3):
    '''
    
    '''
    tempdat = np.nansum(actmat,axis=-1)
    if z_score:
        mu = np.nanmean(tempdat[zrange[0]:zrange[1]])
        sigma = np.nanstd(tempdat[zrange[0]:zrange[1]])
        tempdat = (tempdat - mu) / sigma
    ################################################################################################################
    fwd = fwd * size_scale
    fht = fht * size_scale
    
    fig, ax = plt.subplots(1,1,figsize=cm2inch(fwd,fht))
    xpts = np.linspace(-nms[0],nms[1],tempdat.shape[0])
    ax.bar(xpts,tempdat,width=binwidth,color='k',align='edge')
    
    xlim = (-nms[0],nms[1])
    if set_lims:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ###################################################################################################################
    ax.xaxis.set_major_locator(Ticker.MultipleLocator(x_maj_loc))
    ax.yaxis.set_major_locator(Ticker.MultipleLocator(y_maj_loc))

    ax = sbf.set_axis_bounds(ax,xlim,ylim,sp_len=6)
    ax = sbf.adjust_plot_pub(ax,xlab,ylab,nms=nms,binwidth=binwidth,xscale=1,fscale=fscale,grid=False,pad=pad)
    
    return fig,ax
    
def plot_individual_event_rasters(ifr_dict,
                                  ev_indx_list,
                                  bsnm,
                                  nms,
                                  binwidth,
                                  fwd=10,fht=6,size_scale=1,
                                  fscale=(10,12),
                                  offset=25,
                                  x_maj_loc=100,
                                  y_maj_loc=10,
                                  despine_x=False,despine_y=False,
                                  ylab = 'Principal cell #',
                                  line_col = gray2):
    '''

    '''
    #################################################################################################
    # used to draw dashed lines e.g. +/- 25 ms around the peak of each event
    midpoint = int(np.sum(nms) / 2)
    vline_list = [midpoint - offset, midpoint + offset-1]   
    #################################################################################################
    for key,output_array in ifr_dict.items():
        for fig_indx,ev_indx in enumerate(ev_indx_list):
            actmat = output_array[:,:,ev_indx]
            fig,ax = plot_event_raster(actmat,
                                       nms=nms,
                                       binwidth=binwidth,
                                       fwd=fwd,
                                       fht=fht,
                                       x_maj_loc=x_maj_loc,
                                       y_maj_loc=y_maj_loc,
                                       despine_x=despine_x,
                                       despine_y=despine_y,
                                       lw=lw,
                                       size_scale=size_scale,
                                       fscale=fscale,
                                       ylab=ylab
                                      )
            ##########################################################################################
            ylim = ax.get_ylim()
            for vline in vline_list:
                ax.axvline(vline,linewidth=lw,linestyle='--',color=line_col)
            ##########################################################################################
            ftitle = '_'.join((bsnm,key,str(ev_indx+1)))
            ax.set_title(ftitle, y=1.05, x=0.5)
            plt.show()
##################################################################################################
def get_ylim_ctype(ctype,pulse_type):
    '''
    Helper function to get ylimits for specified cell types to visualise
    group responses during sharp-wave ripples (swr) and dentate spikes (ds)
    '''
    if pulse_type.startswith('ds'):
            ylim_dict = {'pdg':(0,16),'pdgL':(0,20),'p3':(0,12),'p1':(0,12),
                         'bdg':(0,60),'b3':(0,60),'b1':(0,60) }        
    elif pulse_type.startswith('swr'):
        ylim_dict = {'pdg':(0,12), 'pdgL':(0,12),'p3':(0,12),'p1':(0,20),
                     'bdg':(0,50),'b3':(0,30),'b1':(0,30) }
    try:
        ymin,ymax = ylim_dict[ctype]
    except KeyError:
        ymin,ymax = -2,10
        print('no ylim entry')

    return ymin, ymax
    
def plot_group_mean_psth(all_summary_data,
                         event_type,
                         ctype,
                         size_scale=2,
                         fscale=(6,7),
                         fwd=4,fht=1.5,
                         err_bars=True,
                         savgol=False,
                         xlab='Time (ms)',
                         ylab='Firing\nrate (Hz)',
                         nms=[200,200],
                         binwidth=1,
                         lw=0.5,
                         pad=1.5,
                         ctype_dict = {'pdg':'DG','p3':'CA3','p1':'CA1'}):
    ''' 
    
    '''
    ######################################################################################   
    figsize = [fwd*size_scale,
               fht*size_scale]
    ######################################################################################
    idata = np.squeeze(all_summary_data[event_type][ctype][0])
    ######################################################################################
    ymin,ymax = get_ylim_ctype(ctype,event_type)
    xscale = 1
    xtick_width = 100 * xscale
    ytick_width = ymax / 2
    midpoint = int(nms[0] / binwidth)
    baseline = None
    xmin,xmax = [-nms[0],nms[1]]
    if ymax > .9: yt_fmt = '.0f'
    else: yt_fmt = '.1f'
    ####################################################################################################################
    ax_lims = [xmin,xmax]
    min_dur = None
    av = 'mean'
    ####################################################################################################################
    fscale = [x*size_scale for x in fscale]
    pad = pad * size_scale
    set_lims = True
    ####################################################################################################################
    fig,ax = sbf.plot_group_pulse(idata,nms=nms,binwidth=binwidth,figsize=figsize,av=av,savgol=False)
    if err_bars:
        ax = sbf.plot_err_bars(ax,idata,nms,axis=1,lw=lw,color=BLUE)
    ####################################################################################################################
    ax.xaxis.set_major_locator(Ticker.MultipleLocator(xtick_width))
    ax.yaxis.set_major_locator(Ticker.MultipleLocator(ytick_width))
    ####################################################################################################################
    # Plot horizontal line at 0
    ax.axhline(0,linewidth=lw,color='k')
    # set / get x and y limits
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    #################################################################################################################################
    ax = sbf.set_axis_bounds(ax,(xmin,xmax),(ymin,ymax),sp_len=10)
    #################################################################################################################################
    ax = sbf.adjust_plot_pub(ax,
                             xlab,
                             ylab,
                             nms=nms,
                             binwidth=binwidth,
                             xscale=xscale,
                             yt_fmt=yt_fmt,
                             fscale=fscale,
                             grid=False,
                             pad=pad)
    ##################################################################################################################
    stitle = ' '.join(('Group mean',
                      event_type.upper(),
                      ctype_dict[ctype],
                      '(n=' + str(idata.shape[-1]) + ')'))
    ax.set_title(stitle, y=1.0, x=0.5, fontsize=fscale[0])

    return fig,ax


def generate_heatmaps(zscore_dat,
                      ctype,
                      event_type_list,
                      sort_event='ds',
                      size_scale=2,
                      fscale=(5,6),
                      colbar=False,
                      cmap='seismic',
                      interpolation='auto',
                      origin='lower',
                      fwd=2,
                      fht=4,
                      xlab='Time (ms)',
                      ylab='Cell #',
                      vmin=-2,
                      vmax_dict={'ds':4, 'swr':4, 'ds1':4, 'ds2':4},
                      y_maj_loc_dict = {'pdg':200,'p3':100,'p1':200},
                      ctype_dict = {'pdg':'DG','p3':'CA3','p1':'CA1'},
                      lw=0.5,
                      pad=1.5,
                      ):
    '''
    
    '''
    ##############################################################
    fwd, fht = fwd * len(event_type_list) * size_scale, fht * size_scale
    ##############################################################
    xmin, xmax = 50,50
    xoffset = 200
    x_maj_loc = xmax
    pad = pad * size_scale
    fscale = [x * size_scale for x in fscale]
    ################################################################################################
    sorted_strength = True
    binstep = 5
    binsize = 5
    ###############################################################################################
    fig,ax = plt.subplots(1,len(event_type_list),figsize=cm2inch(fwd,fht))
    ###############################################################################################
    for indx,event_type in enumerate(event_type_list):
        im_dat = zscore_dat[event_type][ctype]
        
        if sorted_strength:
            binned_data = bin_array(zscore_dat[sort_event][ctype],
                                    axis=0,
                                    binstep=binstep,
                                    binsize=binsize,
                                    func=np.nanmean)
        
            midpoint = int(binned_data.shape[0] / 2)
            ind = np.lexsort((binned_data[midpoint+1, :],
                              binned_data[midpoint, :]))
            
            im_dat = im_dat[:,ind]
        
        stitle = ctype_dict[ctype] + ' ' + event_type.upper()

        vmax = vmax_dict[event_type]
        
        im = ax[indx].imshow(im_dat.T,
                             aspect='auto',
                             cmap=cmap,
                             vmin=vmin,
                             vmax=vmax,
                             interpolation=interpolation,
                             interpolation_stage='data',
                             origin=origin)
        
        ylim = (0,im_dat.shape[1])
        ax[indx].set_ylim(ylim)
        ax[indx].set_title(stitle,fontsize=fscale[1])
        
        ax[indx].set_xlim(-xmin+xoffset,xmax+xoffset)
        ax[indx].xaxis.set_major_locator(Ticker.MultipleLocator(x_maj_loc))
        ax[indx].yaxis.set_major_locator(Ticker.MultipleLocator(y_maj_loc_dict[ctype]))
        
        ax[indx].xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x - xoffset:.0f}"))

        ax[indx].set_xlabel(xlab,fontsize=fscale[1])
        ax[indx].set_ylabel(ylab,fontsize=fscale[1])
        ax[indx].tick_params(width=lw,length=3*lw,pad=pad,labelsize=fscale[0])
    ###########################################################################################    
    for indx in np.arange(1,len(event_type_list)):
        ax[indx].yaxis.set_visible(False) # Hide y axis
    ###########################################################################################
    for ax_ in ax:
        for sp_pos in ['top','bottom','left','right']:
            ax_.spines[sp_pos].set_linewidth(lw)
    ###########################################################################################
    if colbar:
        tempax = ax[1]
        divider = make_axes_locatable(tempax)
        cax = divider.append_axes("right", size="10%", pad="10%")
        cb1 = fig.colorbar(im,cax=cax,orientation="vertical")
        cb1.ax.tick_params(width=lw,length=3*lw,labelsize=fscale[0])

    return fig,ax
    
####################################################################################################
def cat_plot_clf(df,
                 hue_col='sleep_event',
                 palette=(gray2,LIGHTPURPLE,ORNG,PURPLE),
                 legend=False,
                 msize=2,
                 ylim=(0,1),
                 ytick_width=0.25,
                 figsize=(4,4),
                 ylab = 'Classifier accuracy',
                 pad=1.5,
                 tfsize=7,
                 xlab='',
                 fscale=(10,12)):
    '''
    
    '''
    ################################################################################
    mean_lw = 3
    lw = 0.5
    ls = '-'
    l_offset = .4
    alpha = 1.0
    #################################################################################
    fig,ax = plt.subplots(1,1,figsize=sbf.cm2inch(figsize))

    ## Create scatterplot
    ax = sns.swarmplot(data=df,
                       x="sleep_event",
                       y="Data",
                       alpha=alpha,
                       size=msize,
                       hue=hue_col,
                       palette=palette
                      )
    ## Plot means
    df_mean = df.groupby('sleep_event', sort=False)['Data'].mean()
    _ = [ax.hlines(y, i-l_offset, i+l_offset, zorder=10, color='k',linewidth=mean_lw, 
                   linestyle=ls,alpha=0.3) for i, y in df_mean.reset_index()['Data'].items()]

    # Set y-axis limit and tick width
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(Ticker.MultipleLocator(ytick_width))

    # Set x and y ticklabels fontsize
    ax.tick_params(axis='both', 
                   which='major',
                   labelsize=tfsize)

    # set axis bounds
    xlim = (0,len(np.unique(df.sleep_event))-1)
    ax = sbf.set_axis_bounds(ax,xlim,ylim,sp_len=3)

    # adjust plot
    ax = sbf.adjust_plot_pub(ax,
                            xlab=xlab,
                            ylab=ylab,
                            lw=lw,
                            raster=False,
                            xtwidth=20,
                            xscale=1,
                            yscale=1,
                            xt_fmt='.0f',
                            yt_fmt='.2f',
                            fscale=fscale,
                            grid=False,
                            pad=pad)
    
    return fig,ax
################################################################################################
def plot_single_similarity_matrix(sim_mat,
                                  event_1,
                                  event_2,
                                  cbar=False,
                                  fwd=10,
                                  fht=10,
                                  size_scale=2,
                                  vlim=(-0.3,0.3),
                                  cmap='jet',
                                  nprs=10,
                                  fscale=(5,6,8),
                                  pad=3):
    '''
    '''
    arr = sim_mat
    ################################################
    row_sums = np.sum(arr, axis=1)
    col_sums = np.sum(arr, axis=0)
    
    # Identify rows and columns with zero sums
    rows_to_keep = row_sums != 0
    cols_to_keep = col_sums != 0
    
    # Delete zero-sum rows and columns
    G = arr[rows_to_keep][:, cols_to_keep]
    np.fill_diagonal(G, 0)
    ###########################################################################
    fig,ax = plt.subplots(1,1,figsize=sbf.cm2inch(fwd,fht))
    
    xlab=event_2.upper()
    ylab=event_1.upper()
    fscale = [x*size_scale for x in fscale]
    
    ax = sns.heatmap(G, cmap=cmap, center=0, vmin=vlim[0], vmax=vlim[1], cbar=cbar)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=fscale[1])
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=fscale[1], rotation='horizontal')
    
    ax.set_xticks([0,G.shape[0]], ['0', str(G.shape[0])])
    ax.set_yticks([0,G.shape[0]], ['0', str(G.shape[0])])

    ax = sbf.adjust_plot_pub(ax,
                             xlab=xlab,
                             ylab=ylab,
                             raster=False,
                             fscale=fscale,
                             grid=False,
                             pad=pad)
    
    sns.despine()
    if cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=fscale[1])
        cbar.ax.set_ylabel('Correlation (r)',fontsize=fscale[1])
        cbar.ax.yaxis.set_major_locator(Ticker.MultipleLocator(vlim[1]))

    fig = ax.get_figure()

    return fig,ax
#############################################################################################

#############################################################################################
