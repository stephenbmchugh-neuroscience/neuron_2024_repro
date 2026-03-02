######## Dabest helper functions #################################

import numpy as np
import pandas as pd
import scipy.stats as stats
import dabest as db2
import smBaseFunctions3 as sbf

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from my_mpl_defaults import *
from plotting import cm2inch


def generate_dabest_analysis(idata,
                             outerkey,
                             innerkey,
                             col_head,
                             col_groups,
                             df_idx,
                             paired='baseline',
                             x='Cell_Event',
                             y='Data',
                             id_col='id'):
    
    idata = sbf.nan_free_dict(idata,outerkey,innerkey)
    df = dabest_long_df_2var(idata,innerkey,outerkey,col_groups,col_head)    
    ####################################################################################################
    analysis_of_long_df = db2.load(df,
                                  x=x, 
                                  y=y,
                                  idx=((df_idx)),
                                  paired=paired,
                                  id_col=id_col)

    return analysis_of_long_df
########################################################################################################
def adjust_dabest_labels(swarm_ax,contrast_ax,df_idx,max_char=8,str_indx=1,upper_case=True):
    '''

    '''
    sw_xtlabs,con_xtlabs = generate_dabest_xtlabs(df_idx,max_char=max_char,str_indx=str_indx,upper_case=upper_case)
    swarm_ax.set_xticklabels(sw_xtlabs)
    contrast_ax.set_xticklabels(con_xtlabs)

    return swarm_ax,contrast_ax
########################################################################################################
def plot_dabest_swarm_contrast(analysis_of_long_df,
                               df_idx,
                               show_pairs=False,
                               diff_type = 'mean',
                               swlab = 'Firing rate (Hz)',
                               size_scale = 2,
                               fwd=1.25,
                               fht=4,
                               pad=1.5,
                               swarm_ylim=(-5,60),
                               contrast_ylim=(-5,20),
                               swarm_maj_loc=20,
                               contrast_maj_loc=10,
                               fscale=(5,6,7),
                               raw_marker_size=0.05,
                               jitter=3,
                               my_color_palette=None):
    import warnings
    if diff_type == 'mean':
        eslab = 'Mean ' + r'$\Delta$'
    elif diff_type == 'median':
        eslab = 'Median ' + r'$\Delta$'
    #######################################################################################################
    if my_color_palette == None:
        my_color_palette = [gray2, GREEN, PURPLE, BLUE, ORNG,
                            gray2, GREEN, PURPLE, BLUE, ORNG,
                            gray2, GREEN, PURPLE, BLUE, ORNG
                           ]
    #####################################################################################################
    fwd,fht = fwd*len(sbf.flatten(df_idx))*size_scale, fht * size_scale
    pad = pad * size_scale
    font_scale = [x*size_scale for x in fscale]
    #####################################################################################################
    bw_method = .5 #sbf.calc_mean_bw_1var(idata,innerkey)
    float_contrast = False
    ########################################################################################################
    warnings.filterwarnings("ignore",category=UserWarning,module="dabest")
    fig,swarm_ax,contrast_ax = smdabest_plot(analysis_of_long_df,
                                                float_contrast=float_contrast,
                                                diff_type=diff_type,
                                                figsize=[fwd,fht],
                                                size_scale=1,
                                                raw_marker_size=raw_marker_size,
                                                jitter=jitter,
                                                swarm_ylim=swarm_ylim,
                                                contrast_ylim=contrast_ylim,
                                                my_color_palette = my_color_palette,
                                                swlab=swlab,
                                                eslab=eslab,
                                                fs=font_scale,
                                                show_pairs=show_pairs,
                                                maj_loc=[swarm_maj_loc,contrast_maj_loc],
                                                bw_method = bw_method)
    ########################################################################################################
    return fig,swarm_ax,contrast_ax
############################################################################################################
def smdabest_plot(analysis_of_long_df,
                  float_contrast=False,
			diff_type='median',
			figsize=[8,8],
			size_scale=2,
			raw_marker_size=1,
			jitter=0.5,
			swarm_ylim=None,
                        contrast_ylim=None,
			my_color_palette=None,
			swlab='Decoding accuracy',
			eslab='Median ' + r'$\Delta$',
			fs=[9,12,16],
			show_pairs=False,
			maj_loc=None,
			bw_method=0.5,
			pad = 1.5,
                        color_col=None):
    '''
    
    '''
    fwd,fht = figsize[0],figsize[1]

    lw = 0.5 * size_scale
    if maj_loc is None:
        swarm_maj_loc = .2
        contrast_maj_loc = swarm_maj_loc / 1
    else:
        swarm_maj_loc = maj_loc[0]
        contrast_maj_loc = maj_loc[1]

    swsize = .5 * size_scale
    esmsize = 2 * size_scale
    rlcol = gray2
    rlkwargs = {'linewidth': lw,'color': rlcol,'linestyle': '--'}
    vp_kwargs = {'bw_method':bw_method}
    swarm_desat = 1.0
    halfviolin_desat = 1.0
    

    SMALL_SIZE = fs[0]
    MEDIUM_SIZE = fs[1]
    BIGGER_SIZE = fs[2]
    if my_color_palette is None:
        my_color_palette = [RED,ORNG,gray,TURQ,
                            BLUE,PURPLE,GREEN,PINK]
    if diff_type == 'mean':
        diff_type_plot = analysis_of_long_df.mean_diff.plot
    elif diff_type == 'median':
        diff_type_plot = analysis_of_long_df.median_diff.plot
    ####################################################################################################
    fig = diff_type_plot(
		       custom_palette=my_color_palette,
	               float_contrast=float_contrast,
	               #jitter=jitter,
	               #swarmplot_kwargs={'size': swsize, 'jitter':jitter},
	               #swarmplot_kwargs={'jitter':jitter},
                       #violinplot_kwargs=vp_kwargs,	
	               contrast_marker_size=esmsize,
	               group_summaries_kwargs={'lw': lw},
	               reflines_kwargs=rlkwargs,
		       raw_ylim=swarm_ylim,
                       contrast_ylim=contrast_ylim,
	               raw_desat=swarm_desat,
	               contrast_desat=halfviolin_desat,
	               fig_size=cm2inch(fwd,fht),
	               show_pairs=show_pairs,
                       color_col=color_col,
                       raw_marker_size=raw_marker_size
                       )

    swarm_ax = fig.axes[0]
    contrast_ax = fig.axes[1]

    swarm_ax.set_ylabel(swlab,fontsize=MEDIUM_SIZE)
    contrast_ax.set_ylabel(eslab,fontsize=MEDIUM_SIZE)

    swarm_ax.yaxis.set_major_locator(Ticker.MultipleLocator(swarm_maj_loc))
    contrast_ax.yaxis.set_major_locator(Ticker.MultipleLocator(contrast_maj_loc))
    
    for ax_ in [swarm_ax,contrast_ax]:
        ax_.xaxis.set_tick_params(labelsize=SMALL_SIZE, width=lw,length=lw*3)
        ax_.yaxis.set_tick_params(labelsize=SMALL_SIZE, width=lw,length=lw*3)
        for aa in ['x','y']:
            ax_.tick_params(axis=aa,pad=pad)

    return fig, swarm_ax, contrast_ax
###########################################################################################################
def dabest_long_df(idata,cellkey,sesskey,col_groups,col_head):
    '''

    '''
    data_r = np.array([])
    clab_r = np.array([])
    slab_r = np.array([])
    id_r = np.array([],dtype=int)
    group_r = np.array([])
    for cind, ckey in enumerate(cellkey):
        for sind, skey in enumerate(sesskey):
            data_r = np.r_[data_r,idata[ckey][skey]]
            clab_r = np.r_[clab_r,np.repeat(col_groups[ckey],len(idata[ckey][skey]))]
            slab_r = np.r_[slab_r,np.repeat(skey,len(idata[ckey][skey]))]
            id_r = np.r_[id_r, np.arange(0,len(idata[ckey][skey]))]
            glab = col_groups[ckey] + '_' + skey
            print(glab)
            group_r = np.r_[group_r,np.repeat(glab,len(idata[ckey][skey]))]
            
    print(len(data_r))        
    print(len(group_r))

    df = pd.DataFrame({col_head[0]: id_r,
                       col_head[1]: clab_r,
                       col_head[2]: slab_r,
                       col_head[3]: data_r,
                       col_head[4]: group_r
                       })
    return df
######################################################################
def dabest_long_df_1var(idata,key,col_groups,col_head):
    '''

    '''
    data_r = np.array([])
    ilab_r = np.array([])
    id_r = np.array([],dtype=int)
    group_r = np.array([])
    
    for iind, ikey in enumerate(key):
        data_r = np.r_[data_r,idata[ikey]]
        print(col_groups[ikey])
        ilab_r = np.r_[ilab_r,np.repeat(col_groups[ikey],len(idata[ikey]))]
        id_r = np.r_[id_r, np.arange(0,len(idata[ikey]))]
        glab = col_groups[ikey]
        print(glab)
        group_r = np.r_[group_r,np.repeat(glab,len(idata[ikey]))]
            
    print(len(data_r))        
    print(len(group_r))

    df = pd.DataFrame({col_head[0]: id_r,
                       col_head[1]: ilab_r,
                       col_head[2]: data_r,
                       col_head[3]: group_r
                       })
    return df
######################################################################
def dabest_long_df_2var(idata,innerkey,outerkey,col_groups,col_head):
    '''

    '''
    data_r = np.array([])
    ilab_r = np.array([])
    olab_r = np.array([])
    id_r = np.array([],dtype=int)
    group_r = np.array([])
    
    for iind, ikey in enumerate(innerkey):
        for oind, okey in enumerate(outerkey):
            data_r = np.r_[data_r,idata[ikey][okey]]
            ilab_r = np.r_[ilab_r,np.repeat(col_groups[ikey],len(idata[ikey][okey]))]
            olab_r = np.r_[olab_r,np.repeat(okey,len(idata[ikey][okey]))]
            id_r = np.r_[id_r, np.arange(0,len(idata[ikey][okey]))]
            glab = col_groups[ikey] + '_' + okey
            print(glab)
            group_r = np.r_[group_r,np.repeat(glab,len(idata[ikey][okey]))]
            
    print(len(data_r))        
    print(len(group_r))

    df = pd.DataFrame({col_head[0]: id_r,
                       col_head[1]: ilab_r,
                       col_head[2]: olab_r,
                       col_head[3]: data_r,
                       col_head[4]: group_r
                       })
    return df
    
###########################################################################################################
def generate_dabest_xtlabs(df_idx,max_char=4,str_indx=0,upper_case=True):
    '''
    
    '''
    sw_xtlabs = []
    con_xtlabs = []
    
    for each_ in df_idx:
        n_groups = len(each_)
        if upper_case:
            each_list = [x.split('_',3)[str_indx][:max_char].upper() for x in each_]
        else:
            each_list = [x.split('_',3)[str_indx][:max_char] for x in each_]

        sw_xtlabs.append(each_list)
        ####################################################################
        ostr = ['']
        for group in range(1,n_groups):
            ostr.append("\nminus\n".join([each_list[group],
                                          each_list[0]])
                       )
        con_xtlabs.append(ostr)
        ####################################################################  
    return sbf.flatten(sw_xtlabs),sbf.flatten(con_xtlabs)
############################################################################################################
def update_xticklabels(sw_xtlabs,con_xtlabs,swarm_ax,contrast_ax):

    for ax_,tl in zip([swarm_ax,contrast_ax],[sw_xtlabs,con_xtlabs]):
        ax_.set_xticklabels(tl)
#############################################################################################################
