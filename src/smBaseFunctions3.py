##############################################################
## Import libraries
##############################################################
import pandas as pd
import numpy as np
import sys
import math
import os
import time
import datetime
import collections
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import scipy as scipy
import scipy.stats as stats
import scipy.ndimage as spim
from scipy.signal import savgol_filter
import fnmatch
#########################################################################
import vBaseFunctions3 as vbf
from my_mpl_defaults import *
#########################################################################
class StopExecution(Exception):
    def _render_traceback_(self):
        pass
##########################################################################
def get_fontdict():
    '''
    '''
    # Then, "ALWAYS use sans-serif fonts"
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['font.sans-serif'] = "Arial"
    fontdict = {'family': 'sans-serif',
                'color':  'k',
                'weight': 'normal',
                'size': 10,
                }
    return fontdict
############################################################################################
# Get database entries
############################################################################################
def get_database_dl(day_type,group_type,old_mpath='/lfpd4/SF/',mpath = '/Dupret_Lab/',update=True,pprint=True):
    '''
    updated function to return database when data are mounted on Dupret_Lab
    '''
    
    new_mpath = '/Dupret_Lab/merged/smchugh_merged/'
    ipath = '/analysis/smchugh_analysis/databases'

    database =  get_database(day_type,
                            group_type,
                            mpath=mpath,
                            old_mpath=old_mpath,
                            new_mpath=new_mpath,
                            ipath=ipath,
                            update=update,
                            pprint=pprint)

    return database
########################################################################################################################################################
def get_database(day_type,group_type,mpath='/mnt/smchugh2/',old_mpath='/mnt/smchugh/',new_mpath='',ipath='/lfpd4/databases',update=True,pprint=True):
    '''

    '''   
    print('database rootdir is {}'.format(mpath+ipath[1:]))
    ################################################################
    database = read_db(day_type + '_' + group_type + '.db',
                       ipath,
                       mpath=mpath,
                       omit=True)
    if update:
        database = update_mpath(database,
                                old_mpath=old_mpath,
                                new_mpath=new_mpath)
    if pprint:
        print()
        for indx,val in enumerate(database):
            print(indx,val)

    return database
######################################################################################
def delete_database_entry(database,mindx):
    del database[mindx]
######################################################################################
def adjust_desen(sessions):
    '''
    
    '''
    odata = []
    for s in sessions:
        if 'stim' in s:
            odata.append(s[:2])
        else:
            odata.append(s)
    return odata
##########################################################################
def read_db(ifile,ipath,mpath='/mnt/smchugh',omit=False,printpath=True):
    # read in a list of path names
    fname= str(ifile)
    other_part = ipath
    fullpath = mpath + ipath + '/' + fname
    if printpath:
        print(fullpath)
    odb = []
    if omit:
        mpath = ''
    with open(fullpath) as filen:
        for line in filen: 
            line = line.strip() 
            odb.append(mpath+line)
    filen.close()        
    return odb
################################################################
def get_all_ages(database,extt='.dob'):
    '''
    
    '''    
    ages_array = age_array = np.zeros( (len(database) ) )
    for dindx,eachdir in enumerate(database):
        ipath,bsnm,baseblock,par,desen = get_mouse_info_single(eachdir)
        ages_array[dindx] = get_age(bsnm,ipath,extt=extt)
        
    return ages_array
#################################################################################
def get_age(bsnm,ipath,extt='.dob'):
    '''
    n_days = get_age(bsnm,ipath)
    '''
    rec_dat = (datetime.datetime.strptime(bsnm[-6:], '%y%m%d')).date()
    # Iterate directory
    file_found = False
    for ifile in os.listdir(ipath):
        # check only dob file
        if ifile.endswith(extt):
            file_found=True
            dob = str(np.loadtxt(ifile,dtype='int'))
            dob_obj = (datetime.datetime.strptime(dob, '%y%m%d')).date()
            diff = rec_dat-dob_obj
            n_days = int(diff.days)
            print(bsnm,'is',str(n_days),'days old')
    if file_found == False:
        print(bsnm, 'no dob file')
        n_days = np.nan

    return n_days
##################################################################################
def update_mpath(database,old_mpath='/mnt/smchugh/',new_mpath='/mnt/smchugh2/'):
    '''

    '''
    string_to_replace = old_mpath
    odata = []
    for dindx,eachdir in enumerate(database):
        odata.append(new_mpath+eachdir.replace(string_to_replace,''))
    return odata
###############################################################
def get_files(path,extt,rev=False,npy=True,pprint=True):
    '''

    '''
    iarray = []
    fileID = []
    for root, dirs, files in os.walk(path):
        for filen in sorted(files,reverse=rev):
            if filen.endswith(extt):
                if pprint:
                    print("{0:} files are: {1:}".format(extt,filen))
                fileID.append(filen)
                fullpath = path + '/' + filen
                if npy:
                    iarray.append(np.load(fullpath,allow_pickle=True))
                else:
                    iarray.append(np.loadtxt(fullpath))
    return iarray, fileID 
#######################################################################
def get_files_strmatch(path,extt,rev=False,npy=True):
    '''
    Imports data from files (.npy or text)
    Based on matching or partial matching of filenames
    '''
    iarray = []
    fileID = []
    for root, dirs, files in os.walk(path):
        for filen in sorted(files,reverse=rev):
            #if extt in filen:
            if fnmatch.fnmatch(filen, extt):
                print("{} files are: {}".format(extt,filen))
                fileID.append(filen)
                fullpath = path + '/' + filen
                if npy:
                    iarray.append(np.load(fullpath,allow_pickle=True))
                else:
                    iarray.append(np.loadtxt(fullpath))
    return iarray, fileID
###################################################################################################################################################
def generate_ipath(group_type,
                   day_type,
                   ext,
                   mpath = 'mnt/smchugh2',
                   first_part = 'lfpd4/SF_analysis',
                   subfolder = 'DentateSpike',
                   pulse_type = 'DSResponses',
                   dat_type = 'PSTH',
                   duration_type = '400ms'):
    '''
    '''
    ext_type = ext[1:]
    other_part = os.path.join(first_part,subfolder,pulse_type,group_type,day_type,ext_type,dat_type,duration_type)
    ipath = os.path.join(mpath,other_part)
    
    return ipath
###################################################################################################################################################
def get_FRList(ipath,ext_pt1,ext_pt2):
    '''
    
    '''
    FRList = {}
    fid = []
    for findx,ftype in enumerate(ext_pt1):
        extt1 = '_' + ftype + ext_pt2
        print()
        print(extt1[1:])
        FRList[ftype],fid = get_files(ipath,extt1,rev=False,npy=True)
    
    return FRList
###################################################################################################################################################
def get_actMat(day_type,
               group_type,
               dtype='actMat_ctype',
               cellgroup='allp',
               mpath='/Dupret_Lab/analysis/',
               other_path='smchugh_analysis/SF_analysis/IFR',
               interval='theta'):
    '''
    Returns an activity matrix at the given path
    needs editing for general use
    ## changed May 2023 - check if it works!
    '''
    ipath = os.path.join(mpath,
                         other_path,
                         day_type,
                         group_type,
                         interval,
                         dtype)
    fname = '_'.join((dtype,group_type,cellgroup)) + '.npy'
    fullpath = ipath + '/' + fname
    
    return np.load(fullpath,allow_pickle=True)
#######################################################################################################################
def combine_lcond(actMat,mouseID,ikey_list=['nolight','light'],okey='all'):
    '''
    
    '''
    odata = []
    for mindx,mouse in enumerate(mouseID):
        tempDict = rec_dd()
        for key,val in actMat[mindx].items():
            print(key,actMat[mindx][key][ikey_list[0]].shape,actMat[mindx][key][ikey_list[1]].shape)
            combined_array = np.concatenate((actMat[mindx][key][ikey_list[0]],
                                             actMat[mindx][key][ikey_list[1]]),
                                            axis=1)
            tempDict[key][okey] = combined_array
        oDict = defdict_to_dict(tempDict, {})
        odata.append(oDict)

    return odata
#######################################################################################################################
def rec_dd():
    return defaultdict(rec_dd)
#######################################################################################################################
def defdict_to_dict(defdict, finaldict):
    '''
    e.g. call as odict = sbf.defdict_to_dict(tempdict,{})
    '''
    # pass in an empty dict for finaldict
    for k, v in defdict.items():
        if isinstance(v, defaultdict):
            # new level created and that is the new value
            finaldict[k] = defdict_to_dict(v, {})
        else:
            finaldict[k] = v
            
    return finaldict
#######################################################################################################################
def flatten(lst):
    '''Flattens a list of lists'''
    return [subelem for elem in lst for subelem in elem]
#######################################################################################################################
def create_pixels_to_cm_file(ipath,maze_cm=41.0,f_ext = '.txt',fmt = '%1.3f',pprint=True):
    '''

    '''
    os.chdir(ipath)
    os.getcwd()
    # get basename from database
    bsnm = ipath.rsplit('/', 1)[-1]
    baseblock = ipath + '/' + bsnm
    ########################################################################################
    mazedim = get_mazedim(baseblock)
    print(bsnm,mazedim)
    pixels2cm = (mazedim[1] - mazedim[0]) / maze_cm
    cm2pixels = maze_cm / (mazedim[1] - mazedim[0])
    fname = bsnm + '_pixels_to_cm'
    odata = [pixels2cm]
    if pprint:
        print('there are {:.2f} pixels for each cm'.format(pixels2cm))
        print('each pixel is {:.2f} cm'.format(cm2pixels)) 
    write_to_text(ipath,fname,odata,f_ext,fmt='%d')
##########################################################################
def write_to_text(opath,fname,odata,intv_ext,fmt='%d'):
    '''
    Simple save function, e.g. for pulse files
    '''
    print('saving to: {}'.format(opath))
    os.chdir(opath)
    ofname = fname + intv_ext
    print('saving file {}'.format(ofname))
    fHand = open(ofname,'wb')
    np.savetxt(fHand,odata,fmt=fmt)
    fHand.close()
########################################################################
def save_npy_data(mpath,output_path,fname,odata):
    '''
    
    '''    
    fullpath = mpath + output_path
    print(fullpath)
    os.chdir(fullpath)
    now = datetime.datetime.now()
    #########################################################################################################
    print('saving file {}'.format(fname))
    print ('Time saved: ' + now.strftime("%Y-%m-%d %H:%M:%S"))
    #########################################################################################################
    fHand = open(fname,'wb')
    np.save(fHand,odata)
    fHand.close()
###############################################################################
def savefig(fig,opath,ftitle,ext='.svg',dl=True):
    '''
    '''
    if dl:
        old_path = '/mnt/smchugh2/lfpd4/'
        new_path = '/Dupret_Lab/analysis/smchugh_analysis/'
        opath = opath.replace(old_path,new_path)

    os.chdir(opath)
    ofname = ftitle + ext
    print('Saving {0:} to {1:}'.format(ofname,opath))
    fig.savefig(ofname)
    now = datetime.datetime.now()
    print ('Time saved: ' + now.strftime("%Y-%m-%d %H:%M:%S"))
###############################################################################
def save_IFR_data(mpath,output_path,fname,odata):
    '''

    '''
    fullpath = mpath + output_path
    print(fullpath)
    os.chdir(fullpath)
    now = datetime.datetime.now()
    ##################################################################################
    print('saving file {}'.format(fname))
    print ('Time saved: ' + now.strftime("%Y-%m-%d %H:%M:%S"))
    ##################################################################################
    fHand = open(fname,'wb')
    np.save(fHand,odata)
    fHand.close()
######################################################################################
def update_desen(df,nospaces=True):
    '''
    '''
    if df['filebase'].iloc[0].startswith('sm'):
        df['filebase'] = 'm' + df['filebase']
    if nospaces:
        df['desen'] = df['desen'].str.replace(" ", "")
    return df
#########################################################################################    
def load_units(b, par=None, each_trode_ext='.des.', all_trode_ext='.desf'):
    '''Load "units" information (mostly from des-file).
    INPUT:
    - [b]:       <str> containing "block base"
    OUTPUT:
    - [trodes]:  <DataFrame>'''

    ## If not provided, load the par-file information
    if par is None:
        par = vbf.LoadPar(b)

    ## For each tetrode, read in its "per tetode" des-file
    trode_index = range(1, len(par['trode_ch'])+1)
    units = [pd.read_csv(b+each_trode_ext+str(t), header=None, names=['des']) for t in trode_index]
    units = pd.concat(units, keys=trode_index, names=['trode','trode_unit']).reset_index()
    # -as a check, also read in the "overall" des-file
    all_trodes = pd.read_csv(b+all_trode_ext, header=None, names=['des'])
    if ~np.all(all_trodes.des == units.des):
        units.des = all_trodes.des # note this defaults to all_trodes i.e. bsnm.des file

    ## Let the "index per tetrode" and the index of this <DataFrame> start from 2(!) instead of 0
    units['trode_unit'] += 2
    units.index += 2

    ## Add "unit" as column, and set name of column-index to "unit" (NOTE: not sure why this is needed!?)
    #units['unit'] = units.index
    #units.index.set_names('unit', inplace=True)
    ## Return the "unit"-information as <DataFrame>
    return units
##############################################################
def get_all_mouse_db_info(database,SF=True,each_trode_ext='.des.',all_trode_ext='.desf'):
    '''
    
    '''
    ################################################################
    units = []
    mouseID = []
    alldesen = []
    allBaseblock = []
    allPar = []
    ################################################################
    for dindx, eachdir in enumerate(database):
        os.chdir(eachdir)
        os.getcwd()
        # get basename from database
        ipath = eachdir
        bsnm = ipath.rsplit('/', 1)[-1]
        baseblock = ipath + '/' + bsnm
        allBaseblock.append(baseblock)
        # load units as df into list called units
        par = vbf.LoadPar(baseblock)
        allPar.append(par)
        try:
            units.append(load_units(baseblock, par=par, each_trode_ext=each_trode_ext, all_trode_ext=all_trode_ext))
        except FileNotFoundError:
            units.append(None)
        alldesen.append(vbf.LoadStages(baseblock))
        mouseID.append(bsnm)
    ################################################################
    if SF:
        for desDF in range(0,len(alldesen)):
            alldesen[desDF] = update_desen(alldesen[desDF])
    ################################################################
    return mouseID,allBaseblock,allPar,alldesen,units
######################################################################################
def count_ctype_database(mouseID,units,targ_ctype='p1',output_df=True,pprint=False):
    '''
    '''
    odata = np.zeros((len(mouseID),3), dtype=object)
    for mindx,bsnm in enumerate(mouseID):
        mask = units[mindx]['des'] == targ_ctype
        odata[mindx,:] = [bsnm,targ_ctype,len(units[mindx][mask])]
        if len(units[mindx][mask]) and (pprint):
            print(bsnm,targ_ctype,len(units[mindx][mask]))
    if output_df:
        odf = pd.DataFrame(data=odata, columns=['Bsnm','Ctype','Count'])
        return odf
######################################################################################
def count_cells(df,mouse,ctype,exact=True,prt=False): 
    '''
    Pass a dataframe of units,mouse id (int), and ctype ('p1')
    Function will return the cell_indices
    ...will also print to display if prt=True	
    '''
    if exact:
        cell_inds = [index for index, value in enumerate(df[mouse]['des'].values) \
                          if ctype == value]
    else:
        cell_inds = [index for index, value in enumerate(df[mouse]['des'].values) \
                          if ctype in value]
    if prt:
        print('mouse {0:} {1:} {2:}'.format(mouse,ctype,len(cell_inds)))
    return cell_inds
########################################################################################
def get_all_cell_inds(database,units,celltype=None,pprint=True,thresh1=100,exact=True):
    '''
    
    '''
    indIDs = {}
    origIDs = {}

    for cindx,ctype in enumerate(celltype):
        indIDs[ctype], origIDs[ctype] = append_cell_inds(database,
                                                         units,
                                                         ctype,
                                                         thresh1=thresh1,
                                                         exact=exact)
    return indIDs,origIDs
########################################################################################
def append_cell_inds(database,units,ctype,thresh1=100,exact=True):
    '''
    
    '''
    indxCell_ids = []
    origCell_ids = []
    tsum = 0
    for i,val in enumerate(database):
        tempcells = count_cells(units,i,ctype,exact=exact)
        tsum += len(tempcells)
        if len(tempcells) >= thresh1:
            print(ctype,database[i],len(tempcells))
        indxCell_ids.append(tempcells)    
        origCell_ids.append([x+2 for x in tempcells])
    print(ctype,tsum)
    
    return indxCell_ids,origCell_ids
    
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

######################################################################################
def get_cell_inds_one_mouse(units,ctype_list=['pdg','p3','p1'],exact=True):
    '''
    '''

    origIDs = {}
    for cindx,ctype in enumerate(ctype_list):
        origIDs[ctype] = get_cell_inds(ctype,units,exact=exact)

    return origIDs
######################################################################################
def multi_ctype_IDs(clist,origIDs,mouse=None):
    '''
    
    '''
    odata = []
    for cindx,ctype in enumerate(clist):
        if mouse is not None:
            odata += origIDs[ctype][mouse]
        else:
            odata += origIDs[ctype]
        
    return odata
############################################################################################
def count_cells_multi(mouseID,origIDs,clist=['pdg','p3','p1'],output_df=True):
    '''
    
    '''
    odata = np.zeros((len(mouseID),2),dtype=object)
    for mouse,bsnm in enumerate(mouseID):
        cluIDs = multi_ctype_IDs(clist,origIDs,mouse)
        odata[mouse,:] = [bsnm,len(cluIDs)]
    if output_df:
        odf = pd.DataFrame(data=odata,
                           columns=['Bsnm','Count']
                          )
        return odf
##############################################################################################
def percent_cells(origIDs,celltypes):
    '''

    '''
    flat_list = []
    for cindx,ctype in enumerate(celltypes):
        flat_list.append(len([item for sublist in origIDs[ctype] for item in sublist]))

    return np.round(100 * (flat_list[0] / (flat_list[0] + flat_list[1])),2)
##############################################################################################
def get_mouse_info(alldesen,allBaseblock,mouse):
    '''

    '''
    desen = alldesen[mouse]
    baseblock = allBaseblock[mouse]
    ipath = baseblock.rsplit('/',1)[0]
    bsnm = ipath.rsplit('/',1)[-1]

    return desen,baseblock,ipath,bsnm
##############################################################################################
def get_mouse_info_single(eachdir):
    '''

    '''
    os.chdir(eachdir)
    os.getcwd()
    # get basename from database
    ipath = eachdir
    bsnm = ipath.rsplit('/', 1)[-1]
    baseblock = ipath + '/' + bsnm
    print('baseblock is: {}'.format(baseblock))
    print('basename is: {}'.format(bsnm))
    par = vbf.LoadPar(baseblock)
    desen = vbf.LoadStages(baseblock)
    desen = update_desen(desen)

    return ipath,bsnm,baseblock,par,desen
###############################################################################################
def mouse_info_data_share(mpath,ipath):
    '''
    '''
    mouseID = list(np.load(mpath + ipath + 'mouseID',allow_pickle=True))
    allBaseblock = list(np.load(mpath + ipath + 'allBaseblock',allow_pickle=True))
    allPar = list(np.load(mpath + ipath + 'allPar',allow_pickle=True))
    alldesen = list(np.load(mpath + ipath + 'alldesen',allow_pickle=True))
    units = list(np.load(mpath + ipath + 'units',allow_pickle=True))
    
    return mouseID,allBaseblock,allPar,alldesen,units
################################################################################################
def get_IQR(idata,pc=[25,50,75]):
    '''
    Takes a vector and calculates two percentiles (e.g. 25th, 75th)
    Returns npc values in odata, e.g. Q1: [0], Q3: [1] and Q3-Q1: [2] 
    '''
    odata = []
    for qq in pc:
        odata.append(np.nanpercentile(idata, qq, interpolation = 'midpoint'))

    odata.append(odata[1]-odata[0])

    return odata
#################################################################################################
def get_tconv(spk_sr=20000,lfp_sr=1250,trk_sr=39.0625):
    '''
    
    '''
    tconv = {}
    tconv['lfp_spk'] = (spk_sr / lfp_sr)
    tconv['lfp_trk'] = (trk_sr / lfp_sr)
    tconv['lfp_lfp'] = (lfp_sr / lfp_sr)
    tconv['trk_spk'] = (spk_sr / trk_sr)
    tconv['trk_lfp'] = (lfp_sr / trk_sr)
    tconv['spk_lfp'] = (lfp_sr / spk_sr)
    tconv['spk_trk'] = (trk_sr / spk_sr)
    tconv['spk_spk'] = (spk_sr / spk_sr)
    tconv['spk_ms'] = (1000 / spk_sr)
    tconv['ms_spk'] =  (spk_sr * (1/1000) )
    tconv['spk_sr'] = spk_sr
    tconv['lfp_sr'] = lfp_sr
    tconv['trk_sr'] = trk_sr
    
    return tconv
#################################################################################################
def create_dataframe_from_dict(iDict,ikey_list=['pdg','pdgL'],okey= 'allsess',logT=True):
    '''
    contstruct pandas data frame with two cols: ctype and Firing rate
    '''
    allDat = np.array([])
    allNames = np.array([])
    for ikey_indx,ikey in enumerate(ikey_list):
        if logT:
            tempDat = np.array([np.log10(x) for x in iDict[ikey][okey]])
        else:
            tempDat = np.array(iDict[ikey][okey])
        tempNames = np.tile(ikey,tempDat.shape[0])
        allDat = np.concatenate((allDat,tempDat))
        allNames = np.concatenate((allNames,tempNames))

    return pd.DataFrame(dict(ctype=allNames,FR=allDat))
##################################################################################################
def get_sessions(desenDict,sleepbox=False,sleeponly=False):
    '''

    '''
    if sleepbox:
        oSess = [x for x in desenDict['desen'].values]
    else:
        oSess = [x for x in desenDict['desen'].values if not x.startswith('sb')]
        #oSess = [x for x in desenDict['desen'].values if not x.startswith('s b')]
        oSess = [x for x in oSess if not x.startswith('ss')]
    if sleeponly:
        oSess = [x for x in desenDict['desen'].values if x.startswith('sb')]

    return oSess
#######################################################################################
def get_descode(df,sesstype,debug=False,exact=False):
    '''
    '''
    descode = df[df['desen'].str.contains(sesstype)==True]
    if exact:
        descode = df[df['desen'].str.strip()==sesstype]
    if debug:	
        print(descode)

    return descode
#######################################################################################
def get_cellcode(df,celltype):
    '''
    df = 
    '''
    cellrow = []
    if celltype=='all':
        cellrow = df[df['des'].str.contains(r'p|b|x|a')]
    else:		
    	cellrow = df[df['des'].str.contains(celltype)==True]

    return cellrow
#######################################################################################
def get_session_duration(df,sess,tconv=20000.0):
    getSess = df[df['desen'].str.contains(sess)==True]
    sessDur = (getSess['end_t'].iloc[0] - getSess['start_t'].iloc[0]) / tconv

    return sessDur
#######################################################################################
def get_sess_resofs_by_str(desen,sess,tconv=(20000/20000)):

    return int(tconv * desen['end_t'][desen['desen']==sess].values[0])
#######################################################################################
def get_sess_resofs_by_indx(baseblock,sindx,tconv=(20000/20000),extt='.col2resofs',indx=True):
    '''
    returns the session end time in samples, when baseblock(fullpath)
    and session index (e.g. 4) are passed - indx must be true
    or when the full session-level filename is passed baseblock=msm04-160720_2
    e.g. baseblock = '/mnt/smchugh/lfpd4/SF/msm04-160721/msm04-160721'
    sindx = 4
    will convert from 20kHz samples to 1250Hz or whatever given tconv value
    '''
    if indx:
        fullpath = baseblock + '_' + str(sindx) + extt
    else:
        fullpath = baseblock + extt
    try:
        tempDat = np.loadtxt(fullpath)
        odata = int(tempDat[1] * tconv)
    except:
        print('File cannot be found: {0:}'.format(fullpath))
        odata = None
    return odata
#######################################################################################
def get_sess_by_target_str(sessions,target_str):
    '''
    '''
    return [x for x in sessions if target_str in x]
#######################################################################################
def get_sess_key_from_filebase(desen,okey):
    '''
    
    '''
    return desen[desen['filebase']==okey]['desen'].values[0]
#######################################################################################
def get_all_times(ipath,desen,sessions,pulse_ext,tconv=None):
    '''
    
    '''
    otimes = {}
    for nn,sess in enumerate(sessions):
        otimes[sess] = get_pulsetimes(ipath,desen,sess,ext=pulse_ext,tconv=tconv)
    return otimes
#######################################################################################
def get_pulsetimes_one_sess(sess_fname,ext,one_col=False):
    ''' 
    Read in the file, store as <DataFrame> and return
    '''
    if one_col:
        interval = pd.read_csv(sess_fname+ext, sep='\s+', header=None, names=['begin'])
    else:
        interval = pd.read_csv(sess_fname+ext, sep='\s+', header=None, names=['begin','end'])
        interval['end'] = (interval['end']).astype(int)

    interval['begin'] = (interval['begin']).astype(int)

    return interval
#######################################################################################
def get_pulsetimes(ipath,df,sesstype,ext='.audio_pulse',tconv=None,debug=False):
    '''
    input: path to the pulse file, desen(df), session string, pulse extension,
    output: returns a dataframe
    '''

    session = get_descode(df,sesstype)
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
#######################################################################################
def generate_df_pulse(temp_times,lkey,width=100,spk_sr=20000,single_col=False):
    '''
    
    '''
    df_pulse = pd.DataFrame().reindex(columns=temp_times.columns)
    if lkey == 'light':
        df_pulse['begin'] = temp_times['begin']
        df_pulse['end'] = temp_times['begin'] + int((width/1000)*spk_sr) # width in ms
    elif lkey == 'nolight':
        df_pulse['begin'] = temp_times['begin'] - int((width/1000)*spk_sr) # width in ms
        df_pulse['end'] = temp_times['begin']
        
    return df_pulse
#############################################################################################################
def generate_2col_pulsetimes(idata,offset=[0.025,0.025],spk_sr=20000,from_array=False):
    '''
    
    '''
    df = pd.DataFrame(columns=['begin', 'end'])
    
    if from_array:
        df['begin'] = (idata - offset[0] * spk_sr).astype(int)
        df['end'] =   (idata + offset[1] * spk_sr).astype(int)
    else:
        df['begin'] = (idata['begin'] - offset[0] * spk_sr).astype(int)
        df['end'] =   (idata['begin'] + offset[1] * spk_sr).astype(int)
    
    return df
#############################################################################################################
def gen_valid_times(df):
    odata = np.arange(df['begin'][1],df['end'][1]+1)
    for nn in range(2,len(df)):
        temptimes = np.arange(df['begin'][nn],df['end'][nn]+1)
        odata = np.hstack((odata,temptimes))

    return odata
#######################################################################################
def get_all_edges(itimes,sessions,nbins,binwidth,fw):
    oedges = {}
    for sindx,sess in enumerate(sessions):
        oedges[sess] = pulse_edges(itimes[sess],nbins=nbins,binwidth=binwidth,fw=fw)

    return oedges
########################################################################################    
def get_all_before_after_edges(itimes,sessions,binwidth,nmsBefore,nmsAfter,sr=20000):
    oedges = {}
    for sindx,sess in enumerate(sessions):
        temp_times = itimes[sess]
        ref_times = np.unique(temp_times['begin'].values)
        oedges[sess] = generate_edges(ref_times,binwidth,nmsBefore,nmsAfter,sr=sr)

    return oedges
#############################################################################################
## Divide pulse files into nn (e.g. 150) evenly spaced bins, using interval['begin'][xx]
#############################################################################################
def pulse_edges(df,nbins=150,binwidth=100,sr=20000,fw=False): 
    ''' 
    Input arguments:
	    df is a dataframe with fields of 'index' 'begin' 'end'
	    df should be the auditory or light (or something) pulse file
	    nbins should be the number of bins required, default = 150, e.g. 100ms for a 15s pulse
	    sr is the sample rate, default is 20,000 but change to 1250 for lfp analysis
    Outputs: 
	    the function returns one output, oedges, which is a list of length trials
	    each list contains a vector of time 'edges' for each trial, to be used in histograms
	    this list start at 0 (trial 1) and e.g. ends at 19 (e.g. trial 20)
    '''  
    ##########################################################################################
    eachms = sr/1000 # this converts sr to ms ################################################
    oedges = []
    for xx in df.index:
        startbin = df['begin'][xx] 
        # remember that pulse files are indexed starting at 1, not 0
        endbin = df['end'][xx]
        if fw:
            tempdat = np.round(np.linspace(startbin,endbin+1,nbins))
        else:
            tempdat = np.round(np.arange(startbin,endbin+1,binwidth*eachms,dtype=int))
        oedges.append(tempdat)
    return oedges
#######################################################################################
def generate_edges(itimes,binwidth,nmsBefore,nmsAfter,sr=20000):
    '''
    Generates edges for a single list (e.g. spikes times)
    itimes is a list of ref time of events
    binwidth, nmsBefore, nmsAfter are given in milliseconds
    the default sampling rate (sr) is 20,000 Hz
    '''
    oedges = []
    eachms = sr / 1000
    for indx,val in enumerate(itimes):
        startbin = val-eachms*nmsBefore
        endbin = val+eachms*nmsAfter
	# check first value
        if startbin > 0:
            oedges.append(np.round(np.arange(startbin,endbin+1,binwidth*eachms,dtype=int)))
    return oedges
#######################################################################################
def generate_fixed_intervals(width,maxT,minT=0,tconv=1):
    '''
    
    '''
    tmin = minT * tconv
    tmax = maxT * tconv
    width = width * tconv
    tempint = np.arange(tmin,tmax+1,width)

    intervals = np.array([],dtype='int').reshape(0,2)
    for t in range(tempint.shape[0]-1):
        array_to_add = np.array([tempint[t], tempint[t+1]],dtype='int').reshape(1,2)
        intervals = np.concatenate((intervals,array_to_add),axis=0)
       
    return intervals
#######################################################################################
def get_all_3dIFRmat(sessions,desen,cluid_inds,iedges):
    '''
    Sessions must be a list of sessions as a string,
    Sessions = ['csmb','cspb','csib','csma','cspa','csia']
    '''
    oMat = {}	
    for sindx,sess in enumerate(sessions):	
        session = get_descode(desen,sess)
        ## this returns a dataframe with session info
        ## the file name is contained in the 'filebase' field
        fname = session.iloc[0]['filebase']
        print('res and clu will be retrieved from {}'.format(fname))
        ## then we pass this filename to the function below
        res,clu = vbf.LoadSpikeTimes(fname,trode=None,MinCluId=2,res2eeg=(20000./20000))
        oMat[sess] = generate_3Difr_matrix(res,clu,cluid_inds,iedges[sess])
        ## this returns a time x cells x trials matrix, oMat

    return oMat
#########################################################################################
def generate_3Difr_matrix(res,clu,cluids,edges):
    '''
    
    '''
    try:
        omatrix3d = np.zeros(( len(edges[1])-1,len(cluids),len(edges) ))
    except:
        omatrix3d = np.zeros(( 1,len(cluids),len(edges) ))

    for nn in range(0,len(edges)):
        omatrix3d[:,:,nn] = generate_ifr_matrix(res,clu,cluids,edges[nn])

    return omatrix3d
############################################################################
def generate_ifr_matrix(res,clu,cluids,edges):
    '''

    '''
    omatrix = np.zeros((len(edges)-1,len(cluids)))
    #print(omatrix.shape)
    for indx,val in enumerate(cluids):
        idata = res[clu==val]
        temp,oedges = np.histogram(idata,bins=edges)
        omatrix[:,indx] = temp

    return omatrix
############################################################################
def get_ifr_trk_eachdir(eachdir,
                        width=0.1,
                        ctype_list=['pdg','p3','p1'],
                        pulse_ext='.light_pulse',
                        adjust_okey=True,
                        stype='nov_one_sess',
                        theta=False,
                        trk_smoothing=5,
                        minT_s=0,
                        maxT_s=None):
    '''

    '''
    tconv = get_tconv()
    spk_sr = tconv['spk_sr']; lfp_sr = tconv['lfp_sr']; 
    res2eeg = tconv['spk_lfp']; lfp2trk = tconv['lfp_trk']
    
    ipath,bsnm,baseblock,par,desen = get_mouse_info_single(eachdir)
    units = load_units(baseblock, par=par, each_trode_ext='.des.', all_trode_ext='.desf')
    origIDs = get_cell_inds_one_mouse(units,ctype_list=ctype_list,exact=True)

    
    cluID_list = multi_ctype_IDs(ctype_list,origIDs,mouse=None)
    print('{0:} {1:} has {2:} cells'.format(str(0),bsnm,len(cluID_list)))

    sessions = get_sessions(desen,sleepbox=True)
    if stype is not None:
        sessions = get_valid_sessions(sessions,stype=stype)
    print(sessions)
    print()
    ####################################
    IFRAllSess = {}
    trkAllSess = {}
    
    for sindx,sess in enumerate(sessions):
        
        spkMat = {}
        trkMat = {}

        ## Get session info
        descode = get_descode(desen,sess)
        print(descode)
        maxT = int(descode['end_t'] - descode['start_t'])
        maxT_lfp = int(maxT*res2eeg)
        print(maxT,maxT/spk_sr)
        baseSession = descode['filebase'].values[0]
        print(baseSession)

        ## Get session filename
        sess_fname = baseblock.rsplit('/',1)[0] + '/' + baseSession

        ## Get tracking for the session
        track = vbf.load_tracking(sess_fname,smoothing=trk_smoothing)
        
        ## Get max session length index
        if maxT_s == None:
            maxT_s = int(maxT_lfp / lfp_sr)
        
        # Get all spikes from a session                                
        spikes,binspikes = get_sess_binspikes(sess_fname,
                                                  maxT_lfp,
                                                  cluID_list=cluID_list,
                                                  dT=1,
                                                  tconv=res2eeg)
        # Get all intervals
        # Get theta cycles
        if theta:
            tetLabel = slf.get_ripple_chan(baseblock)
            thetacycles = get_theta_cycles(sess_fname,tetLabel)
            intervals = get_valid_cycles(thetacycles)
        else:            
            intervals = generate_fixed_intervals(width,maxT_s,tconv=lfp_sr)
        print(len(intervals))
        
        ## Get all pulse intervals
        try:
            light = get_pulses(sess_fname,pulse_ext=pulse_ext,tconv=res2eeg)
            print('light',light)
            if light:
                pulses_present = True
                print('not none')
            else:
                pulses_present = False
        except FileNotFoundError:
            pulses_present = False
            print('There are no {0:} files. Try another extension?'.format(pulse_ext))
            
        if pulses_present:
            theta_light = get_theta_light(intervals,light,maxT_lfp)
            # Get the nolight theta cycles by taking the difference
            theta_nolight = get_no_light(intervals,theta_light)
            spkMat['nolight'] = IFR_by_theta(theta_nolight,binspikes)
            trkMat['nolight'] = trk_by_theta(theta_nolight,track,tconv=lfp2trk)
        else:
            theta_nolight = intervals
            spkMat['nolight'] = IFR_by_theta(theta_nolight,binspikes)
            trkMat['nolight'] = trk_by_theta(theta_nolight,track,tconv=lfp2trk)
        
        if adjust_okey:
            if sess.startswith('f1'):
                okey = 'f1b'
            if sess.startswith('n1'):
                okey = 'n1b'
        else:
            okey = sess

        IFRAllSess[okey] = spkMat
        trkAllSess[okey] = trkMat
    ######################################################################################
    return IFRAllSess,trkAllSess
##########################################################################################
def hist2(data1,data2,nbins):
    hist, xedges, yedges = np.histogram2d(data1,data2,nbins)
    xBinCenters = np.convolve(xedges,[.5,.5],'same')
    xBinCenters = xBinCenters[1::]
    yBinCenters = np.convolve(yedges,[.5,.5],'same')
    yBinCenters = yBinCenters[1::]	

    Edges = [None]*2
    Edges[0] = xedges
    Edges[1] = yedges

    return hist,xBinCenters,yBinCenters,Edges
########################################################################################
def calc_meanFR_one_mouse(alldesen,allBaseblock,units,mouse,intv_type,sleepbox=True):
    '''

    '''
    sessions = get_sessions(alldesen[mouse],sleepbox=sleepbox)
    desen,baseblock,ipath,bsnm = get_mouse_info(alldesen,allBaseblock,mouse)
    intvTimes = get_all_times(ipath,desen,sessions,intv_type)
    cluID_list = list(units[mouse]['des'].index)
    
    allSessFR = {}
    for sindx,sess in enumerate(sessions):
        ### Select session
        sessionLabel = get_descode(desen,sess)
        fname = baseblock + '_' + str(sessionLabel['filebase'].index[0])
        #################################################################################
        res,clu = \
        vbf.LoadSpikeTimes(fname,trode=None,MinCluId=2,res2eeg=(20000./20000))
        #################################################################################
        allSessFR[sess] = calc_meanFR(res,clu,cluID_list,intvTimes[sess],sr=20000.)
    _,gmeanFR = grandMeanFR(allSessFR,sessions)
    
    return allSessFR,gmeanFR
######################################################################################
def calc_meanFR_one_mouse_sess(alldesen,allBaseblock,units,mouse,intv_type,sessions,clist=None):
    '''
    Like above but now we pass sessions, rather than generate in the function
    '''
    desen,baseblock,ipath,bsnm = get_mouse_info(alldesen,allBaseblock,mouse)
    intvTimes = get_all_times(ipath,desen,sessions,intv_type)
    if clist == None:
        cluID_list = list(units[mouse]['des'].index)
    else:
        cluID_list = clist
    
    allSessFR = {}
    for sindx,sess in enumerate(sessions):
        ### Select session
        sessionLabel = get_descode(desen,sess)
        fname = baseblock + '_' + str(sessionLabel['filebase'].index[0])
        ##############################################################################
        res,clu = \
        vbf.LoadSpikeTimes(fname,trode=None,MinCluId=2,res2eeg=(20000./20000))
        ##############################################################################
        allSessFR[sess] = calc_meanFR(res,clu,cluID_list,intvTimes[sess],sr=20000.)
    _,gmeanFR = grandMeanFR(allSessFR,sessions)
    
    return allSessFR,gmeanFR
######################################################################################
def calc_meanFR(res,clu,cluid_inds,itimes,sr=20000):
    ''' 
    Input: session level res and clu
        cluIDs 
        itimes e.g. from .rrem file
    Output: cluID x 4 array 
        with cluID in col 1,  
        intv spike count in col 2 (i.e. during .rrem),
        total intv duration (in samples) in col 3
        firing rate in Hz col 4
   '''
    
    intervalFR = np.zeros((len(cluid_inds),4))
    sstart = (itimes['begin']).values.astype(int)
    eend = (itimes['end']).values.astype(int)
    for nn,xx in enumerate(cluid_inds):
        tempdat = res[clu==xx]
        tempCount = 0
        tempDur = 0
        for index,times in enumerate(sstart):
            eachInterval = eend[index] - sstart[index]
            tempDur = tempDur + eachInterval
            tempCount = \
		tempCount + len(tempdat[(tempdat>sstart[index])*(tempdat<eend[index])])
        intervalFR[nn,0] = xx        # the cluID
        intervalFR[nn,1] = tempCount # the spike count
        intervalFR[nn,2] = tempDur   # the total duration of intv_type in that session
        intervalFR[nn,3] = (tempCount / tempDur) * sr # the firing rate in Hz

    return intervalFR
######################################################################################
def calc_spk_counts(res,clu,cluid_inds,itimes,nTrials,sr=20000):
    ''' 
    Input: session level res and clu
        cluIDs 
        itimes e.g. from .light_pulse dataframe
        nTrials (e.g. number of pulses)
    Output: cluID x 3 x nTrials array 
        with cluID in col 1,  
        intv spike count in col 2
        total intv duration (in samples) in col 3
   '''
    
    intervalFR = np.zeros((len(cluid_inds),4,nTrials))
    
    sstart = (itimes['begin']).values.astype(int)
    eend = (itimes['end']).values.astype(int)

    for cindx,cluID in enumerate(cluid_inds):
        tempdat = res[clu==cluID]
        for tindx,times in enumerate(sstart):
            tempDur = eend[tindx] - sstart[tindx]
            tempCount = len(tempdat[(tempdat>sstart[tindx])*(tempdat<eend[tindx])])
            
            intervalFR[cindx,0,tindx] = cluID     # the cluID
            intervalFR[cindx,1,tindx] = tempCount # the spike count
            intervalFR[cindx,2,tindx] = tempDur   # the total duration of intv_type in that session in sr resolution
            intervalFR[cindx,3,tindx] = (tempCount / tempDur) * sr # firing rate in Hz

    return intervalFR
######################################################################################################################
def FR_over_sess(allSessFR,sessions,cluID):
    
    eachSessFR = np.zeros((len(sessions)))
    for sindx,sess in enumerate(sessions):
        eachSessFR[sindx] = np.round(allSessFR[sessions[sindx]][cluID-2,3],3)   
    
    return eachSessFR
######################################################################################
def grandMeanFR(iDict,sessions,col=1,sr=20000,debug=False):
    '''
    Updated 10th September 2020.
    Input:
	Dictionary of FRs by sess (generated by sbf.calc_meanFR_one_mouse 
        or sbf.calc_meanFR_one_mouse_sess),
        List of Sessions
    Output:
    FRarray: 2d array of FRs (session x cells)
    gmeanFR: vector of FRs (len = number of cells)
    '''

    # Create empty array for spike counts per session (sess x cells)
    arr_shape = iDict[sessions[0]][:,col].shape
    FRarray = np.empty((0,arr_shape[0]))
    #print(FRarray.shape)

    # Initialize time counter note that sDur is in sample time (20k)
    sDur = 0
    
    # for each session, extract spike counts
    for sindx,sess in enumerate(sessions):
        array_to_add = np.array(iDict[sess][:,col])
        if debug:
            print(array_to_add)
        FRarray = np.append(FRarray,[array_to_add], axis=0)
        try:
            sDur = sDur + iDict[sess][0,2]
        except IndexError:
            sDur = sDur    
    #print(FRarray.shape)  
    gmeanFR = np.nansum(zero_to_nan(FRarray),axis=0) / (sDur / sr)
    
    return FRarray,gmeanFR
######################################################################################
def get_all_meanFR_eachTrial(sessions,desen,cluid_inds,itimes):
    '''
    
    '''
    oMat = {}
    print(sessions)
    for sindx,sess in enumerate(sessions):
        print(sess)
        session = get_descode(desen,sess)
        fname = session.iloc[0]['filebase']
        res,clu = \
            vbf.LoadSpikeTimes(fname,trode=None,MinCluId=2,res2eeg=(20000./20000))
        oMat[sess] = calc_meanFR_eachTrial(res,clu,cluid_inds,itimes[sess])

    return oMat
######################################################################################
def calc_meanFR_eachTrial(res,clu,cluid_inds,itimes,nTrials=20,sr=20000):
    '''
    '''
    intervalFR = np.zeros((len(cluid_inds),nTrials+1))
    
    sstart = (itimes['begin']).values.astype(int)
    eend = (itimes['end']).values.astype(int)
    
    for nn,xx in enumerate(cluid_inds):
        tempdat = res[clu == xx]
        intervalFR[nn,0] = xx
        for index,times in enumerate(sstart):
            eachInterval = eend[index] - sstart[index]
            spkCount = \
            int(len(tempdat[(tempdat > sstart[index]) * (tempdat < eend[index])]))
            intervalFR[nn,index+1] = np.round((spkCount / (eachInterval / sr)),3)

    return intervalFR
###################################################################################################

######################################################################################
# Autocorrelation and cross-correlation functions
######################################################################################
def random_add(arr,int_to_add=1):

    mask = np.random.choice([False, True],size=len(arr)//2,replace=True)
    result = arr.copy()
    result[:len(arr)//2][mask] += int_to_add

    return result
######################################################################################
def clean_clu(spikes,thresh=3,int_to_add=0):
    '''
    
    '''
    x = (spikes / 20).astype(int)
    isi = np.diff(x)

    all_isi_inds = np.arange(x.shape[0])
    bad_isi_inds = np.squeeze(np.argwhere(isi < thresh)+1)
    good_isi_inds = np.setdiff1d(all_isi_inds, bad_isi_inds, assume_unique=False)

    return spikes[good_isi_inds]
#######################################################################################
def autocorrelation(baseblock,desen,sessions,cluID,binwidth=1,nms=[40.,40.],clean_ref=False,thresh=3):
    '''
    This function takes a desen dataframe and a list of sessions
    and a cluID and generates an autocorrelation based on firing in all sessions given
    binwidth and nmsBefore, nmsAfter are passed as options
    returns a matrix which is timebins x 1 x totalspikes
    use plot_ac to visualize
    '''
    # Create output matrix
    tbins  = int(2*(nms[0]/binwidth))
    oMat = np.zeros((tbins,1,0))
    # Loop through sessions
    for sindx,sess in enumerate(sessions):
        ### Select session
        sessionLabel = get_descode(desen,sess)
        fname = baseblock + '_' + str(sessionLabel['filebase'].index[0])
        ############################################################################################
        res,clu = vbf.LoadSpikeTimes(fname,trode=None,MinCluId=2,res2eeg=(20000./20000))
        refSpikes = res[clu==cluID]
        if clean_ref:
            refSpikes = clean_clu(refSpikes,thresh=thresh)
        refEdges = generate_edges(refSpikes,binwidth=binwidth,nmsBefore=nms[0],nmsAfter=nms[1])
        try:
            tempMat = generate_3Difr_matrix(res,clu,[cluID],refEdges) # changed from generate_3Difr_matrix
            oMat = np.concatenate((oMat,tempMat),axis=-1)
        except:
            print('no spikes in session {} {}'.format(sindx,sess.replace(" ","")) )

    return oMat
#####################################################################################################
def crosscorrelation(baseblock,desen,sessions,refcluID,targcluID_list,binwidth=1,nms=[40.,40.]):
    '''
    This function takes a desen dataframe and a list of sessions
    and a cluID and generates a cross-correlation based on firing in all sessions given
    binwidth and nmsBefore, nmsAfter are passed as options
    returns a matrix which is timebins x nCells x totalspikes
    use plot_ac to visualize
    '''

    # Create output matrix
    tbins  = int(2*(nms[0]/binwidth))
    nCells = len(targcluID_list)
    oMat = np.zeros((tbins,nCells,0))

    # Loop through sessions
    for sindx,sess in enumerate(sessions):
        # Select session
        sessionLabel = get_descode(desen,sess)
        fname = baseblock + '_' + str(sessionLabel['filebase'].index[0])
        ############################################################################################
        res,clu = vbf.LoadSpikeTimes(fname,trode=None,MinCluId=2,res2eeg=(20000./20000))
        refSpikes = res[clu==refcluID]
        refEdges = generate_edges(refSpikes,binwidth=binwidth,nmsBefore=nms[0],nmsAfter=nms[1])
        try:
            tempMat = generate_3Difr_matrix(res,clu,targcluID_list,refEdges) # changed from generate_3Difr_matrix
            oMat = np.concatenate((oMat,tempMat),axis=-1)
        except:
            print( 'no spikes in session {} {}'.format(sindx,sess.replace(" ","")) )

    return oMat
#####################################################################################################
def gen_prob_2d(idata):
    '''
    input: idata, e.g. spike counts in (timebins x trials) for a single cell
    output: sum of spikes across all trials / total number of spikes
    '''
    return np.nansum(idata,axis=1) / np.nansum(idata,axis=(0,1))
#####################################################################################################
def zscore_1dList(idata):
    mu = np.nanmean(idata)
    sd = np.nanstd(idata)
    return [((x-mu)/sd) for x in idata]
#####################################################################################################
def z_test_model(idata,isem,combomat):
    '''
    '''
    z = np.zeros((combomat.shape[0]))
    p = np.zeros((combomat.shape[0]))
    for indx,value in enumerate(combomat):
        indd1 = combomat[indx][0]
        indd2 = combomat[indx][1]
        #print indx,indd1,indd2
        z[indx] = (idata[indd1] - idata[indd2]) / ((isem[indd1] + isem[indd2]) / 2)
        p[indx] = stats.norm.sf(abs(z[indx])) * 2

    return z, p
########################################################################################################################################
def plot_ac(iMat,refCluID,units,mouse,nms=[40.,40.],binwidth=1.0,baseline=None,figsize=[8,8],xlim=None,ac=True,prob=False,zscore=False):
    '''
    Takes a time x 1 x totalspikes
    '''
    if prob and zscore:
        print('prob and zscore set to True, is this what you want?')
    rr,cc = 1,1
    fwd,fht = figsize
    fig, ax = plt.subplots(rr,cc,figsize=cm2inch(fwd,fht))
    ylab = 'Counts'
    ###############################################################################
    stitle = str(refCluID) + '_' + units[mouse]['des'][refCluID]
    ###############################################################################
    idata = np.nansum(iMat,axis=-1) # sums all spikes in last dim of iMat
    
    if prob:
        idata = idata / np.nansum(iMat,axis=(0,-1))
        ylab = 'Probability'
    if zscore:
        midpoint = int(nms[0] / binwidth)
        if baseline is not None:
            startpoint = midpoint - int((1000 / binwidth) * baseline)
        else:
            startpoint = 0
        idata = (idata - np.nanmean(idata[startpoint:midpoint,0])) / np.nanstd(idata[startpoint:midpoint,0]) 
        ylab = 'Z-score'
    #########################################################################################################
    zero_ind = int(nms[0] / binwidth)
    if ac:
        idata[zero_ind] = 0 # sets timebin 0 to zero 
    xpts = np.arange(-nms[0],nms[1],binwidth)

    ax.bar(xpts,np.squeeze(idata),width=binwidth,color='k',align='edge')
    if xlim == None:
    	ax.set_xlim(-nms[0],nms[1])
    else:
    	ax.set_xlim(-xlim[0],xlim[1])
    ax.set_ylabel(ylab,fontname='Arial')
    ax.set_xlabel('Time (ms)',fontname='Arial')
    ax.set_title(stitle,y=+1.0,x=+.25,fontsize=10)
    ###############################################################################
    return fig,ax
######################################################################################################
'''
def pulsecorrelation(baseblock,desen,sessions,cluID,binwidth=1,nms=[40.,40.],ext='.light_pulse'):

    This function takes a baseblock,desen dataframe and a list of sessions
    and a cluID and generates an cross-correlation based on pulsetimes in all sessions given
    binwidth and nmsBefore, nmsAfter are passed as options
    returns a matrix (oMat) which is timebins x 1 x totalspikes
    use plot_ac to visualize
    
    # Create output matrix
    tbins  = int(2*(nms[0]/binwidth))
    oMat = np.zeros((tbins,1,0))
    # Loop through sessions
    for sindx,sess in enumerate(sessions):
        ### Select session
        sessionLabel = get_descode(desen,sess)
        fname = baseblock + '_' + str(sessionLabel['filebase'].index[0])
        ##############################################################################################
        res,clu = \
        vbf.LoadSpikeTimes(fname,trode=None,MinCluId=2,res2eeg=(20000./20000))
        ipath = os.path.split(baseblock)[0] + '/'
        #refTimes = []
        refTimes = get_pulsetimes(ipath,desen,sess,ext,tconv=None,debug=False)
        refTimes = refTimes['begin'].values
        if len(refTimes) > 0:
            refEdges = generate_edges(refTimes,binwidth=binwidth,nmsBefore=nms[0],nmsAfter=nms[1])
            tempMat = generate_3Difr_matrix(res,clu,[cluID],refEdges)
            oMat = np.concatenate((oMat,tempMat),axis=2)
        ##############################################################################################          
    return oMat # timebins x 1 x totalspikes
'''
################################################################################################################
def pulsecorrelation(baseblock,desen,sessions,cluID,binwidth=1,nms=[40.,40.],ext='.light_pulse',min_dur=None,max_dur=None,samp_rate=20000,single_clu=True):
    '''
    This function takes a baseblock,desen dataframe and a list of sessions
    and a cluID and generates an cross-correlation based on pulsetimes in all sessions given
    binwidth and nmsBefore, nmsAfter are passed as options
    returns a matrix (oMat) which is timebins x 1 x totalspikes
    use plot_ac to visualize
    '''
    # Create output matrix
    tbins  = int(2*(nms[0]/binwidth))
    oMat = np.zeros((tbins,1,0))
    if single_clu:
        cluID_list = [cluID]
    else:
        cluID_list = cluID
    # Loop through sessions
    for sindx,sess in enumerate(sessions):
        ### Select session
        sessionLabel = get_descode(desen,sess)
        fname = baseblock + '_' + str(sessionLabel['filebase'].index[0])
        ##############################################################################################
        res,clu = vbf.LoadSpikeTimes(fname,trode=None,MinCluId=2,res2eeg=(20000./20000))
        ipath = os.path.split(baseblock)[0] + '/'
        reftemp = get_pulsetimes(ipath,desen,sess,ext,tconv=None,debug=False)
        duration = [(y-x) for x,y in zip(reftemp['begin'].values,reftemp['end'].values)]
        #print(duration)
        refTimes = np.unique(reftemp['begin'].values)
        if min_dur is not None:
            refTimes = [x for (x,y) in zip(refTimes,duration) if y > min_dur*samp_rate]
            min_pulses = 1
        if max_dur is not None:
            refTimes = [x for (x,y) in zip(refTimes,duration) if y < max_dur*samp_rate]
        else:
            min_pulses = 1
        #print('Min pulse duration is {0:}. Min no pulses/sess is {1:}'.format(min_dur,min_pulses))
        if len(refTimes) > min_pulses: # Must be more than one Light ON trial in the session
            refEdges = generate_edges(refTimes,binwidth=binwidth,nmsBefore=nms[0],nmsAfter=nms[1])
            tempMat = generate_3Difr_matrix(res,clu,cluID_list,refEdges)
            oMat = np.concatenate((oMat,tempMat),axis=-1)
            #oMat = np.concatenate((oMat,tempMat),axis=1)
        ##############################################################################################          
    return oMat # timebins x 1 x totalspikes
################################################################################################################
def pulsecorrelation_multi(baseblock,desen,sessions,cluID_list,binwidth=1,nms=[40.,40.],ext='.light_pulse',min_dur=None,max_dur=None,samp_rate=20000):
    '''
    This function takes a baseblock,desen dataframe and a list of sessions
    and a cluID and generates an cross-correlation based on pulsetimes in all sessions given
    binwidth and nmsBefore, nmsAfter are passed as options
    returns a matrix (oMat) which is timebins x 1 x totalspikes
    use plot_ac to visualize
    '''
    # Create output matrix
    tbins  = int(2*(nms[0]/binwidth))
    oMat = np.zeros((tbins,len(cluID_list),0))
    #if len(cluID_list) == 1:
    #    cluID_list = [cluID_list]
    # Loop through sessions
    for sindx,sess in enumerate(sessions):
        ### Select session
        sessionLabel = get_descode(desen,sess)
        fname = baseblock + '_' + str(sessionLabel['filebase'].index[0])
        ##############################################################################################
        res,clu = vbf.LoadSpikeTimes(fname,trode=None,MinCluId=2,res2eeg=(20000./20000))
        ipath = os.path.split(baseblock)[0] + '/'
        reftemp = get_pulsetimes(ipath,desen,sess,ext,tconv=None,debug=False)
        duration = [(y-x) for x,y in zip(reftemp['begin'].values,reftemp['end'].values)]
        #print(duration)
        refTimes = reftemp['begin'].values
        if min_dur is not None:
            refTimes = [x for (x,y) in zip(refTimes,duration) if y > min_dur*samp_rate]
            min_pulses = 1
        if max_dur is not None:
            refTimes = [x for (x,y) in zip(refTimes,duration) if y < max_dur*samp_rate]
        else:
            min_pulses = 0
        #print('Min pulse duration is {0:}. Min no pulses/sess is {1:}'.format(min_dur,min_pulses))
        if len(refTimes) > min_pulses: # Must be more than one Light ON trial in the session
            refEdges = generate_edges(refTimes,binwidth=binwidth,nmsBefore=nms[0],nmsAfter=nms[1])
            tempMat = generate_3Difr_matrix(res,clu,cluID_list,refEdges)
            oMat = np.concatenate((oMat,tempMat),axis=-1)
        ##############################################################################################          
    return oMat # timebins x len(cluID_list) x totalspikes
######################################################################################################
def plot_group_pulse(iMat,xpts=None,nms=[40.,40.],binwidth=1.0,barcol='k',figsize=[8,8],xlim=None,av='median',savgol=False,npts=3):
    '''
    Takes a time x 1 x totalspikes input as iMat
    ''' 
    rr, cc = 1,1
    fwid, fht = figsize
    fig, ax = plt.subplots(rr, cc, figsize = cm2inch(fwid, fht))
    ################################################################################
    stitle = 'Group mean'
    ################################################################################
    if av == 'mean':
        idata = np.nanmean(inf_to_nan(iMat), axis=-1) # mean all spikes in last dim of iMat
    if av == 'median':
        idata = np.nanmedian(inf_to_nan(iMat),axis=-1)
    if savgol:
        idata = savgol_filter(np.squeeze(idata),npts,1)
    ylab = 'Z-score'
    idata = np.squeeze(idata)
    ################################################################################
    zero_ind = int(nms[0] / binwidth)
    if xpts is None:
        xpts = np.arange(-nms[0], nms[1], binwidth)

    ax.bar(xpts, idata, color=barcol, align='edge', width=binwidth)
    if xlim == None:
        ax.set_xlim(-nms[0], nms[1])
    else:
        ax.set_xlim(-xlim[0], xlim[1])
    #ax.axvline(0, linewidth =.5, color='r', linestyle='--')
    # ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylab)
    ax.set_xlabel('Time (ms)')
    ax.set_title(stitle, y = +1.0, x = +.25, fontsize = 10)
    ###############################################################################
    return fig, ax
##########################################################################################
def plot_group_average(lightC2,
                       group_type,
                       ctype,
                       #nms=[15000,15000],
                       #binwidth=250,
                       ylim=(-.8,.8),
                       ytick_width=.4,
                       #xtick_width=5000,
                       figsize=(10,6),
                       fscale=(14,16),
                       #pulse_width = 15000,
                       pulse_col = '#ffff00',
                       box=True,
                       box_col='#ffcc00ff',
                       err_bar=False):
    '''
    
    '''
    try: sns.reset_orig()
    except: print('seaborn is not imported')
    ###################################################################################################################
    cellcol = {'pdgL':PURPLE,'pdg':BLUE,'p3':RED,'p1':ORNG,'bdg':GREEN,'b3':gray2,'b1':PINK}
    ###################################################################################################################
    ## Basic parameters ##
    ###################################################################################################################
    if group_type == 'chr2':
        nms = [80,80]
        binwidth = nms[0] / 40
        xscale = 1
        xtick_width = 5 * xscale
        #pulse_col = '#0099ff'
        pulse_width = 5
        zthresh = 3.0
        midpoint = int(nms[0] / binwidth)
        baseline = None
        startpoint = 1
        xlab = 'Time (ms)'
        ylab = 'Z-score'
        xmin,xmax = [-nms[0],nms[1]]
    ####################################################################################################################
    group_list = ['chr2_gfp','grm','aged','cfos','archT']
    print(group_type)
    if group_type in group_list:
        nms = [15000,15000]
        binwidth = 250 #nms[0] / 40
        xscale = 1000
        xtick_width = 5 * xscale
        #pulse_col = '#ffff00'
        pulse_width = 15000 #nms[1] 
        zthresh = -2.0
        midpoint = int(nms[0] / binwidth)
        baseline = 15 #.4
        startpoint = midpoint - int((1000 / binwidth) * baseline)
        xlab = 'Time (s)'
        ylab = 'Z-score FR'
        xmin,xmax = (-15000,15000)
        xlab = 'Times (ms)'
    ###################################################################################################################
    alpha = 0.3
    lw = 0.5
    pad = 1.5
    set_lims = True
    av = 'mean'
    savgol = True
    barcol = 'k'
    ###################################################################################################################
    ## Generate plot, axes 
    ###################################################################################################################
    fig, ax = plot_group_pulse(lightC2,
                               nms=nms,
                               binwidth=binwidth,
                               barcol=barcol,
                               figsize=figsize,
                               av=av,
                               savgol=savgol)
    if err_bar:
        meanval = np.nanmean(lightC2,axis=-1)
        stderr = np.nanstd(lightC2,axis=-1) / np.sqrt(lightC2.shape[-1])
        xpts = np.linspace(-nms[0],nms[1],lightC2.shape[0])
        ax.plot(xpts,meanval+stderr,color=BLUE,linewidth=lw,linestyle='--')
        ax.plot(xpts,meanval-stderr,color=BLUE,linewidth=lw,linestyle='--')
    ###################################################################################################################
    # set / get x and y limits
    #ax.set_xlim(xmin,xmax)
    ax.set_ylim(ylim)
    ymin,ymax = ax.get_ylim()
    ymin_ = ymin + .05

    # add laser pulse
    ax.fill_between([0,pulse_width],ymax,ymin_,color=pulse_col,alpha=alpha,edgecolor=None,linewidth=0)
    
    # add black line at 0 on the y-axis
    ax.axhline(0,linewidth=lw)

    # add box around fill
    if box:
        ax.plot((-nms[0], 0), (ymin_, ymin_), linewidth=lw, color=box_col)
        ax.plot((0,nms[1]), (ymax,ymax), linewidth=lw, color=box_col)
        ax.plot((nms[1],16000), (ymin_,ymin_), linewidth=lw, color=box_col)
        ax.plot((0,0), (ymin_,ymax), linewidth=lw, color=box_col)
        ax.plot((nms[1],nms[1]), (ymin_,ymax), linewidth=lw, color=box_col)

    # Customize tick marks
    ax.xaxis.set_major_locator(Ticker.MultipleLocator(xtick_width))
    ax.yaxis.set_major_locator(Ticker.MultipleLocator(ytick_width))

    # set axis bounds
    ax.spines['left'].set_position(('outward', 3))
    ax.spines['bottom'].set_position(('outward', 3))
    ax.spines['left'].set_bounds((ymin,ymax))
    ax.spines['bottom'].set_bounds((-nms[0], nms[1]))

    ## adjust ax as appropriate
    ax = adjust_plot_pub(ax,xlab=xlab,ylab=ylab,lw=lw,raster=False,nms=nms,binwidth=binwidth,
                            xscale=xscale,yscale=1,xt_fmt='.0f',yt_fmt='.1f',fscale=fscale,grid=False,
                            gridcolor='k',pad=pad
                            )
    # add title
    ax.set_title('Group Mean: '+ctype+'_n'+str(lightC2.shape[-1]),fontsize=fscale[0])

    return fig,ax
##########################################################################################
def plot_pulse_cross_correl(mouse,
                            cluID,
                            alldesen,
                            allBaseblock,
                            units,
                            group_type,
                            nms=[40,40],
                            binwidth=1,
                            xscale=1,
                            ext='.light_pulse',
                            zscore=True,
                            zlims=(-3,8),
                            rawlims=(0,20),
                            y_offset=0.1,
                            size_scale=5,
                            fwid=4,
                            fht=1):
    '''
    
    '''
    ###################################################################################################################
    try: sns.reset_orig()
    except: print()
    desen,baseblock,ipath,bsnm = get_mouse_info(alldesen,allBaseblock,mouse)
    os.chdir(ipath)
    print(bsnm)
    ###################################################################################################################
    group_list = ['chr2','chr2_pdgL','chr2_archT']
    if group_type in group_list:
        pulse_col = '#0099ff'
        pulse_width = 5
        zthresh = 3.0
        midpoint = int(nms[0] / binwidth)
        baseline = None
        #startpoint = 1
        xlab = 'Time (ms)'
        ylab = 'Z-score'
        xmin,xmax = [-nms[0],nms[1]]
        ymin,ymax = [-5,20]
        xtick_width = 20 * xscale
        ytick_width = 5
        box_lims = [0,5]
        ax_lims = [-nms[0],nms[1]]
        min_dur = None
    ####################################################################################################################
    group_list = ['archT','gfp','grm']
    if group_type in group_list:
        pulse_col = '#ffff00'
        pulse_width = 30000
        zthresh = -2.0
        midpoint = int(nms[0] / binwidth)
        baseline = 8
        startpoint = midpoint - int((1000 / binwidth) * baseline)
        xlab = 'Time (s)'
        ylab = ''#'Z-score FR'
        xmin,xmax = [-5000,35000]
        xtick_width = 15 * xscale
        min_dur = None
    ###################################################################################################################
    if zscore:
        ymin,ymax = [zlims[0],zlims[1]]
        ytick_width = int(ymax/2)
        ylab = 'Z-Score FR'
    else:
        ymin,ymax = [rawlims[0],rawlims[1]]
        ytick_width = ymax #int(ymax/2)
        ylab = 'Counts'

    figsize = [fwid*size_scale,fht*size_scale]
    alpha = 0.2
    lw = 0.5
    pad = 1.5
    set_lims = True
    ###################################################################################################################
    sessions = get_sessions(desen,sleepbox=True)
    lightC = pulsecorrelation(baseblock,desen,sessions,cluID,nms=nms,binwidth=binwidth,ext=ext,min_dur=min_dur)
    print(lightC.shape)
    #################################################################################################################
    fig,ax = plot_ac(lightC,cluID,units,mouse,nms=nms,binwidth=binwidth,baseline=baseline,
                         figsize=figsize,ac=False,prob=False,zscore=zscore)
    ####################################################################################################################
    if set_lims:
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
    ####################################################################################################################
    ax.xaxis.set_major_locator(Ticker.MultipleLocator(xtick_width))
    ax.yaxis.set_major_locator(Ticker.MultipleLocator(ytick_width))
    ####################################################################################################################
    # Plot horizontal line at 0
    ax.axhline(0,linewidth=lw,color='k')

    # set / get x and y limits
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ymin,ymax = ax.get_ylim()
    ymin_ = ymin + y_offset

    allSessFR,gmeanFR = calc_meanFR_one_mouse(alldesen,allBaseblock,units,mouse,'.col2resofs')
    cell_ID = units[mouse]['des'][cluID]
    FRate = 'FR: {0:.2f} Hz'.format(np.round(gmeanFR[cluID-2],1))
    ax.set_title(bsnm + ' ' + str(cluID) + '_' + cell_ID + ': ' + FRate, fontsize=8)

    ax = set_axis_bounds(ax,(xmin,xmax),(ymin,ymax),sp_len=3)
    ax = adjust_plot_pub(ax,xlab,ylab,nms=nms,binwidth=binwidth,xscale=xscale,fscale=[6,7],grid=False,pad=pad)

    # add box around color fill
    ax.fill_between([0,pulse_width],ymax,ymin_,color=pulse_col,alpha=alpha,edgecolor=None,linewidth=0)
    box_col = '#00FFFF' #'k'
    ax.plot((ax_lims[0], box_lims[0]), (ymin_, ymin_), linewidth=lw, color=box_col)
    ax.plot((box_lims[0],box_lims[1]), (ymax,ymax), linewidth=lw, color=box_col)
    ax.plot((box_lims[1],ax_lims[1]), (ymin_,ymin_), linewidth=lw, color=box_col)
    ax.plot((box_lims[0],box_lims[0]), (ymin_,ymax), linewidth=lw, color=box_col)
    ax.plot((box_lims[1],box_lims[1]), (ymin_,ymax), linewidth=lw, color=box_col)

    return fig,ax
#########################################################################################################################################################
def add_box_fill(ax,pulse_width,ymin,ymax,ax_lims,box_lims,lw=0.5,alpha=0.2,pulse_col='#0099ff',box_col='#00FFFF',y_offset=0.1,add_fill=True,add_box=True):
    '''
    '''
    ymin_ = ymin + y_offset
    if add_fill:
        ax.fill_between([0,pulse_width],ymax,ymin_,color=pulse_col,alpha=alpha,edgecolor=None,linewidth=0)
    if add_box:
        ax.plot((ax_lims[0], box_lims[0]), (ymin_,ymin_), linewidth=lw, color=box_col)
        ax.plot((box_lims[0],box_lims[1]), (ymax,ymax),   linewidth=lw, color=box_col)
        ax.plot((box_lims[1],ax_lims[1]),  (ymin_,ymin_), linewidth=lw, color=box_col)
        ax.plot((box_lims[0],box_lims[0]), (ymin_,ymax),  linewidth=lw, color=box_col)
        ax.plot((box_lims[1],box_lims[1]), (ymin_,ymax),  linewidth=lw, color=box_col)
    return ax
#########################################################################################################################################################
def genOccMap(track,validsamples,nbins=70,mazedim=None,debug=False,returnEdges=False):
    '''
    This will generate an occupancy map
    based on the tracking data and validsamples (i.e. intervals)
    the number of edges = (bins+1)
    the min and max x dimensions are mazedim[0] and mazedim[1]
    the min and max y dimensions are mazedim[2] and mazedim[3]
    or if absent, will be taken from the tracking data
    '''
    nEdges = nbins+1
    if mazedim is None:
    	Xedges = np.linspace(np.min(track['x']),np.max(track['x']),nEdges)  	
    	Yedges = np.linspace(np.min(track['y']),np.max(track['y']),nEdges)
    else:
        Xedges = np.linspace(mazedim[0],mazedim[1],nEdges)  	
        Yedges = np.linspace(mazedim[2],mazedim[3],nEdges)
    if debug:
        print(Xedges[0],Xedges[-1])
        print(Yedges[0],Yedges[-1])
    totalPos = validsamples.astype(int)[:-1]
    
    y = np.copy(track['y'])[totalPos]
    y = y[~np.isnan(y)]
    x = np.copy(track['x'])[totalPos]
    x = x[~np.isnan(x)]
    OccMap,xBinCenters,yBinCenters,Edges = hist2(x,y,[Xedges,Yedges]) 
    if returnEdges:
        return OccMap,Xedges,Yedges
    else:
        return OccMap
#######################################################################################
def genSpikeMap(track,spikes,nbins=70,mazedim=None,debug=False):
    '''
    spikes must be the spike times for a single cluster at 20,000Hz
    e.g. spikes = res[clu==cluID]
    track must be the whl file in normal sampling i.e. 39.0625Hz
    valid samples are not necessary but note that this returns all
    spikes in the session level file for a given clu
    '''
    spks2tracking = (1./512)
    
    nEdges = nbins+1
    if mazedim is None:
    	Xedges = np.linspace(np.min(track['x']),np.max(track['x']),nEdges)
    	Yedges = np.linspace(np.min(track['y']),np.max(track['y']),nEdges)
    else:
        Xedges = np.linspace(mazedim[0],mazedim[1],nEdges)
        Yedges = np.linspace(mazedim[2],mazedim[3],nEdges)
    if debug:    
        print(Xedges[0],Xedges[-1])
        print(Yedges[0],Yedges[-1])

    spikesPos = (np.round(spikes*spks2tracking)).astype(int)
    spikesPos = [x for x in spikesPos if x < len(track['y'])]

    y = np.copy(track['y'])[spikesPos]
    y = y[~np.isnan(y)]
    x = np.copy(track['x'])[spikesPos]
    x = x[~np.isnan(x)]
    nSpikesPerSpace,_,_,_ = hist2(x,y,[Xedges,Yedges])
    
    return nSpikesPerSpace
#######################################################################################
def shuffle_spkMap(spkMap):
    '''
    Used to generate null distributions for spatial info and coherence
    '''
    matsize = int(spkMap.shape[0]*spkMap.shape[1])
    tempdat = spkMap.reshape(matsize,)
    
    return np.random.permutation(tempdat).reshape(spkMap.shape)
#######################################################################################
def calc_spatial_info_all_sess(allOcc,allPlace,sessions):
    '''
    Inputs:
        Takes allOccMaps, allPlaceMaps, and list of sessions
        to feed into calc_spatial_info function below
    Outputs:
        Dictionaries of total information, information / Spike, 
        place map coherence for all sessions
    '''
    totalInfo = {}
    InfoPerSpike = {}
    Coh = {}
    for indx,sess in enumerate(sessions):
        totalInfo[sess], InfoPerSpike[sess], Coh[sess], zCoh[sess] = calc_spatial_info(allOcc[sess],allPlace[sess])
        
    return totalInfo,InfoPerSpike,Coh,zCoh
###################################################################################################
def calc_peak_bin(OccMap,SpkMap,smooth=1.2,debug=False):
    '''
    Inputs: unsmoothed OccMap and SpkMap (not masked)
    Outputs: the peak firing rate in a given pixel, the pixel location in cartesian.
    '''

    # Generate masked 2d arrays for Occ and Spk
    value = 0
    OccMap = np.ma.masked_where(OccMap == value, OccMap)
    SpkMap = np.ma.masked_where(OccMap == value, SpkMap)
    # Smooth these maps with Gaussian Filter
    smOcc = spim.filters.gaussian_filter(OccMap,smooth,mode='constant')
    smSpk = spim.filters.gaussian_filter(SpkMap,smooth,mode='constant')
    # Generate a smoothed placeMap from these data, and re-mask
    smPlaceMap = smSpk / smOcc
    masked_data = np.ma.masked_where(OccMap == value, smPlaceMap)
    # find max FR pixel and its location
    peakRate = np.nanmax(masked_data)
    peakLoc = np.unravel_index(np.nanargmax(masked_data),masked_data.shape)

    if debug:
        print('Peak rate: ', peakRate)
        print('peakLoc: ', peakLoc)
        print(placeMap[peakLoc[0],peakLoc[1]])

    return peakRate,peakLoc
###################################################################################################
def calc_spatial_info(OccMap,placeMap):
    '''
    Inputs:
        Takes the OccMap (in Hz) and the placeMap (also in Hz)
        and calculates standard information measures
        ...and the spatial coherence
    Outputs:
        total information, information / Spike, place map coherence
    '''
    ##############################################################################################
    mask = (OccMap) > 0
    #print 'mask contains %d values' % (len(mask))
    meanRate = np.mean(placeMap[mask])
    OccMapProb = OccMap / np.sum(OccMap[mask])
    ##############################################################################################
    information = 0
    auxCoh = np.array([]).reshape(0,2)
    ##############################################################################################
    for bini in range(np.size(OccMapProb,0)):
        for binj in range(np.size(OccMapProb,1)):
            if (mask[bini,binj]) & (placeMap[bini,binj] > 0):
                ## information = information + [FReachbin * (log2(FReachbin / meanRate)) * ProbOccEachbin]
                information += placeMap[bini,binj] *\
                                np.log2(placeMap[bini,binj] / meanRate) * OccMapProb[bini,binj]
                try:
                    ## calc the mean number of spikes in the 8 surrounding pixels(bins)
                    ## how does this deal with edges?
                    ## could reorder this so more logical, i.e. clockwise from 12 o'clock
                    aux1 = np.nanmean(np.array([placeMap[bini,binj+1],placeMap[bini+1,binj],\
                                         placeMap[bini+1,binj+1],placeMap[bini,binj-1],\
                                         placeMap[bini-1,binj],placeMap[bini-1,binj-1],\
                                         placeMap[bini+1,binj-1],placeMap[bini-1,binj+1]]))
                    ## then, if the result is a number (i.e. is finite) 
                    ## paste the cells FR and the surround FR into a 2 col array
                    if np.isfinite(aux1):
                        #print aux1
                        auxCoh = np.vstack((auxCoh,np.array([placeMap[bini,binj],aux1]).T))
                except:
                    1 # not sure what this exception is for!
    ##############################################################################################
    placeMapInfoPerSpike = information / meanRate
    try:
        placeMapCoh = stats.pearsonr(auxCoh[:,0],auxCoh[:,1])[0]
        zplaceMapCoh = ( placeMapCoh - np.nanmean(placeMap,axis=(0,1)) ) / np.nanstd(placeMap,axis=(0,1))
    except:
        placeMapCoh = np.nan
        zplaceMapCoh = np.nan
    ##############################################################################################
    return information, placeMapInfoPerSpike, placeMapCoh, zplaceMapCoh
##################################################################################################
'''
from gpgFunc3.py:

    meanRate = np.mean(placemap[mask])
    OccMapProb = OccMap / np.sum(OccMap[mask])
    information = 0
    spN = 0; spD = 0
    auxCoh = np.array([]).reshape(0, 2)
    for bini in range(np.size(OccMapProb, 0)):
        for binj in range(np.size(OccMapProb, 1)):
            if (mask[bini, binj]) & (placemap[bini, binj] > 0):
                information += placemap[bini, binj] *\
                np.log2(placemap[bini, binj] / meanRate) * OccMapProb[bini, binj]
'''
########################################################################################
def stack_1d_array(IFRList,sess,cell_inds,mouseID,col_ind=3,thresh=100.0,mask=False):
    '''
	a generic version of the function above 
	you pass a col_index
	e.g. for Spatial_info data, the indices are:
	0: cluID, 1: totalSpikes, 2: meanRate, 3: totalInfo
	4: infoPerSpike, 5: spatial coherence
    '''
    mouse = 0
    odata = IFRList[sess][mouse][cell_inds[mouse],col_ind]
    for mouse in range(1,len(mouseID)):
        array_to_add = IFRList[sess][mouse][cell_inds[mouse],col_ind]
        odata = np.hstack((odata,array_to_add))
    if mask:
        #odata = np.ma.masked_where(odata > thresh, odata)
        odata = gt_to_nan(zero_to_nan(odata),thresh)
    return odata
##################################################################################
def stack_2d_array(IFRList,sess,cell_inds,mouseID,col_range):
    '''
    '''
    mouse = 0
    odata = IFRList[sess][mouse][cell_inds[mouse],col_range[0]:col_range[1]]
    ####################################################################################
    for mouse in range(1,len(mouseID)):
        array_to_add = IFRList[sess][mouse][cell_inds[mouse],col_range[0]:col_range[1]]
        odata = np.concatenate((odata,array_to_add),axis=0)   
    return odata
##################################################################################
def stack_vector(IFRList,sess,cell_inds,mouseID,mask=False,thresh=100.0):
    '''
    '''
    mouse = 0
    odata = IFRList[sess][mouse][cell_inds[mouse]]
    for mouse in range(1,len(mouseID)):
        array_to_add = IFRList[sess][mouse][cell_inds[mouse]]
        odata = np.hstack((odata,array_to_add))
    if mask:
        odata = gt_to_nan(zero_to_nan(odata),thresh)
    return odata
##################################################################################

##################################################################################
def calc_mean_std_sem(idata,sessionList,validlt=False,value=100,zeroToNan=False):
    '''
    Function to calculate means,stds, and sems from dict
    idata is the name of the dict and sessionList
    are strings for the keys (fields)
    Outputs to a list (i.e. 1D vector of floats)
    '''
    oMean = []
    oStd = []
    oSem = [] 
    for nn,sess in enumerate(sessionList):
        tempDat = idata[sess]
        if zeroToNan:
            tempDat = np.array(zero_to_nan(tempDat))
        if validlt:
            tempDat = np.array(get_valid_lt(tempDat,value))
        oMean.append(np.nanmean(tempDat))
        oStd.append(np.nanstd(tempDat))
        oSem.append(np.nanstd(tempDat) / np.sqrt(tempDat.shape[0]))               
    return oMean, oStd, oSem
##########################################################################################################################
def create_place_maps(mouse,cluID_list,alldesen,allBaseblock,units,intv_type='.col2resofs',
                      nbins=35,smooth=1.2,sessMax=None,figsize=[4,4],cmax=0.8,mazedim=None,figOnly=True,colbar=False):
    '''
    
    '''
    ################################################################
    framesPerSec = 39.0625
    spk_sr = 20000
    tconv = (framesPerSec / spk_sr) # ratio for fps to spike sampling
    mazedim = None
    #################################################################################################
    desen,baseblock,ipath,bsnm = get_mouse_info(alldesen,allBaseblock,mouse)
    os.chdir(ipath)
    ##################################################################################################
    allSessPlaceMaps = {}
    allSessOccMaps = {}
    allSessSpkMaps = {}
    rate = {}
    #################################################################################################
    fig = {}
    ##################################################################################################
    for cindx,val in enumerate(cluID_list):
        sessions = get_sessions(alldesen[mouse],sleepbox=False)
        sessions = sessions[:sessMax]
        print(val,'\t','\t','\t','mFR','\t','MaxFR','\t','I/sp','\t','Coh')
        for sindx,sess in enumerate(sessions):
            ### Select session
            sessionLabel = get_descode(desen,sess)
            fname = baseblock + '_' + str(sessionLabel['filebase'].index[0])
            #print(fname)
            ### Load tracking
            track = vbf.load_tracking(fname,smoothing=1)
            sessDur = get_session_duration(desen,sess)
            ###########################################################################################
            intervalTimes = get_pulsetimes(ipath,desen,sess,ext=intv_type,tconv=tconv,debug=False)
            trkIntervalTimes = gen_valid_times(intervalTimes)
            ### Generate occupancy map
            OccMap = genOccMap(track,trkIntervalTimes,nbins=nbins,mazedim=mazedim)
            allSessOccMaps[sess] = (OccMap / framesPerSec) # convert to Hz
            ### Generate spike map
            ############################################################################################
            cluID = val
            res,clu = vbf.LoadSpikeTimes(fname,trode=None,MinCluId=2,res2eeg=(20000./20000))
            spikes = res[clu==cluID]
            rate[sess] = np.around((len(spikes) / sessDur),2)
            allSessSpkMaps[sess] = genSpikeMap(track,spikes,nbins=nbins,mazedim=mazedim)
            ### Generate placemap by diving spike map by occupancy map
            allSessPlaceMaps[sess] = (allSessSpkMaps[sess] / allSessOccMaps[sess])
            ### Calc spatial info
            info,infoPerSpike,coh,zcoh = calc_spatial_info(allSessOccMaps[sess],allSessPlaceMaps[sess])
            peakRate,peakLoc = calc_peak_bin(allSessOccMaps[sess],allSessSpkMaps[sess],smooth=1.2,debug=False)
            print(sess[:6],'\t','\t','\t',np.round(rate[sess],2),'\t',np.round(peakRate,2),'\t',np.round(infoPerSpike,2),'\t',np.round(coh,2),'\t',np.round(zcoh,2))
        print()
        #############################################################################################################
        try:
            fig[val],ax = plot_placemaps_one_cell(allSessOccMaps,allSessSpkMaps,allSessPlaceMaps,
                                sessions,bsnm,units[mouse],cluID,nbins,smooth=smooth,cmax=cmax,size=figsize,axoff=True,colbar=colbar)
        except TypeError:
            fig[val],ax = plot_placemaps_one_sess(OccMap,allSessSpkMaps[sess],allSessPlaceMaps[sess],
                                                  sess=sess,nbins=nbins,smooth=smooth,cmax=cmax,size=figsize,axoff=True,colbar=colbar)
    if figOnly:
        return fig,bsnm
    else:
        return allSessOccMaps,allSessPlaceMaps,allSessPlaceMaps
    #################################################################################################################
#####################################################################################################################
def plot_placemaps_one_cell(allOcc,allSpk,allPlace,sessions,bsnm,units,cluID,
				nbins=40,smooth=None,cmax=.5,size=[4,4],axoff=True,showX=False,colbar=False):
    '''
    Inputs: OccMap,
    
    Output: fig
    '''
    nSess = len(sessions)
    rr,cc = 1,nSess
    wcm = size[0] * nSess # should be 2
    hcm = size[1] # should be 2 
    fig, ax = plt.subplots(rr,cc,figsize = cm2inch(wcm,hcm),
                           gridspec_kw = {'wspace':0,'hspace':0},
                           )
    value = 0
    cmap = plt.get_cmap('jet')
    cmap.set_bad(color='white')
    fontdict = get_fontdict()
    ###########################################################################################
    for indx,sess in enumerate(sessions):
        # find peak rate first - it's smoothed in the function
        peakRate,peakLoc = calc_peak_bin(allOcc[sess],allSpk[sess],smooth=smooth,debug=False)

        if smooth is not None:
            smOcc = spim.filters.gaussian_filter(allOcc[sess],smooth,mode='constant')
            smSpk = spim.filters.gaussian_filter(allSpk[sess],smooth,mode='constant')
            smPlaceMap = smSpk / smOcc
            masked_data = np.ma.masked_where(allOcc[sess] == value, smPlaceMap)
        else:
            masked_data = np.ma.masked_where(allOcc[sess] == value, allPlace[sess])

        hmin = np.nanmin(masked_data)
        hmax = cmax * np.nanmax(masked_data)

        if nSess == 1:
            tempax = ax
        else:
            tempax= ax[indx]

        im = tempax.imshow(masked_data.T,clim=(hmin,hmax),cmap=cmap,interpolation='Nearest')
        tempax.set_xlim(0,nbins)
        tempax.set_ylim(nbins,0)

        if axoff:
            tempax.set_axis_off()

        tempax.set_title(sess,y=-.3,fontsize=8)
        tempax.text(.35, 1.05, str(np.around(peakRate,1)),
                      verticalalignment='bottom',horizontalalignment='left',
                      transform=tempax.transAxes,fontdict=fontdict)
        if showX:
            tempax.text(peakLoc[0],peakLoc[1],'X', verticalalignment='center',horizontalalignment='center',fontdict=fontdict)
        
    if colbar:
        divider = make_axes_locatable(tempax)
        cax = divider.append_axes("right", size="10%", pad="5%")
        cb1 = fig.colorbar(im,cax=cax,orientation="vertical")

    ftitle = bsnm + '_' + str(units['des'][cluID]) + '_' + str(cluID)    
    fig.suptitle(ftitle, y=-.1,fontsize=10)
    #############################################################################################
    return fig,ax
#################################################################################################
def cartesian_dist(xx,yy,smoothing=1):
    ''' 
        calc the cartesian distance and return
        dist = sqrt ( (x2-x1)^2 + (y2-y1)^2 )
    '''
    xxdiff = np.diff(xx)
    yydiff = np.diff(yy)
    dist = np.zeros((xxdiff.shape))
    for nn in range(0,len(xxdiff)):
        dist[nn] = np.sqrt((xxdiff[nn]**2 + yydiff[nn]**2))
    ##########################################################################
    if smoothing is not None:
        dist = scipy.ndimage.filters.gaussian_filter1d(dist,smoothing,axis=0)

    sumDist = np.nansum(dist,axis=0)
    return dist,sumDist
######################################################################################
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
######################################################################################
def calc_mean_sem(idata,sessionList,validlt=False,value=100,zeroToNan=False):
    '''
    Function to calculate means and sems from dict
    idata is the name of the dict and sessionList
    are strings for the keys (fields)
    Outputs to a list (i.e. 1D vector of floats)
    '''
    oMean = []
    oSem = [] 
    for nn,sess in enumerate(sessionList):
        tempDat = idata[sess]
        if zeroToNan:
            tempDat = np.array(zero_to_nan(tempDat))
        if validlt:
            tempDat = np.array(get_valid_lt(tempDat,value))
        oMean.append(np.nanmean(tempDat))
        oSem.append(np.nanstd(tempDat) / np.sqrt(tempDat.shape[0]))               
    return oMean,oSem
######################################################################################
def get_isi(spikes,tconv=(20000/1000)):
    '''
    converst the isi to ms so it can be passed to get_burst_index
    assumes sampling rate is 20kHz, so ms is this divided by 1000
    '''
    return np.diff(spikes) / tconv
######################################################################################
def get_burst_index(isi,burstwin=6):
    return len(isi[isi < burstwin]) / len(isi)
######################################################################################
#def calc_burst(isi, clui, burstwin=6, ubspks=150):
#    return (100 * np.sum(isi[clui] < burstwin) / np.sum(isi[clui] < ubspks))
######################################################################################
def calc_spatial_sparsity(OccMap,placeMap):
    '''
    Inputs:
        Takes the OccMap (in Hz) and the placeMap (also in Hz)
        and calculates sparsity
    Outputs:
        sparsity
    '''
    ##############################################################################################
    mask = (OccMap) > 0
    #meanRate = np.mean(placeMap[mask])
    OccMapProb = OccMap / np.sum(OccMap[mask])
    ##############################################################################################
    spN = 0; spD = 0
    ##############################################################################################
    for bini in range(np.size(OccMapProb,0)):
        for binj in range(np.size(OccMapProb,1)):
            if (mask[bini,binj]) & (placeMap[bini,binj] > 0):
                
                spN += OccMapProb[bini, binj] * placeMap[bini, binj]
                spD += OccMapProb[bini, binj] * placeMap[bini, binj]**2
           
    ##############################################################################################
    try: sparsity = spN**2 / spD
    except: sparsity = np.nan
    ##############################################################################################
    return sparsity
###############################################################################################
def gen_hist2d(xdat,ydat,xedg,yedg,gdim):
    '''
    '''
    H, xe, ye = np.histogram2d(xdat, ydat, bins=(xedg, yedg))
    H = H.T
    mid = int((gdim - 1) / 2)
    edge = int(gdim-1)
    print(mid,edge)
    nesw_cnt = [H[0,mid],H[mid,edge],H[edge,mid],H[mid,0]] # note: NESW order

    return H, nesw_cnt
####################################################################################################
def get_nesw(track,mazedim,trialT=300,grid=[3,3],gridoffset=[5,5],fps=39.025,debug=False):
    '''
    
    '''
    
    objPos = get_objPos_new(mazedim) # import object locations
    
    x_divs,y_divs = grid[0],grid[1] # divide arena into e.g. 3 x 3 grid

    gridxOffset,gridyOffset = gridoffset[0],gridoffset[1] # add a few pixels to the edges

    # g stands for grid
    gxmin = objPos['w'][0]-gridxOffset
    gxmax = objPos['e'][0]+gridxOffset
    gymin = objPos['n'][1]-gridyOffset
    gymax = objPos['s'][1]+gridyOffset

    pix_x = (gxmax - gxmin)/(x_divs*1.0)
    pix_y = (gymax - gymin)/(y_divs*1.0)

    xedges = [gxmin]
    yedges = [gymin]

    for x in range(x_divs):
        xedges.append(np.round(gxmin+(x+1)*pix_x))  #gxmin+(x+1)*pix_x   
    for y in range(y_divs):
        yedges.append(np.round(gymin+(y+1)*pix_y))  #gxmin+(x+1)*pix_x    

    tind = int(trialT * fps)
    xvalid = track['x'].values[:tind]
    yvalid = track['y'].values[:tind]
    
    hist_all,hist_nesw = gen_hist2d(xvalid,yvalid,xedges,yedges,gdim=grid[0])
    
    return hist_all,hist_nesw
####################################################################################################
def myround(x, base=100):
    return int(base * np.floor(x/base))
###################################################################
def zero_to_nan(idata):
    odata=np.array(idata).astype('float')
    odata[odata==0] = np.nan
    return odata
###################################################################
def gt_to_nan(idata,val):
    odata=np.array(idata).astype('float')
    odata[odata > val] = np.nan
    return odata
###################################################################
def lt_to_nan(idata,val):
    odata=np.array(idata).astype('float')
    odata[odata < val] = np.nan
    return odata
###################################################################
def inf_to_nan(idata):
    odata=np.array(idata).astype('float')
    odata[odata==np.inf] = np.nan
    odata[odata==-np.inf] = np.nan

    return odata
#################################################################
def nan_to_zero(idata):
    odata=np.array(idata).astype('float')
    odata[odata==np.nan] = 0
    odata[odata=='nan'] = 0
    return odata
#################################################################
def isvalid(number):
    if number is None or np.isnan(number):
        return False
    else:
        return True
######################################################################################
def remove_nan(y):
    return y[~np.isnan(y)]
######################################################################################
def npairs(n):
    '''
    Returns the number of pairwise comparisons from a list of length n
    '''
    return int((n*(n-1)) / 2)
######################################################################################
def calc_percentiles(idata,pc=[25,75]):
    '''
    Takes a vector and calculates two percentiles (e.g. 25th, 75th)
    Returns 3 values in odata, e.g. Q1: [0], Q3: [1] and Q3-Q1: [2] 
    '''
    odata = []
    for qq in pc:
        odata.append(np.nanpercentile(idata, qq, interpolation = 'midpoint'))

    odata.append(odata[1]-odata[0])

    return odata
######################################################################################
def list_duplicates(seq):
    '''
    Returns a list of duplicates from a list, seq
    '''
    seen = set()
    seen_add = seen.add
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = set( x for x in seq if x in seen or seen_add(x) )
    # turn the set into a list (as requested)
    return list( seen_twice )
######################################################################################
def reverse_dict(iDict):
    '''
    '''
    from collections import defaultdict

    flipped = defaultdict(dict)
    for key, val in iDict.items():
        for subkey, subval in val.items():
            flipped[subkey][key] = subval
        
    return flipped
############################################################################
def keys_to_list(iDict):
    '''
    '''
    olist = []
    for key,val in iDict.items():
        olist.append(key)
    return olist
###############################################################################
def get_unique_keys_by_level(d, level=0, keys_by_level=None):
    '''
    
    '''
    if keys_by_level is None:
        keys_by_level = {}

    if not isinstance(d, dict) or not d:
        return keys_by_level

    # Initialize the set for the current level if not already present
    if level not in keys_by_level:
        keys_by_level[level] = set()

    # Add current level keys
    keys_by_level[level].update(d.keys())

    # Recurse into the dictionary
    for value in d.values():
        if isinstance(value, dict):
            get_unique_keys_by_level(value, level + 1, keys_by_level)

    return keys_by_level
###################################################################################
def dict_var3_to_var2(iDict,ikey_list,okey_list,fixed_key='p1',fixed_key_level=1):
    '''

    '''
    output_dict = dict_from_list(ikey_list)

    for ikey_indx,ikey in enumerate(ikey_list):

        temp_dict = dict_from_list(okey_list)

        for okey_indx,okey in enumerate(okey_list):
            if fixed_key_level == 0:
                tempdat = iDict[fixed_key][ikey_key][okey]
            elif fixed_key_level == 1:
                tempdat = iDict[ikey][fixed_key][okey]
            elif fixed_key_level == 2:
                tempdat = iDict[ikey][okey][fixed_key]

            temp_dict[okey] = tempdat

        output_dict[ikey] = temp_dict

    return output_dict
##################################################################################
def double_to_single_key(iDict):
    '''
    
    '''
    oDict = {}
    for key1,val1 in iDict.items():
        for key2,val2 in iDict[key1].items():
            okey = '_'.join([key1,key2])
            oDict[okey] = iDict[key1][key2]
            
    return oDict
#############################################################################
def dict_from_list(key_list,two_var=True):
    '''
    Create two var dict if two_var = True
    or create dictionary of lists if two_var=False
    '''
    if two_var:
        return {key: {} for key in key_list}
    else:
        return {key: [] for key in key_list}
############################################################################
def dataMetric_ctype(iDict,ikeys,okeys,method=np.nanmedian,round_df=3):
    '''
    Generates mean,median,var,etc.
    '''
    odata = {}
    for ikindx,ikey in enumerate(ikeys):
        tempDat = {}
        for okindx,okey in enumerate(okeys):
            idata = method(iDict[ikey][okey])
            print(okey,ikey,np.around(idata,round_df))
            tempDat[okey] = idata
        odata[ikey] = tempDat
    return odata
############################################################################
def percentile_ctype(iDict,ikeys,okeys,lb=25,ub=75,round_df=3):
    '''
    Generates mean,median,var,etc.
    '''
    odata = {}
    for ikindx,ikey in enumerate(ikeys):
        tempDat = {}
        for okindx,okey in enumerate(okeys):
            idata = np.percentile(iDict[ikey][okey],[lb,ub])
            print(okey,ikey,np.around(idata,round_df))
            tempDat[okey] = idata
        odata[ikey] = tempDat
    return odata


def remove_nan_two_lists(idata1,idata2):
    '''
    
    '''
    invalid_ind = np.logical_or(np.isnan(idata1), np.isnan(idata2))

    x = idata1[~invalid_ind]
    y = idata2[~invalid_ind]

    return x,y
######################################################################
def set_axis_bounds(ax,xlim,ylim,sp_len=3):
    ''' 
    set axis bounds
    '''
    ax.spines['left'].set_position(('outward', sp_len))
    ax.spines['bottom'].set_position(('outward', sp_len))
    ax.spines['left'].set_bounds((ylim[0],ylim[1]))
    ax.spines['bottom'].set_bounds((xlim[0],xlim[1]))

    return ax
######################################################################
def plot_ngroups_bar(idata,isem,iparams):
    '''
    Think about what can be parameterized and how these should be passed
    e.g. as pd dataframe or just iparams dict?
    '''
    ###########################################################
    from matplotlib.ticker import FormatStrFormatter

    try:
        sns.reset_orig()
    except:
        print('Seaborn not imported')
    ##
    if len(idata[0]) > 3:
        barspace = 2
    else:
        barspace = 1		   
    ind = np.arange(0,barspace*len(idata[0]),barspace)  # the x locations for the groups
    ############################################################
    if iparams['width'] is not None:
        width = iparams['width']
    else:
        width = 0.4 # the width of the bars
    ############################################################
    offset = [0, width] 
    alpha = 1
    area = 5
    ############################################################
    if iparams['figsize'] is not None:
        fwid,fht = iparams['figsize'][0],iparams['figsize'][1]
    else:    
        fwid,fht = 6,4
    ############################################################
    if iparams['lw'] is not None:
        lw = iparams['lw']
    else:
        lw = 0.5 # line width
    ############################################################
    if iparams['capsize'] is not None:
        capsize = iparams['capsize']
    else:
        capsize = 1.5
    ######################################
    if iparams['cols'] is not None:
        col = iparams['cols']
    else:
        col = ['white','k','r']
    ######################################
    if iparams['ec'] is not None:
        ec = iparams['ec']
    else:
        ec = [*np.repeat('k',20)]
    ######################################
    if iparams['xlabs'] is not None:
        xlabs = iparams['xlabs']
    else:
        xlabs = ['Fam','CS-','CS+','CSa']
    ####################################### 
    if iparams['ylab'] is not None:
        ylab = iparams['ylab']
    else:
        ylab = ['Classifier accuracy']
    ########################################   
    if iparams['chanceLine'] is not None:
        chanceLine = iparams['chanceLine']
    else:
        chanceLine = (1./len(xlabs)) * 100
    ###########################################################################################
    fig, ax = plt.subplots(figsize=cm2inch(fwid,fht))
    ###########################################################################################
    for nn in range(0,len(idata)):
        xpts = ind+(width*nn)
        rects = ax.bar(xpts,idata[nn],width,linewidth=lw,color=col[nn],edgecolor=ec[nn],alpha=alpha,
                    yerr=isem[nn],error_kw=dict(lw=lw, capsize=capsize, capthick=lw),
                    )
    ###########################################################################################
    # y-axis and yticks
    if iparams['yrange'] is not None:
        ymin,ymax = iparams['yrange'][0],iparams['yrange'][1]
        ytick = iparams['yrange'][2]
    else:
        ymin,ymax = 0,100
        ytick = 20
    if iparams['yfmat'] is not None:
        yfmat = iparams['yfmat']
    else:
        yfmat = '%.2f'
    ###########################################################################################
    ax.set_ylabel(ylab)
    ax.set_ylim(ymin,ymax)
    # yticks
    yticks = np.arange(ymin,ymax+0.01,ytick)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, rotation=0, fontsize=iparams['fontsize'])
    ax.yaxis.set_major_formatter(FormatStrFormatter(yfmat))
    # xticks
    if len(idata) == 1:
        ax.set_xticks(np.arange(0,len(idata[0]),1))
    else:
        ax.set_xticks(ind + width / 2)
    ax.set_xticklabels((xlabs),fontsize=iparams['fontsize'])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('none')
    # Show legends
    #ax.legend((rects), ('DG','CA3','CA1'),\
     #         frameon=False,bbox_to_anchor=(1, 1))
    # Plot chance line
    ax.axhline(chanceLine,color='k',linestyle='--',linewidth=lw)   
    ###########################################################################################
    return fig,ax
################################################################################################
def genEdges2d(mazedim,nbins=3):
    '''
    
    '''
    nEdges = nbins+1
    Xedges = [int(x) for x in np.linspace(mazedim[0],mazedim[1],nEdges)]
    Yedges = [int(x) for x in np.linspace(mazedim[2],mazedim[3],nEdges)]

    return Xedges,Yedges
################################################################################################
def diff_dates(date1, date2):
    return abs(date2-date1).days
################################################################################################
def days_since_injection(bsnm,mpath='/mnt/smchugh2/',ifile='injectdate.db',ipath='/lfpd4/databases'):
    '''
    
    '''
    targ_id = bsnm.rsplit('-',1)[0]
    targ_date = datetime.datetime.strptime(bsnm.rsplit('-',1)[-1], '%y%m%d')

    injectdate = read_db(ifile,ipath,mpath=mpath,omit=True,printpath=False)
    
    for indx,val in enumerate(injectdate):
        inject_id = injectdate[indx].rsplit('-',1)[0]
        if inject_id == targ_id:
            ref_date = datetime.datetime.strptime(injectdate[indx].rsplit('-',1)[1],'%y%m%d')
            odata = diff_dates(ref_date,targ_date)-1
    try:
        return odata
    except:
        print('Basename not in database')
        return
###################################################################################
def plot_speed_hist(iDict,iKeys,nbins=40,wcm=4,hcm=4):
    '''
    
    '''
    rr,cc = 1,len(iKeys)
    wcm,hcm = wcm*cc,hcm*rr
    fig,ax = plt.subplots(rr,cc,figsize=cm2inch(wcm,hcm),
                         gridspec_kw = {'wspace':0,'hspace':0},
			 sharex=True
                         )
    for sindx,sess in enumerate(iKeys):
        idata = iDict[sess]
        idata = idata[np.isfinite(idata)]
        xedges = np.linspace(0,nbins,nbins+1)
        histdat = np.histogram(idata,xedges)
        meanS = np.nanmean(idata)
        ax[sindx].bar(histdat[1][:-1],histdat[0])
        # Hide the right and top spines
        ax[sindx].spines['right'].set_visible(False)
        ax[sindx].spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax[sindx].yaxis.set_ticks_position('left')
        ax[sindx].xaxis.set_ticks_position('none')
        if sindx > 0:
            ax[sindx].get_yaxis().set_visible(False)
            ax[sindx].spines['left'].set_visible(False)
        # Print title and avxline
        subtitle = sess + ': ' + str(np.round(meanS,1))
        ax[sindx].set_title(subtitle,y=1,fontsize=8)
        ax[sindx].axvline(meanS,color='red',linestyle='--',linewidth=.5)

    return fig,ax
####################################################################################
def fisherZtest(r1,r2,n1,n2):
    '''
    calc z1 and z2 based on r1 and r2
    calc sigma(z1 - z2)
    then the Z output value = (z2 - z1) / sigma(z1 - z2)
    Note that +ve Z values indicate higher corr for r2 over r1!!
    '''      
    
    z1 = 0.5 * np.log((1+r1) / (1-r1))
    z2 = 0.5 * np.log((1+r2) / (1-r2))
    
    zsigma = np.sqrt( (1.0/(n1-3)) + (1.0/(n2-3)) )
    
    zout = (z2 - z1) / zsigma
    pout = stats.norm.sf(abs(zout)) * 2

    return zout,pout
#######################################################################################
def simple_corr(xx,yy,sess='',xlab='',ylab='',corr_type='pearson',pplot=True,
                xlim=[30,50],figsize=[8,8],mcol='b',msize=2,alpha=1,fsize=10,fit=False,title=True):
    '''
    fig,ax,[rval,pval] = sbf.simple_corr(xx,yy,sess='',xlab='',ylab='',corr_type='pearson',pplot=True,
                xlim=[30,50],figsize=[8,8],mcol='b',msize=2,alpha=1,fsize=10,fit=False,title=True)
    '''
    bad = ~np.logical_or(np.isnan(xx), np.isnan(yy))

    xx = np.compress(bad, xx)
    yy = np.compress(bad, yy)
    
    if corr_type == 'pearson' or corr_type == 'Pearson':
        rval,pval = stats.pearsonr(xx,yy)
    else:
        rval,pval = stats.spearmanr(xx,yy)
    
    if pplot:
        rr,cc = 1,1
        fig,ax = plt.subplots(rr,cc,figsize=cm2inch(figsize[0],figsize[1]))
	
        ax.scatter(xx,yy,color=mcol,s=msize,alpha=alpha)
        ax.set_xlim(xlim)
        if fit:
            pfit = np.poly1d(np.polyfit(xx, yy, 1))
            xpts = np.arange(xlim[0]-5,xlim[1]+5)
            ax.plot(xpts,pfit(xpts),'-b',linestyle = '--',label='fit')

        ax.set_ylabel(ylab,fontsize=fsize)
        ax.set_xlabel(xlab,fontsize=fsize)
        if title:
            ax.set_title('{0:}: r = {1:.2f}, p = {2:.3f}'.format(sess,rval,pval),fontsize=fsize)
    
        return fig,ax,[rval,pval]

    else:
        return [rval,pval]
###########################################################################################
def plot_fit(ax,xx,yy,xlim,lcol='b',lw=2,pf=1):
    '''
    
    '''
    pfit = np.poly1d(np.polyfit(xx, yy, pf))
    xpts = np.arange(xlim[0],xlim[1])
    ax.plot(xpts,pfit(xpts),color=lcol,linestyle='--',linewidth=lw,label='fit')
    
    return ax
###########################################################################################
def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of
    variables in C, controlling for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    import scipy.linalg as linalg
    #################################################################################
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    #################################################################################
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):            
            # Create matrix of indices
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            # Computes beta values using least squares
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]
            # Computes residuals
            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)            
            # Compute correlation between residuals to get partial correlation
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr # P_corr is the full correlation matrix with 1 on diagonal
################################################################### 
def square_rooted(x):
    return np.sqrt(sum([a*a for a in x]))
################################################################### 
def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    try:
        odata = round(numerator/float(denominator),3)
    except ZeroDivisionError:
        odata = np.nan
    return odata
#######################################################################################
def smc_ttest(idata,cond_list=['cond1','cond2'],pop_mean=0,paired=False,pprint=True):
    '''
    
    '''
    if len(idata) < 2:
        tval,pval = stats.mstats.ttest_1samp(idata[0],pop_mean)
        df = int(len(idata[0]) - 1)
        print('{0:} xbar={1:.2f}'.format(cond_list[0],np.nanmean(idata[0])))
        print('t({0:})={1:.2f}, p={2:.3f}'.format(df,tval,pval))
        pprint = False
    if paired:
        tval,pval = stats.mstats.ttest_rel(idata[0],idata[1])
        df = int(len(idata[0]) - 1)
    else:
        tval,pval = stats.mstats.ttest_ind(idata[0],idata[1])
        df = int(len(idata[0])+len(idata[1]) - 2)
    if pprint:
        print('{0:} xbar={1:.3f}, {2:} xbar={3:.3f}'.format(cond_list[0],
                                                            np.nanmean(idata[0]),
                                                            cond_list[1],
                                                            np.nanmean(idata[1])))
        print('t({0:})={1:.2f}, p={2:.3f}'.format(df,tval,pval))

    return tval,pval
#####################################################################################
def ztest_two_prop(f1,f2,n1,n2):
    '''
    
    '''
    p1 = f1 / n1
    p2 = f2 / n2
    phat = (f1+f2) / (n1+n2)
    print('p1={0:.2f} vs p2={1:.2f}, pooled prop={2:.2f}'.format(p1,p2,phat))
    
    denom = np.sqrt( phat*(1-phat) * (1/n1 + 1/n2) )
    zval = (p1 - p2) / denom
    pval = stats.norm.sf(abs(zval))*2
    print('z={0:.1f}; p={1:.2f}'.format(zval,pval))

    return zval,pval
##########################################################################################
def stdErr(idata):
    return np.nanstd(idata)/np.sqrt(len(idata)) #This is also in smCofiringFunctions.py
##########################################################################################
def cohensd(d1,d2):

    # calculate the size of samples
    n1, n2 = np.array(d1).shape[0], np.array(d2).shape[0]

    # calculate the variance of the samples
    s1, s2 = np.nanvar(d1, ddof=1), np.nanvar(d2, ddof=1)

    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    # calculate the means of the samples
    u1, u2 = np.nanmean(d1), np.nanmean(d2)

    # return the effect size
    return (u1 - u2) / s
##########################################################################################
def calc_err_bars(idata,axis=0,savgol=False,npts=3,n_samples=None,pprint=False):
    '''
    '''
    if n_samples is None:
        n_samples = idata.shape[axis]
    if pprint:
        print('there are', n_samples,'samples')
    mean_ = np.nanmean(idata,axis=axis)
    sem_ = np.nanstd(idata,axis=axis) / np.sqrt(n_samples)
    pos_err = mean_ + sem_
    neg_err = mean_ - sem_

    if savgol:
        pos_err = savgol_filter(pos_err,npts,1)
        neg_err = savgol_filter(neg_err,npts,1)
    
    return pos_err,neg_err
##########################################################################################
def plot_err_bars(ax,idata,nms,axis=0,lw=0.5,color='b',savgol=False,npts=3):
    '''
    
    '''
    pos_err,neg_err = calc_err_bars(idata,axis=axis,savgol=savgol,npts=npts)
    
    xpts = np.linspace(-nms[0],nms[1],pos_err.shape[0])
    ax.plot(xpts,pos_err,color=color,linewidth=lw,linestyle='--')
    ax.plot(xpts,neg_err,color=color,linewidth=lw,linestyle='--')
    
    return ax
######################################################################################################## 
def create_place_maps_one_mouse(mouse,cluID_list,alldesen,allBaseblock,sessions,intv_type='.col2resofs',
                      nbins=35,sessMax=None,figsize=[4,4],mazedim=None,smooth=1.2):
    '''
    
    '''
    ################################################################
    framesPerSec = 39.0625
    spk_sr = 20000
    tconv = (framesPerSec / spk_sr) # ratio for fps to spike sampling
    #################################################################################################
    desen,baseblock,ipath,bsnm = get_mouse_info(alldesen,allBaseblock,mouse)
    os.chdir(ipath)
    ##################################################################################################
    meanRate = {}
    allPlaceMap = {}
    allOccMap = {}
    allSpkMap = {}
    #################################################################################################
    for sindx,sess in enumerate(sessions):
        ### Select session
        sessionLabel = get_descode(desen,sess)
        fname = baseblock + '_' + str(sessionLabel['filebase'].index[0])
        ### Load tracking
        track = vbf.load_tracking(fname,smoothing=1)
        sessDur = get_session_duration(desen,sess)
        ### Load interval times
        #print('ipath: {}'.format(ipath))
        intervalTimes = get_pulsetimes(ipath,desen,sess,ext=intv_type,tconv=tconv,debug=False)
        trkIntervalTimes = gen_valid_times(intervalTimes)
        ### Generate occupancy map
        OccMap = genOccMap(track,trkIntervalTimes,nbins=nbins,mazedim=mazedim)
        OccMap = (OccMap / framesPerSec) # convert to Hz
        ### Initialize containers
        PVpMap = np.array([]).reshape(nbins,nbins,0)
        PVsMap = np.array([]).reshape(nbins,nbins,0)
        rate = []
        for cindx,cluID in enumerate(cluID_list):
            print('Processing {0:}, cluID: {1:}'.format(bsnm,cluID))
            ### Generate spike map
            res,clu = \
            vbf.LoadSpikeTimes(fname,trode=None,MinCluId=2,res2eeg=(20000./20000))
            spikes = res[clu==cluID]
            rate.append(np.around((len(spikes) / sessDur),3))
            spkMap = genSpikeMap(track,spikes,nbins=nbins,mazedim=mazedim)
            ### stack spkMaps
            PVsMap = np.dstack((PVsMap,spkMap))
            ### Create placemap by diving spike map by occupancy map
            placeMap = (spkMap / OccMap)
            #print(placeMap.shape)
            ## Stack smoothed placeMaps for each cell
            smPlaceMap = smooth_map(OccMap,spkMap,smooth=smooth)
            PVpMap = np.dstack((PVpMap,smPlaceMap))#,axis=2)
            #print(PVpMap.shape)
            ### Get peak firing bin and rate from smoothed map
            peak_bin, peak_val = get_max_bin(OccMap,spkMap,placeMap,smooth=smooth)
        allOccMap[sess] = OccMap
        allSpkMap[sess] = PVsMap
        allPlaceMap[sess] = PVpMap
        meanRate[sess] = rate
        ##############################################################################################       
    return allOccMap,allSpkMap,allPlaceMap,meanRate
######################################################################################################
def smooth_map(OccMap,spkMap,smooth=1.0):
    '''
    
    '''
    smOcc = spim.filters.gaussian_filter(OccMap,smooth,mode='constant')
    smSpk = spim.filters.gaussian_filter(spkMap,smooth,mode='constant')

    return smSpk / smOcc
######################################################################################################
def get_max_bin(OccMap,spkMap,placeMap,smooth=1.0):
    '''
    
    '''
    if smooth is not None:
        smPlaceMap = smooth_map(OccMap,spkMap,smooth=smooth)
        max_bin = np.unravel_index(np.nanargmax(smPlaceMap), smPlaceMap.shape)
        max_val = smPlaceMap[max_bin]
    else:
        max_bin = np.unravel_index(np.nanargmax(placeMap), placeMap.shape)
        max_val = placeMap[max_bin]

    return max_bin,np.round(max_val,3)
######################################################################################################
def plot_placemaps_one_sess(OccMap,spkMap,placeMap,sess='',nbins=30,smooth=None,cmax=.5,size=[4,4],axoff=True,colbar=False):
    '''
    Inputs: OccMap,
    
    Output: fig
    '''
    rr,cc = 1,1 #len(sessions)
    wcm = size[0] * 1 # nSess # should be 2
    hcm = size[1] # should be 2 
    fig, ax = plt.subplots(rr,cc,figsize = cm2inch(wcm,hcm),\
                           gridspec_kw = {'wspace':0,'hspace':0})
    value = 0
    cmap = plt.get_cmap('jet')
    cmap.set_bad(color='white')
    ###########################################################################################
    if smooth is not None:
        smOcc = spim.filters.gaussian_filter(OccMap,smooth,mode='constant')
        smSpk = spim.filters.gaussian_filter(spkMap,smooth,mode='constant')
        smPlaceMap = smSpk / smOcc
        masked_data = np.ma.masked_where(OccMap == value, smPlaceMap)
    else:
        masked_data = np.ma.masked_where(OccMap == value, placeMap)
    hmin = np.min(masked_data)
    hmax = cmax * np.max(masked_data)
    im = ax.imshow(masked_data.T,clim=(hmin,hmax),cmap=cmap,interpolation='Nearest')
    ax.set_xlim(0,nbins)
    ax.set_ylim(nbins,0)
    if axoff:
        ax.set_axis_off()
    ax.set_title(sess,y=-.3,fontsize=8)
    if colbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad="5%")
        cb1 = fig.colorbar(im,cax=cax,orientation="vertical")
    #ftitle = bsnm + '_' + str(units['des'][cluID]) + '_' + str(cluID)
    #fig.suptitle(ftitle, y=-.1,fontsize=10)
    #############################################################################################
    return fig,ax
#################################################################################################
def mean_two_lists(idata,keys):
    
    return [np.nanmean(k) for k in zip(idata[keys[0]],idata[keys[1]])]
#################################################################################################
def merge_all_ctype(dataMetric,keys,celltype):
    '''
    
    '''
    okey = keys[0] + '_' + keys[1]

    odata = {}
    for cindx,ctype in enumerate(celltype):
        tempDat = {}
        idata = dataMetric[ctype]
        tempDat[okey]= mean_two_lists(idata,keys)
        odata[ctype] = tempDat

    return odata
#################################################################################################
def bin_spikes(spiketrain,edges,num_neurons):
    '''
    
    '''
    num_bins = len(edges)-1 # Number of bins
    #num_neurons = ncells #spiketrain.shape[0] # Number of neurons
    #print('spiketrain dimensions:', num_neurons,',',num_bins)
    print('There are {0:} bins and {1:} neurons'.format(num_bins,num_neurons))
    neural_data = np.empty([num_bins,num_neurons]) # Initialize array for binned neural data
    
    #Count number of spikes in each bin for each neuron, and put in array
    for i in range(num_neurons):
        neural_data[:,i] = np.histogram(spiketrain[i],edges)[0]

    return neural_data
##############################################################################################
def get_sess_binspikes(sess_fname,maxT,cluID_list=None,dT=1,tconv=1):
    '''
    
    '''
    spikes = []
    res,clu = vbf.LoadSpikeTimes(sess_fname,res2eeg=tconv)
    des = load_units(sess_fname.rsplit('_',1)[0],each_trode_ext='.des.', all_trode_ext='.desf')
    if cluID_list is None:
        cluID_list = list(enumerate(np.asarray(des['des']),start=2))
    
    #print(cluID_list,len(res),len(clu))
    
    for cindx,cluID in enumerate(cluID_list):
        spikes.append(res[clu==cluID])

    bins = np.arange(0,maxT+1,dT)
    num_neurons = len(spikes)
    binSpikes = bin_spikes(spikes,bins,num_neurons)

    return spikes,binSpikes
################################################################################################   
def show_mat(oMat,sess='test',cmap='binary',xlab='time (s)',ylab='Cells',fs=12,figsize=(6,6),vlim=(0,2)):
    '''
    oMat should be 2d - time x cells, with firing rate as z (colour)
    '''
    fig,ax = plt.subplots(1,1,figsize=cm2inch(figsize))
    ax.imshow(oMat,aspect='auto',cmap=cmap,vmin=vlim[0],vmax=vlim[1])
    ax.set_xlabel(xlab,fontsize=fs)
    ax.set_ylabel(ylab,fontsize=fs)
    ax.set_title(sess,fontsize=fs)
    
    return fig,ax
################################################################################################ 

#####################################################
def get_no_light(arr1,arr2):
    '''
    
    '''
    a = arr1[:,0]
    b = arr2[:,0]
    c = np.isin(a, b, invert=True)
    
    return arr1[c,:]
######################################################
def get_theta_cycles(sess_fname,chLabel):
    '''
    
    '''
    th_fname = sess_fname + '.theta.cycles.' + str(chLabel)
    tempTheta = np.load(th_fname)
    thetacycles = np.array((tempTheta[:,1],tempTheta[:,5])).T
    
    return thetacycles
######################################################################################
def get_valid_cycles(cycles,nsamples=[0,250]):
    '''
    
    '''
    lb,ub = nsamples[0],nsamples[1]
    width = cycles[:,1] - cycles[:,0]
    mask = np.where((width > lb) & (width < ub))

    return cycles[mask]
######################################################################################
def get_pulses(sess_fname,pulse_ext='.light_pulse',tconv=1250./20000):
    '''
    
    '''
    odata = open(sess_fname + pulse_ext).readlines()
    odata = [np.floor(np.array(l.split()).astype(int)*tconv).astype(int) for l in odata]
    
    return odata
######################################################################################
def get_theta_light(thetacycles,light,maxT_lfp):
    '''
    
    '''
    theta_light = np.array([]).reshape(0,2)
    for l in light:
        for t in thetacycles:
            if t[0] >= l[0] and t[1] <= l[1] and t[0] <= maxT_lfp:
                theta_light = np.vstack((theta_light,t.reshape(1,2)))

    return theta_light
######################################################################################
def IFR_by_theta(intervals,binspikes):
    '''
    
    '''
    nCells = binspikes.shape[1]
    oMat = np.array([]).reshape(nCells,0)

    for t in intervals.astype(int):
        #if t[1] - t[0] < 250: # 250 * 1/1250 = 0.2s (i.e. 5Hz)
        temp_spk = np.nansum(binspikes[t[0]:t[1],:], axis=0).reshape(nCells,1)
        oMat = np.concatenate((oMat,temp_spk),axis=1)

    return oMat
######################################################################################
def trk_by_theta(intervals,track,tconv=(39.0625/1250)):
    '''
    
    '''
    oMat = np.array([]).reshape(2,0)

    for t in intervals.astype(int):
        #if t[1] - t[0] < 250:
	    t_track = (t*tconv).astype(int)
	    x_track = np.nanmean(track['x'][t_track[0]:t_track[1]])
	    y_track = np.nanmean(track['y'][t_track[0]:t_track[1]])
	    
	    temp_trk = np.array([x_track, y_track]).reshape(2,1)
	    oMat = np.concatenate((oMat,temp_trk),axis=1)
            
    return oMat
######################################################################################
def trkCS_dict(baseblock,desen,itimes,sessions,trsm=1,distsm=5):
    '''
    Returns the full tracking (at 39.0625 Hz)
    ...and the distance travelled e.g. for 15s this will be shape (586,3)
    '''
    cstrk = {}
    for sindx,sess in enumerate(sessions):
        sessionLabel = get_descode(desen,sess)
        fname = baseblock + '_' + str(sessionLabel['filebase'].index[0])
        track = vbf.load_tracking(fname,smoothing=trsm)
        lims = [np.nanmin(track['x'].values),np.nanmax(track['x'].values),
                np.nanmin(track['y'].values),np.nanmax(track['y'].values)] 
        
        temptrk = []
        for ntrials in itimes[sess].index:
            ind1 = np.arange(itimes[sess]['begin'][ntrials],itimes[sess]['end'][ntrials])
            xx = np.copy(track['x'])[ind1]
            yy = np.copy(track['y'])[ind1]
            #xx,yy = track['x'][ind1].values,track['y'][ind1].values
            dist,_ = cartesian_dist(xx,yy,smoothing=distsm)
            trk = np.zeros((len(xx),3))
            trk[:,0] = xx
            trk[:,1] = yy
            trk[0,2] = 0 # dist is 1 element shorter than xx and yy so pad first ind with 0
            trk[1:,2] = dist
            temptrk.append(trk)
            
        cstrk[sess] = temptrk
    
    return cstrk
##########################################################################################
def adjust_plot_pub(
        ax, xlab='', ylab='', lw=0.5, raster=False, nms=[80,80], binwidth=2,
        xtwidth=20, xscale=1, yscale=1, xt_fmt='.0f', yt_fmt='.0f', fscale=[5,6],
        grid=True, gridcolor='k', pad=1.5):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FixedLocator, FixedFormatter

    plt.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['font.sans-serif'] = "Arial"

    SMALL_SIZE = fscale[0]
    MEDIUM_SIZE = fscale[1]

    # Get current tick positions (may be numpy array)
    raw_xticks = np.asarray(ax.get_xticks())
    raw_yticks = np.asarray(ax.get_yticks())

    # Keep only ticks inside the visible axis limits to avoid "out-of-range" ticks
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    eps = 1e-12
    xticks = raw_xticks[(raw_xticks >= xmin - eps) & (raw_xticks <= xmax + eps)]
    yticks = raw_yticks[(raw_yticks >= ymin - eps) & (raw_yticks <= ymax + eps)]

    # If filtering removed all ticks (rare), fall back to the raw ticks
    if xticks.size == 0:
        xticks = raw_xticks
    if yticks.size == 0:
        yticks = raw_yticks

    # format the labels as strings
    if raster:
        xtlabs = [format((x - (nms[0] / binwidth)), xt_fmt) for x in xticks]
    else:
        xtlabs = [format(x / xscale, xt_fmt) for x in xticks]
    ytlabs = [format(y / yscale, yt_fmt) for y in yticks]

    # axis labels
    ax.set_xlabel(xlab, fontsize=MEDIUM_SIZE)
    ax.xaxis.labelpad = pad
    ax.set_ylabel(ylab, fontsize=MEDIUM_SIZE)
    ax.yaxis.labelpad = pad

    # tick appearance
    ax.tick_params(width=lw, length=3 * lw, pad=pad)

    # spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(lw)
        ax.spines[axis].set_position(('outward', 3))

    # === SAFELY set tick positions + labels (keeping ticks within axis limits) ===
    try:
        # Preferred: set ticks with labels together (modern Matplotlib)
        ax.set_xticks(xticks, labels=xtlabs)
        ax.set_yticks(yticks, labels=ytlabs)
        ax.tick_params(labelsize=SMALL_SIZE)
    except TypeError:
        # Fallback: explicitly set FixedLocator and FixedFormatter for compatibility
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_major_formatter(FixedFormatter(xtlabs))
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_major_formatter(FixedFormatter(ytlabs))
        ax.tick_params(labelsize=SMALL_SIZE)

    if grid:
        ax.grid(grid, color=gridcolor, linestyle='--', linewidth=lw, alpha=.5)

    return ax
###############################################################################################################################
def adjust_plot_pub_old(
		ax,xlab='',ylab='',lw=0.5,raster=False,nms=[80,80],binwidth=2,
                xtwidth=20,xscale=1,yscale=1,xt_fmt='.0f',yt_fmt='.0f',fscale=[5,6],
                grid=True,gridcolor='k',pad=1.5):
    '''
    
    '''
    plt.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['font.sans-serif'] = "Arial"

    SMALL_SIZE = fscale[0]
    MEDIUM_SIZE = fscale[1]
    try:
        xtlabs = ax.get_xticks().tolist()
    except AttributeError:
        xtlabs = ax.get_xticks()

    if raster:
        xtlabs = [format((x-(nms[0]/binwidth)),xt_fmt) for x in xtlabs]
    else:
        xtlabs = [format(x/xscale,xt_fmt) for x in xtlabs]
    try:
        ytlabs = ax.get_yticks().tolist()
    except AttributeError:
        ytlabs = ax.get_yticks()

    ytlabs = [format(y/yscale,yt_fmt) for y in ytlabs]

    # Print x and y axis labels
    ax.set_xlabel(xlab,fontsize=MEDIUM_SIZE)
    ax.xaxis.labelpad = pad 
    ax.set_ylabel(ylab,fontsize=MEDIUM_SIZE)
    ax.yaxis.labelpad = pad 
        
    # Set tick width
    ax.tick_params(width=lw,length=3*lw,pad=pad)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(lw)
        ax.spines[axis].set_position(('outward', 3))

    # Set x and y ticklabels
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(xtlabs, fontsize=SMALL_SIZE)

    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ytlabs, fontsize=SMALL_SIZE)
    
    #ax.xaxis.set_ticklabels(xtlabs,fontsize=SMALL_SIZE)
    #ax.yaxis.set_ticklabels(ytlabs,fontsize=SMALL_SIZE)

    if grid:
        ax.grid(grid,color=gridcolor,linestyle='--',linewidth=lw,alpha=.5)
    
    return ax
######################################################################################################
def remove_outliers(idata,thresh=3):
    '''

    '''
    mean = np.nanmean(idata)
    std = np.nanstd(idata)
    lb = mean-(thresh*std)
    ub = mean+(thresh*std)
    idata = idata[idata>lb]
    idata = idata[idata<ub]

    return idata
######################################################################################################
def remove_outliers_dict(iDict,ikeys,okeys,n_sd=3):
    '''
    
    '''
    oDict = {}
    for i_indx,ikey in enumerate(ikeys):
        tempDict = {}
        for o_indx,okey in enumerate(okeys):
            idata = np.array(iDict[ikey][okey])
            sd = np.nanstd(idata)
            xbar = np.nanmean(idata)
            ub_thresh = xbar + (n_sd * sd)
            lb_thresh = xbar - (n_sd * sd)
            print(ikey,okey,ub_thresh,lb_thresh)
            mask = np.where(np.logical_and(idata>=lb_thresh, idata<=ub_thresh))
            #print(mask)
            tempDict[okey] = idata[mask]
        oDict[ikey] = tempDict
        
    return oDict
#########################################################################################################
def bandwidth_estimator(x):
    '''

    '''
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import LeaveOneOut

    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
	            {'bandwidth': bandwidths},
	            cv=None) #LeaveOneOut(len(x)))
    grid.fit(x[:, None]);
    
    return grid.best_params_['bandwidth']
#######################################################################################################
def calc_mean_bw(iDict,innerkey,outerkey,mean=True):
    '''
    
    '''
    odata = []
    datametric = iDict
    for ikey_indx,ikey in enumerate(innerkey):
        for okey_indx,okey in enumerate(outerkey):
            x = np.array(datametric[ikey][okey])
            x = x[~np.isnan(x)]
            print(ikey,okey,bandwidth_estimator(x))
            odata.append(bandwidth_estimator(x))
    print()
    if mean:
        print(np.nanmean(odata))
        return np.nanmean(odata)
    else:
        print(np.nanmedian(odata))
        return np.nanmedian(odata)
########################################################################################################
def calc_mean_bw_1var(iDict,innerkey):
    '''
    
    '''
    odata = []
    datametric = iDict
    for ikey_indx,ikey in enumerate(innerkey):
        x = np.array(datametric[ikey])
        x = x[~np.isnan(x)]
        print(ikey,bandwidth_estimator(x))
        odata.append(bandwidth_estimator(x))
    print()        
    print(np.nanmean(odata))

    return np.nanmean(odata)
######################################################################################################
def update_df(df,gp1=13,gp2=3,nconds=2,gp_keys=['nes','mmlv'],okey='subgroup'):
    '''
    Enables using a different colour for dabest data points, as specified in column 'subgroup'
    Assumes nconds e.g. 4 if 2 ctypes and 2 sess types
    '''
    each_cond = np.concatenate((np.repeat(gp_keys[0],gp1),np.repeat(gp_keys[1],gp2)))
    subgroup = np.tile(each_cond,nconds)
    df[okey] = subgroup

    return df
###########################################################################################
def isvalid(number):
    if number is None or np.isnan(number):
        return False
    else:
        return True
#####################################################################    
def random_select(iDict,n=200,lb=None):
    '''
    Takes dictionary as input and randomly selects n items
    If number of elements is < n, it takes numel
    '''
    import random
    
    random.seed(999)
    oDict = {}
    for key,val in iDict.items():
        if lb is None:
            val = [x for x in val]
        else:
            val = [x for x in val if x > lb]
        print(key,len(val))
        numel = len(val)
        if numel > n:
            new_vals = random.sample(val,n)
        else:
            new_vals = random.sample(val,numel)
        oDict[key] = np.array(new_vals)
    
    return oDict
#####################################################################
def get_mask_between_lb_ub(idata,irange=[0.1,.999]):
    '''
    returns indices that lie between lb and ub as a list
    '''
    lb,ub = irange[0],irange[1]
    idata = np.array(idata).astype('float')
    mask = np.where((idata > lb) & (idata < ub))
    
    return list(mask[0])
#####################################################################
def find_common(a,b):
    '''
    a and b must be lists
    returns a list
    ''' 
    return [x for x in a if x in b]
#####################################################################
def get_valid_sessions(sessions,stype='fam_nov'):
    '''
    
    '''
    if stype == 'nov_one_sess':
        tempSess1 = [x for x in sessions if x.startswith('n1')][0]
        
        return [tempSess1]

    if stype == 'fam_one_sess':
        tempSess1 = [x for x in sessions if x.startswith('f1b') or x.startswith('f1nostim')][0]
        
        return [tempSess1]
    if stype == 'fam_nov':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('f1') or x.startswith('n1')]

        # second pass filter for fam:
        tempSess1 = [x for x in sessions if x.startswith('f1nostim') or x.startswith('f1b')][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x.startswith('n1b') or x.startswith('n1nostim') \
                     or x.startswith('n1c') or x.startswith('n1train1')][0]
    if stype == 'fam':
        # first pass filter for fam:
        sessions = [x for x in sessions if x.startswith('f1') or x.startswith('n1')]

        # second pass filter for 1st fam:
        tempSess1 = [x for x in sessions if x.startswith('f1nostim1') or x.startswith('f1b')][0]

        # second pass filter for 2nd fam:
        try:
            tempSess2 = [x for x in sessions if x.startswith('f1b2') or x.startswith('f1a') or \
                     x.startswith('f1nostim2')][0]
        except:
            tempSess2 = [x for x in sessions if x.startswith('f1stim')][0]
            
    if stype == 'nov':
        # first pass filter for fam:
        sessions = [x for x in sessions if x.startswith('n1')]

        # second pass filter for 1st fam:
        tempSess1 = [x for x in sessions if x.startswith('n1b') or x.startswith('n1nostim') \
                    or x.startswith('n1csmb') or x.startswith('n1train1')][0]

        # second pass filter for 2nd fam:
        try:
            tempSess2 = [x for x in sessions if x.startswith('n1a') or x.startswith('n1nostim2') or \
                     x.startswith('n1train2') or x.startswith('n1cspb')][0]
        except:
            tempSess2 = [x for x in sessions if x.startswith('n1stim') or x.startswith('n1csma')][0]
            
    if stype == 'fam_stim':
        # first pass filter for fam:
        sessions = [x for x in sessions if x.startswith('f1') or x.startswith('n1')]

        # second pass filter for 1st fam:
        tempSess1 = [x for x in sessions if x.startswith('f1nostim')][0]
        
        # second pass filter for 2nd fam:
        try:
            tempSess2 = [x for x in sessions if x.startswith('f1stim1') or x.startswith('f1stim2')][0]
        except:
            tempSess2 = [x for x in sessions if x.startswith('f1stim')][0]

    if stype == 'fam_novstim':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('f') or x.startswith('n')]

        # second pass filter for fam:
        tempSess1 = [x for x in sessions if x.startswith('f1nostim') or x.startswith('f1b')][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x.startswith('n1stim') or x.startswith('n2stim')][0]
        
    if stype == 'n2orn3_n2orn3stim':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('n')]

        # second pass filter for fam:
        tempSess1 = [x for x in sessions if x == 'n2' or x == 'n3'][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x == 'n2stim' or x == 'n3stim'][0]
        
    if stype == 'n2orn3stim_n4orn5':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('n')]

        # second pass filter for fam:
        tempSess1 = [x for x in sessions if x == 'n2stim' or x == 'n3stim'][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x == 'n4' or x == 'n5'][0] 
    
    if stype == 'n2orn3stim_n4orn5stim':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('n')]

        # second pass filter for fam:
        tempSess1 = [x for x in sessions if x == 'n2stim' or x == 'n3stim'][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x == 'n4stim' or x == 'n5stim'][0]
        
    if stype == 'f1b_f1a':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('f')]

        # second pass filter for fam:
        tempSess1 = [x for x in sessions if x == 'f1b'][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x == 'f1a'][0]
        
    if stype == 'f1b_n1b':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('f') or x.startswith('n')]

        # second pass filter for fam:
        tempSess1 = [x for x in sessions if x == 'f1b'][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x == 'n1b'][0]
        
    if stype == 'n1b_n1a':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('n')]

        # second pass filter for fam:
        tempSess1 = [x for x in sessions if x == 'n1b'][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x == 'n1a'][0]
        
    if stype == 'n2orn3_n4orn5':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('n')]

        # second pass filter for fam:
        tempSess1 = [x for x in sessions if x == 'n2' or x == 'n3'][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x == 'n4' or x == 'n5'][0]
        
    if stype == 'n2orn3_n4orn5stim':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('n')]

        # second pass filter for nov:
        tempSess1 = [x for x in sessions if x == 'n2' or x == 'n3'][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x == 'n4stim' or x == 'n5stim'][0]
        
    if stype == 'n2orn3stim_n3orn4':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('n')]

        # second pass filter for fam:
        tempSess1 = [x for x in sessions if x == 'n2stim' or x == 'n3stim'][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x == 'n3' or x == 'n4'][0]
        
    if stype == 'n1b_n2orn3':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('n')]

        # second pass filter for fam:
        tempSess1 = [x for x in sessions if x == 'n1b'][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x == 'n2' or x == 'n3'][0]
        
    if stype == 'n1b_n2orn3stim':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('n')]

        # second pass filter for fam:
        tempSess1 = [x for x in sessions if x == 'n1b'][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x == 'n2stim' or x == 'n3stim'][0]
    
    if stype == 'n1b_n4orn5':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('n')]

        # second pass filter for fam:
        tempSess1 = [x for x in sessions if x == 'n1b'][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x == 'n4' or x == 'n5'][0]
        
    if stype == 'n1b_n4orn5stim':
        # first pass filter for nov_fam:
        sessions = [x for x in sessions if x.startswith('n')]

        # second pass filter for fam:
        tempSess1 = [x for x in sessions if x == 'n1b'][0]

        # second pass filter for nov:
        tempSess2 = [x for x in sessions if x == 'n4stim' or x == 'n5stim'][0]
           

    return [tempSess1,tempSess2]
########################################################################################################################################
def subset_df(df,col_head='group',group='SHAM'):
    '''
    
    '''
    df = df[df[col_head] == group]
    return df
########################################################################################################################################
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
##########################################################################################################################################
def interpolate_nan_1d(y):

    nans,x = nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    return y
##########################################################################################################################################
def take_closest(myList,myNumber,choose_pos='before'):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    You can take the value before, after, or the closest regardless
    If two numbers are equally close, return the smallest number.
    """
    from bisect import bisect_left
    
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    ############################################################################
    if choose_pos == 'before':
        return before
    elif choose_pos == 'after':
        return after
    elif choose_pos == 'closest':
        if after - myNumber < myNumber - before:
            return after
        else:
            return before
###############################################################################################################
def get_closest_index(input_list,targ_val):
    '''
    '''
    difference_array = np.absolute(np.array(input_list)-targ_val)
    return difference_array.argmin()
###############################################################################################################
def generate_IFR(iDict,ikey_list,smooth=True,smpts=3):
    '''
    
    '''
    odata = {}
    for ikey_indx,ikey in enumerate(ikey_list):
        tempdict = np.squeeze(iDict[ikey][0])
        tempdict = inf_to_nan(tempdict)
        odata[ikey] = tempdict[:, ~np.isnan(tempdict).any(axis=0)]
        # x[:, ~np.isnan(x).any(axis=0)]
        if smooth:
            odata[ikey] = savgol_filter(odata[ikey],smpts,1,axis=0)
        
    return odata
#####################################################################################################
def generate_metric(iDict,x_range=(150,250),method=np.nansum,smooth=True):
    '''
    
    '''
    odata = {}
    for key,val in iDict.items():
        print(key,val.shape)
        odata[key] = method(val[x_range[0]:x_range[1],:],axis=0)

    return odata
######################################################################################################
def generate_z_score(allDat,celltype,smpts=None):
    '''
    '''
    odata = {}
    for ikey,val in allDat.items():
        tempdict = {}
        for cindx,ctype in enumerate(celltype):
            if smpts is not None:
                tempdat = savgol_filter(np.squeeze(val[ctype][0]),smpts,1,axis=0)
            else:
                tempdat = np.squeeze(val[ctype][0])
            tempdict[ctype] = stats.zscore(tempdat,axis=0)
        odata[ikey] = tempdict
        
    return odata
########################################################################################################
def generate_z_max(idata,xrange=(160,240)):
    
    odata = {}
    for ikey,val in idata.items():
        tempdict = {}
        for cindx,ctype in enumerate(celltype):
            tempdat = np.nanmax(val[ctype][xrange[0]:xrange[1],:],axis=0)
            tempdict[ctype] = tempdat
            #print(ikey,ctype,tempdict[ctype].shape)
        odata[ikey] = tempdict

    return odata
####################################################################################
def find_argmax(idata,xrange=(150,250),offset=50):
    argmax = np.argmax(np.squeeze(idata[xrange[0]:xrange[1],:]),axis=0)
    return np.array([x-offset for x in argmax])
####################################################################################
def create_argmax_dict(iDict,xrange=(150,250),offset=20):
    
    oDict = {}
    for ikey,val in iDict.items():
        tempdict = {}
        for key2,val2 in val.items():
            tempdict[key2] = find_argmax(val2,xrange=xrange,offset=offset)   
        oDict[ikey] = tempdict
        
    return oDict
#####################################################################################
def remove_nans_paired(iDict,okey_list,ikey_list=['ds','swr']):
    
    for okey_indx,okey in enumerate(okey_list):
        
        a = iDict[ikey_list[0]][okey]
        b = iDict[ikey_list[1]][okey]
        
        mask = np.logical_and(~np.isnan(a).any(axis=0),~np.isnan(b).any(axis=0))
        
        print(okey,np.squeeze(a).shape)
        iDict[ikey_list[0]][okey] = np.squeeze(a[:,mask])
        iDict[ikey_list[1]][okey] = np.squeeze(b[:,mask])

    return iDict
#####################################################################################
def remove_non_sig(iDict,iDict_z,z_thresh=3):
    
    oDict = {}
    for ikey,val in iDict.items():
        tempdict = {}
        for key2,val2 in val.items():
            ref2 = iDict_z[ikey][key2]
            tempdict[key2] = val2[ref2 > z_thresh]
        oDict[ikey] = tempdict
  
    return oDict
####################################################################################
def bin_array(idata, axis=0, binstep=10, binsize=10, func=np.nanmean):
    
    idata = np.array(idata)
    dims = np.array(idata.shape)
    argdims = np.arange(idata.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]
    idata = idata.transpose(argdims)
    idata = [func(np.take(idata,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
    
    odata = np.array(idata).transpose(argdims)
    
    return odata
####################################################################################
def generate_desv_field(database,ctype,ext='.desv',field='trough2peak',pprint=False):
    '''
    e.g.  call as:
    for cindx,ctype in enumerate(ctype_list):
        odata[ctype] = generate_desv_field(database,
                                           ctype,
                                           ext='.desv',
                                           field='trough2peak',
                                           pprint=True)
    '''
    tempdat = []
    for dindx,ipath in enumerate(database):
        os.chdir(ipath)
        os.getcwd()
        # get basename from database
        bsnm = ipath.rsplit('/', 1)[-1]
        baseblock = ipath + '/' + bsnm  
        ##############################################
        desv = pd.read_csv(baseblock+ext,index_col=0)
        df = desv[desv['des']==ctype]
        if pprint:
            print(bsnm,len(df))
            print(df[field].values)
        tempdat.extend(df[field].values)
        
    return np.array(tempdat)
################################################################################################
def find_common(a,b):
    '''
    a and b must be lists
    returns a list
    ''' 
    return [x for x in a if x in b]
################################################################################################
def find_difference(a,b):
    '''
    a and b must be lists
    returns a list
    ''' 
    return [x for x in a if x not in b]
################################################################################################
def find_intersection(a,b):
    return set(a).intersection(b)
################################################################################################
def get_session_data(Baseblock, session):
    
    #Compiles data (speed, ripple count, theta-delta ratio) from a session into a dataframe
    
    #Reading pulse file 
    import smLfpFunctions3 as slf
    
    print('Finding ripples...                                       ', end = '\r')
    pulsepath = Baseblock + '/' + Baseblock.split('/')[-1] + '_' + str(session) + '.swr_pulse'
    peaks = []
    with open(pulsepath) as filen:
        for line in filen: 
            line = line.strip() 
            peaks.append(line)
    filen.close() 
    Peaks = pd.DataFrame({'Peak':[int(x.split()[0]) for x in peaks]})                          #Reading in peak timestamps
    Peaks['Time']=[x//(20000) for x in Peaks['Peak']]                                          #Binning by second
    
    #Reading pulse file
    print('Finding dentate spikes...                                       ', end = '\r')
    pulsepath = Baseblock + '/' + Baseblock.split('/')[-1] + '_' + str(session) + '.ds_pulse'
    temp_ds = []
    with open(pulsepath) as filen:
        for line in filen: 
            line = line.strip() 
            temp_ds.append(line)
    filen.close() 
    DStimes = pd.DataFrame({'DS_':[int(x.split()[0]) for x in temp_ds]})                          #Reading in peak timestamps
    DStimes['Time']=[x//(20000) for x in DStimes['DS_']]                                          #Binning by second
    
    #Getting tracking data
    print('Obtaining tracking data...                                       ', end = '\r')
    trackpath = Baseblock + '/' + Baseblock.split('/')[-1] + '_' + str(session)
    track = vbf.load_tracking(trackpath, smoothing = 3)                                        #Loading tracking data
    xx = track['x'].values                                                                     #Obtaining x coordinates
    yy = track['y'].values                                                                     #Obtaining y coordinates

    dist,totaldist = cartesian_dist(xx,yy,smoothing=2)                                     #Calculating the distanced traversed

    fps = 39.0625                                                                              #Frame rate
    total_dur_sec = (dist.shape[0] / fps)                                                      #The total length of the session seconds

    xpts = np.linspace(0,total_dur_sec,dist.shape[0])                                          #Timestamps for each measurement of speed
    Track= pd.DataFrame({'xpts':xpts, 'dist':dist}) 
    Track['Time']=[x//(1) for x in Track['xpts']]                                              #Binning by second
    
    #Getting theta-delta ratio
    print('Calculating theta-delta ratio...                                       ', end = '\r')
    baseblock = Baseblock + '/' + Baseblock.split('/')[-1]
    trodes = vbf.LoadTrodes(baseblock,par=None)
    tetrode,channel = slf.get_ripple_chan(baseblock,trodes,tet_only=False)
    
    tdpath = Baseblock + '/' + Baseblock.split('/')[-1] + '_' + str(session)
    
    lfps = vbf.loadlfps(tdpath)[tetrode-1]                                                     #Loading in LFP data from the specified tetrode
    th_filt = slf.BPFilter(lfps,SamplingRate=1250,Low=4,High=12,FilterOrder=2,axis=0)          #Filtering out theta from the LFP
    del_filt = slf.BPFilter(lfps,SamplingRate=1250,Low=1,High=4,FilterOrder=2,axis=0)          #Filtering out delta from the LFP
    time = np.array([(i/1250) for i in range(len(lfps))])                                      #Time is in seconds!!!
    ThetaDelta = pd.DataFrame({'Time':time,                                                    #The time at which each measurement was made
                       'LFP':lfps,                                                             #LFP data
                       'Theta':abs(th_filt),                                                   #Theta strength
                       'Delta':abs(del_filt)})                                                 #Delta strength                                                 #Provisional ratio between theta and delta strength
    ThetaDelta['Time'] = [x//(1) for x in ThetaDelta['Time']]                                  #Binning by second
    
    keys = ThetaDelta.keys()
    out_dict = {key:[] for key in keys}
    
    for ind,val in enumerate(list(ThetaDelta['Time'].unique())):                                #iterating over time-bins
        for key in keys:                                                                        #iterating over columns in the dataframe
            out_dict[key].append(np.nanmean(list(ThetaDelta[ThetaDelta['Time']==val][key])))    #computing average of values that correspond to a key / time-bin
            
    out_df = pd.DataFrame(out_dict)
    out_df['Ratio'] = out_df['Theta']/out_df['Delta']                                           #calculating theta-delta ratio

    #Compiling information into a single DF
    print('Assembling data...                                       ', end = '\r')
    session_length = max(out_df['Time'])                                                        #using the max time-stamp from theta-delta ratio as session length
    
    time = []
    speed = []
    swr = []
    ds = []
    tdr = []
    for i in range(0,int(session_length)+1):                                                    #iterating over each second in the session
        time.append(i)
        speed.append(sum(Track[Track['Time']==i]['dist']))                                      #calculating total distance travelled per sec
        swr.append(len(Peaks[Peaks['Time']==i]))                                                #counting ripples per sec
        ds.append(len(DStimes[DStimes['Time']==i]))
        tdr.append(np.nanmean(out_df[out_df['Time']==i]['Ratio']))                              #finding the theta-delta ratio per sec
        
    df = pd.DataFrame({'Time':time, 'Speed':speed, 'RippleCount':swr, 'DSCount':ds, 'ThetaDeltaRatio':tdr})
    
    return df
###########################################################################################################################################################################################################################
def get_mazedim(baseblock):
    '''

    '''
    #baseblock = mainpath + bsnm + '/' + bsnm
    
    try:
        mazedim = np.loadtxt(baseblock + '.mazedim',dtype='int')
    except OSError:
        print('Cant find file')
        if 'msm19' in baseblock:
            mazedim = [230,615,90,465]
        elif 'msm22' in baseblock:
            if (baseblock == 'msm22-210308') or (baseblock == 'msm22-210309'):
                mazedim = [205,635,100,500]
            else:
                mazedim =  [180,610,55,455]
        elif 'msm23' in baseblock:
            mazedim = [200,640,80,480]
        else:
            mazedim = None #[265,640,130,530]
            
    return mazedim
##########################################################################################################################################################################################################################  
def count_events(pulse_list,bsnm):
    '''
    
    '''
    event_sum = 0
    for pl in pulse_list:
        try:
            event_sum += pl.shape[0]
        except IndexError:
            event_sum += 0
            print(bsnm,'no pulses?')

    return event_sum
########################################################################################################################################################################################################################## 
