from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import glob
import numpy as np
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from math import fsum, exp
from random import uniform
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
from math import cos, asin, sqrt
import scatter_plot as sca

def read_l2_GEMS( Filename ):
    """
    Read GEMS level2 aerosol product: groups ['Data_Fields', 'Geolocation_Fields']
    """
    data = dict()
    #-- open the file to read
    h5f = h5py.File( Filename, 'r' )
    #-- read variables of each group
    groups = ['Data Fields', 'Geolocation Fields']
    for grp in groups:
        tempgrp = h5f[ grp ]
        for var in tempgrp.keys():
            data[var] = tempgrp[var][:]
    #-- close file
    h5f.close()
    return data

#####
def quality_control( variable, flag, product ):
    """
    Mask the data if the quality flag is bad
    """
    # check matrix shape
    if len(variable.shape) == len(flag.shape):
        for i in range(len(variable.shape)):
            if (variable.shape[i] != flag.shape[i]):   
                sys.exit('The {:d} flag shape is not the same as variable shape!!!  '.format(i+1) )
    else:
        sys.exit(flag.shape, ' flag != variable shape '+ variable.shape )   

    if product == 'AERAOD':   
        QA_msk    = QA_flag_AOD(flag)
        masked_data = np.ma.masked_where(QA_msk['b_msk'] != 0, variable)   
        
    return masked_data

####
def QA_flag_AOD(QA):

        # User's guide GEMS AOD L2: mask 2~8 binary 
        QA_msk = {}

        # Reliable mask
        r_msk = QA & 3

        # Bad pixel Mask
        b_msk = QA & 508
        b_msk = b_msk >>2

        # surface Mask
        s_msk = QA & 15872
        s_msk = s_msk >>9

        QA_msk['r_msk']   = r_msk
        QA_msk['b_msk']   = b_msk
        QA_msk['s_msk']   = s_msk

        return QA_msk
    
def narrowDomain(data, cord, var):
    '''
    cord, boundary of the DOI   
    lon: [array-like], longitude of the region that covered by the
         pixel of interest
    lat: [array-like], latitude of the region that covered by the
         pixel of interest
    measure: [array-like], data to be localized
    
    Function to localized a small region for regriding
    '''
    
    import numpy as np
    lat = data['latitude']
    lon = data['longitude']

    uppLat = cord[0] ; lowLat = cord[1]
    lefLon = cord[2] ; rigLon = cord[3]
    result = np.where((lat >= lowLat ) & (lat <= uppLat ) & \
                      (lon >= lefLon ) & (lon <= rigLon ))
    if len(result[0]) == 0:
        return 0
    H_min = np.min(result[0])
    H_max = np.max(result[0])
    V_min = np.min(result[1])
    V_max = np.max(result[1])
    
    for key in var:
    	data[key] = data[key][H_min:(H_max+1),V_min:(V_max+1)]
    	data['latitude'] = data['latitude'][H_min:(H_max+1),V_min:(V_max+1)]
    	data['longitude'] = data['longitude'][H_min:(H_max+1),V_min:(V_max+1)]

    return data	

def get_data_on_aeronet_irr(data, sat_lat, sat_lon, aer_lat, aer_lon, radi):
    # data: satellite (GEMS, TROPOMI, EPIC) aod
    # a_lat, a_lon: aeronet lat, lon
    # radi: desired radius for distance

    ixs   = [] # index for data
    iys   = [] # index for data
    iaers = [] # index for AERONET
    a_lon   = [] # selected AERONET lon
    a_lat   = [] # selected AERONET lat
    sat_aod = [] # selected AOD
    sat_aod_std = [] # AOD standard dev
    aer_lat = np.array(aer_lat)
    aer_lon = np.array(aer_lon)
    dis = np.sqrt((aer_lat - sat_lat)**2 + (aer_lon - sat_lon)**2)
    i0,j0 = np.where(dis < radi)
    data_0 = data[i0,j0]
    idx = range(len(aer_lon))
    ixs.append(i0)
    iys.append(j0)
    a_lat.append(aer_lat)#[i])
    a_lon.append(aer_lon)#[i])
    data_0 = np.ma.masked_invalid(data_0)

    if np.ma.count(dis[i0,j0])>1 and np.ma.count(data_0) / np.ma.count(dis[i0,j0]) > 0.3:#0.3
        if (np.ma.std(data_0) < 0.5): #0.3    
            sat_aod.append(np.nanmean(data_0))
            sat_aod_std.append(np.nanstd(data_0))
        else:
            sat_aod.append(np.nan)
            sat_aod_std.append(np.nan)
    else:
        sat_aod.append(np.nan)
        sat_aod_std.append(np.nan)
    return sat_aod, sat_aod_std, a_lat, a_lon
    
### REAL CODE STARTS HERE ###

dateee = ['210328','210426','220410','230310','230519','230520','210331','210810','210811','220409','230326','230417']

dust = True #False

for idate, yymmdd in enumerate(dateee):
    idfile = yymmdd
    if yymmdd == '210331' or yymmdd == '210810' or yymmdd =='210811' or yymmdd == '210328' or yymmdd=='210426':
        new_c = False
    else:
        new_c = True
    if dust == True:
        region = [100,135,30,50]
    elif idfile == '210810' or idfile == '210811':
        region = [95, 120, 35, 50]
    else:
        region = [90,110,10,26]
        
    sat = ['epic','trop']
    prods =['aod']
    aer_prod = ['AERAOD']
    cord = [region[3],region[2],region[0],region[1]]
    iday = str(yymmdd)[4:]
    mm_str   = '20'+str(yymmdd)[:4]

    # 1. Find GEMS files from date
    fdir = '/Dedicated/jwang-data/shared_satData/GEMS/L2/V2.0/'

    varnames = {'AERAOD':['FinalAerosolOpticalDepth','FinalAlgorithmFlags'], 
                'AEH':['AerosolEffectiveHeight','QualityFlag'], 'UVAI':['UVIndex', 'FinalAlgorithmFlags']}

    gems_file = []

    for iprod in aer_prod:
        data_dir = fdir + iprod + '/' + mm_str + '/' + iday +'/'
        flist    = sorted( glob.glob( data_dir + 'GK2_GEMS_L2_*_DPRO_ORI.nc') )
        gems_file.append(flist)

    # find gems times
    gems_time = np.ones((len(gems_file[0])),dtype='object')
    for i in range(len(gems_file[0])):
        gems_time[i] = gems_file[0][i][79:83]+'-'+gems_file[0][i][83:85]+'-'+gems_file[0][i][85:87]+'T'+gems_file[0][i][88:90]+':'+gems_file[0][i][90:92]

    gems_times = np.array(gems_time,dtype='datetime64')


    # TROPOMI file
    if dust == True:
        dir_trop = '/Dedicated/jwang-data2/xchen/data/tropomi/ALH/Asia/dust/'#smoke/'
    else:
        dir_trop = '/Dedicated/jwang-data2/xchen/data/tropomi/ALH/Asia/smoke/'

    trop_time = []
    trop_file = []
    if dust == True:
        for name in sorted(glob.glob(dir_trop + 'tropomi_'+str(idfile)+'T*DUST.v3_BlueRAZAOD2H0.2_NBAR_NDVI_slp_NoAgg.nc')):#'T*SMK.v3SMK.v4_BlueRAZAOD2H0.*_NBAR_NDVI_slp_NoAgg.nc'):#
            file_list = name
            trop_file.append(file_list)
            time_list = file_list[-52:-48]#[-57:-53]#
            trop_time.append( '20'+ str(yymmdd)[:2] + '-' + str(yymmdd)[2:4] + '-' + str(yymmdd)[4:6]+ 'T'+str(time_list)[:2] + ':' +str(time_list)[2:4])
    else:
        for name in sorted(glob.glob(dir_trop + 'tropomi_'+str(idfile)+'T*SMK.v3SMK.v4_BlueRAZAOD2H0.*_NBAR_NDVI_slp_NoAgg.nc')):
            file_list = name
            trop_file.append(file_list)
            time_list = file_list[-57:-53]
            trop_time.append( '20'+ str(yymmdd)[:2] + '-' + str(yymmdd)[2:4] + '-' + str(yymmdd)[4:6]+ 'T'+str(time_list)[:2] + ':' +str(time_list)[2:4])
    trop_times = np.array(trop_time,dtype='datetime64')

    # EPIC file
    dir_epic = '/Dedicated/jwang-data2/zhendlu/ALH/global_operational/20'+str(idfile[:2])+'/'+str(idfile)[2:4]+'/'
    epic_time = []
    epic_file = []
    if dust == True:
        if new_c == False:
            for name in sorted(glob.glob(dir_epic + 'epic_AOCH_20'+str(idfile)+'*.DST.v1_UVAI_Asia_Dust_newcali.nc')):
                file_list = name
                time_list = file_list[-42:-38]#[-24:-20]#
                epic_file.append(file_list)
                epic_time.append( '20'+ str(yymmdd)[:2] + '-' + str(yymmdd)[2:4] + '-' + str(yymmdd)[4:6]+ 'T'+str(time_list)[:2] + ':' +str(time_list)[2:4])
        else:
            for name in sorted(glob.glob(dir_epic + 'DSCOVR_EPIC_L2_AOCH_01_20'+str(idfile)+'*.nc')):
                file_list = name
                time_list = file_list[-12:-8]
                epic_file.append(file_list)
                epic_time.append( '20'+ str(yymmdd)[:2] + '-' + str(yymmdd)[2:4] + '-' + str(yymmdd)[4:6]+ 'T'+str(time_list)[:2] + ':' +str(time_list)[2:4])
            
    else:
        for name in sorted(glob.glob(dir_epic + 'DSCOVR_EPIC_L2_AOCH_01_20'+str(idfile)+'*_Asia_Smoke.nc')):
            file_list = name
            time_list = file_list[-24:-20]
            epic_file.append(file_list)
            epic_time.append( '20'+ str(yymmdd)[:2] + '-' + str(yymmdd)[2:4] + '-' + str(yymmdd)[4:6]+ 'T'+str(time_list)[:2] + ':' +str(time_list)[2:4])
    epic_times = np.array(epic_time,dtype='datetime64')

    # now GEMS
    sat = ['gems','trop','epic']
    aod_680 = dict()

    for isat, vsat in enumerate(sat):
        aod_680[vsat] = dict()

        if vsat == 'gems':
            for itime,vtime in enumerate(gems_times):
                str_vtime = str(vtime)
                aod_680[vsat][str_vtime] = dict()

                file = read_l2_GEMS(str(gems_file[0][itime]))
                var     = varnames['AERAOD'][0]
                fname   = varnames['AERAOD'][-1]
                aod_500 = quality_control(file['FinalAerosolOpticalDepth'][2,:,:],file['FinalAlgorithmFlags'],'AERAOD')
                aod_443 = quality_control(file['FinalAerosolOpticalDepth'][1,:,:],file['FinalAlgorithmFlags'],'AERAOD')
                ae = -(np.log(aod_500[:]/aod_443[:]))/(np.log(500/443))
                aod_680_value = aod_500*((680/500)**(-ae))
                data = {}
                data['latitude'] = file['Latitude'][:]
                data['longitude'] = file['Longitude'][:]
                data['aod'] = aod_680_value[:]
                var = ['aod']
                new = narrowDomain(data,cord,var)
                if new != 0:
                    aod_680[vsat][str_vtime]['lat'] = new['latitude'][:]
                    aod_680[vsat][str_vtime]['lon'] = new['longitude'][:]
                    aod_680[vsat][str_vtime]['aod_680'] = new['aod'][:]
                    aod_680[vsat][str_vtime]['aod_680'] = aod_680[vsat][str_vtime]['aod_680'].filled(np.nan)

        if vsat == 'trop':
            for itime,vtime in enumerate(trop_times):
                str_vtime = str(vtime)
                aod_680[vsat][str_vtime] = dict()
                file = Dataset(trop_file[itime])
                data = {}
                data['latitude'] = file['lat'][:]
                data['longitude'] = file['lon'][:]
                data['aod'] = file['aod'][:,:,0]
                data['aod'] = np.where(data['aod']>0.2,data['aod'],np.nan)
                var = ['aod']
                new = narrowDomain(data,cord,var)
                if new!=0:
                    aod_680[vsat][str_vtime]['lat'] = new['latitude'][:]
                    aod_680[vsat][str_vtime]['lon'] = new['longitude'][:]
                    aod_680[vsat][str_vtime]['aod_680'] = new['aod'][:]

        if vsat == 'epic':
            for itime,vtime in enumerate(epic_times):
                str_vtime = str(vtime)
                aod_680[vsat][str_vtime] = dict()
                file = Dataset(epic_file[itime])
                data = {}
                data['latitude'] = file['lat'][:]
                data['longitude'] = file['lon'][:]
                if new_c == False:
                    data['aod'] = file['aod'][:]
                    data['aod'] = np.where(data['aod']>0.2,data['aod'],np.nan)

                else:
                    data['aod'] = file['AOD'][:]
                    data['aod'] = np.where(data['aod']>0.2,data['aod'],np.nan)

                var = ['aod']
                new = narrowDomain(data,cord,var)
                if new != 0:
                    aod_680[vsat][str_vtime]['lat'] = new['latitude'][:]
                    aod_680[vsat][str_vtime]['lon'] = new['longitude'][:]
                    aod_680[vsat][str_vtime]['aod_680'] = new['aod'][:]

    # gather AERONET AOD data
    dd = '20'+ str(yymmdd)[:2]+'-'+str(yymmdd)[2:4]+'-'+str(yymmdd)[4:6]
    read = pd.read_csv('site_name_new2.csv',header=None)
    sitename = np.array(read).flatten()
    all_sites = pd.read_csv('AERONET_SITES.txt',header=1)
    sn = all_sites['Site_Name']

    a_aod = dict()
    a_lat = []
    a_lon = []
    for isite, vsite in enumerate(sitename):
        aeronet = glob.glob('./aeronet_csv_new/'+'*'+str(vsite)+'.lev15_AOD.csv')
        if len(aeronet)!=0:
            df = pd.read_csv(aeronet[0])
            if len(df)!=0 :
                if 'AOD_675nm' in df:
                    df[['Date','time']] = df['Time'].str.split(' ', expand=True)
                    aod = df['AOD_675nm'][df['Date']== str(dd)].values # or df['AOD_500nm'][df['Time']== '2021-04-16'].values
                    if aod.size > 0:
                        a_aod[vsite] = dict()
                        test = all_sites[(sn == vsite)]
                        a_aod[vsite]['lat'] = test['Latitude(decimal_degrees)']
                        a_aod[vsite]['lon'] = test['Longitude(decimal_degrees)']
                        a_aod[vsite]['aod_675'] = aod
                        a_aod[vsite]['time'] = df['time'][df['Date']==  str(dd)].values
                        

    times = [gems_times,trop_times,epic_times]
    radi = [0.2,0.2,0.2]
    valid_sites = a_aod.keys()
    x_aeronet = dict()
    y_sat = dict()

    for isat, vsat in enumerate(sat):
        y_sat[vsat] = dict()
        x_aeronet[vsat] = dict()
        for itime,vtime in enumerate(times[isat]):
            str_vtime = str(vtime)
            x_aeronet[vsat][str_vtime] = dict()
            y_sat[vsat][str_vtime] = dict()
            if len(aod_680[vsat][str_vtime]) != 0:
                lat = aod_680[vsat][str_vtime]['lat']
                lon = aod_680[vsat][str_vtime]['lon']
                aod_680_value = aod_680[vsat][str_vtime]['aod_680']
                dis = np.ones((lat.shape))
                for isite,vsite in enumerate(valid_sites):
                    x_aeronet[vsat][str_vtime][vsite] = dict()
                    y_sat[vsat][str_vtime][vsite] = dict()
                    obs_lat = a_aod[vsite]['lat']
                    obs_lon = a_aod[vsite]['lon']
                    obs_aod = a_aod[vsite]['aod_675']
                    aod_mean, aod_sh, aeronet_lat, aeronet_lon = get_data_on_aeronet_irr(aod_680_value, lat, lon, obs_lat, obs_lon ,radi[isat])
                    y_sat[vsat][str_vtime][vsite]['aod_mean'] = aod_mean
                    y_sat[vsat][str_vtime][vsite]['aod_std'] = aod_sh

                    # time
                    obs_time = a_aod[vsite]['time']

                    obs_time_64 = np.array(str(times[isat][0])[:-5]+obs_time,dtype='datetime64')
                    sats_times = np.array(times[isat],dtype='datetime64')

                    time_delta = []
                    for t in range(len(obs_time)):
                        time_delta.append(vtime - obs_time_64[t])
                    ttt = []
                    aodd = []
                    for tt in range(len(time_delta)):
                        if abs(time_delta[tt])<np.timedelta64(30,'m'):
                            ttt.append(obs_time_64[tt])
                            aodd.append(obs_aod[tt])
                    x_aeronet[vsat][str_vtime][vsite]['aod_mean'] = np.mean(aodd)
                    x_aeronet[vsat][str_vtime][vsite]['aod_std'] = np.std(aodd)

    sat_title = ['GEMS', 'TROPOMI', 'EPIC']
    for isat, vsat in enumerate(sat):
        x = []
        y = []
        x_std = []
        y_std = []
        for itime,vtime in enumerate(times[isat]):
            str_vtime = str(vtime)
            if len(aod_680[vsat][str_vtime]) != 0:
                for isite,vsite in enumerate(valid_sites):
                    x.append(x_aeronet[vsat][str(vtime)][vsite]['aod_mean'])
                    x_std.append(x_aeronet[vsat][str(vtime)][vsite]['aod_std'])
                    y.append(y_sat[vsat][str(vtime)][vsite]['aod_mean'])
                    y_std.append(y_sat[vsat][str(vtime)][vsite]['aod_std'])

        yy = []
        yy_std = []
        for i in range(len(y)):
            if y[i] == []:
                yy.append(np.nan)
            else:
                yy.append(y[i][0])

        for i in range(len(y_std)):
            if y_std[i] == []:
                yy_std.append(np.nan)
            else:
                yy_std.append(y_std[i][0])
        # save data for plotting all cases
        np.savez('./data_v3/'+str(yymmdd)+'_'+str(vsat), x = x, y = yy, x_std = x_std, y_std = yy_std)
        if len(x) > 0:
            maskx = x[yy!=np.nan]

        mask_test = ~np.isnan(x) & ~np.isnan(yy)
        if np.count_nonzero(mask_test==True) < 2:
            print('exit2')
        else:
            print(x,yy)
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
            paths, slope, intercept = sca.scatter_plot(axs, np.array(x), np.array(yy), x_std=x_std, y_std=y_std, markersize=10, eeOn = True, ee_line1 = False, ee_line2 = True, msize=8, fcolor='k', mcolor='k', fsize=9, alpha=1, lcolor='r', label_p='upper left' )
            axs.errorbar(np.array(x), np.array(yy), xerr =x_std, yerr =yy_std, fmt='o',marker='o', ecolor='k',
                          mec='k', mfc='w', markersize=4, mew=0.5, elinewidth=0.5, ls='',capsize = 2,
                          label='test', zorder=2)
            vmin = 0
            vmax = 3.5
            axs.set_ylabel( str(sat_title[isat])+' AOD 680 nm' )
            axs.set_xlabel( 'AERONET AOD 675 nm' )
            axs.set_title( '20'+str(yymmdd)[:2]+'-'+str(yymmdd)[2:4]+'-'+str(yymmdd)[4:6] )
            axs.set_xticks(np.arange(np.ceil(vmin),np.floor(vmax)+1,0.5))
            axs.set_yticks(np.arange(np.ceil(vmin),np.floor(vmax)+1,0.5))
            axs.set_ylim([vmin, vmax])
            axs.set_xlim([vmin, vmax])

            plt.tight_layout()
            plt.savefig('./img_v3/'+str(vsat)+'_'+str(yymmdd)+'.png',dpi=200 )
