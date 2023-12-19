from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import glob
import numpy as np
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from math import fsum, exp
from scipy import stats
from random import uniform
from matplotlib import cm
from matplotlib.patches import BoxStyle
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import h5py
from scipy.interpolate import interp1d

import sys
sys.path.append('../GEMS_lib/')
import scatter_plot_rimiee_new as sca
import cal_l2_APro as call2

### functions
def get_extinction(hdf_file_name,
               domain=[-180, -90, 180, 90], QC=True,
               max_cldfrac=20, prov_data=False, b1064=False):
    
    '''
    :param hdf_file_name: name of HDF file containing calipso sds
    :type hdf_file_name: string
    :returns, f_ext, f_lon, f_lat: 
    WHERE
        :f_ext: 2D array with the filtered extinction coefficient for the interest domain 
        :f_lon: 2D array with longitude values for the interest domain 
        :f_lat: 2D array with latitude values for the interest domain
    '''
    f   = SD( hdf_file_name, SDC.READ )
    if b1064:
        variables = [ 'Extinction_Coefficient_1064', 'Backscatter_Coefficient_1064', 'Extinction_QC_Flag_1064' ]
    else: 
        variables = ['Extinction_Coefficient_532', 'Total_Backscatter_Coefficient_532','Extinction_QC_Flag_532']
    variables += ['Cloud_Layer_Fraction', 'Longitude', 'Latitude', 'Surface_Elevation_Statistics', 
                  'Column_Optical_Depth_Cloud_532','Profile_UTC_Time']
    if prov_data:
        variables.append( 'Column_Optical_Depth_Aerosols_532' ) 
    else:
        variables.append( 'Column_Optical_Depth_Tropospheric_Aerosols_532' ) 
    varall = []
    fillv  = []
    for variable in variables:
        sds_obj = f.select( variable )
        for key, value in sds_obj.attributes().items():
            if key == 'fillvalue':
                fillv  = value

        var1    = sds_obj.get()
        var     = var1 * 1.0
        if fillv: 
            var[ (var1 == fillv) ] = np.nan
        varall.append( var )
    ext = varall[0]
    bsc = varall[1]
    qc  = varall[2][:,:,0]
    cld = varall[3]
    lon = varall[4][:,1]
    lat = varall[5][:,1]
    alt = varall[6][:,2]
    cod = varall[7]
    time= varall[8]
    aod = varall[9]

    print(ext.shape, qc.shape)
    #-- mask data with low quality
    if QC:
        ext[qc > 16] = np.nan
        bsc[qc > 16] = np.nan

    if domain is not None:

        #-- filter data using the interest domain
        if ( domain[0] < domain[2] ):
            ll_sub_l = np.where((lat >= domain[1]) & (lat <= domain[3]) & \
                             (lon >= domain[0]) & (lon <= domain[2]))[0]
        else:
            ll_sub_l = np.where(((lat >= domain[1]) & (lat <= domain[3])) & \
                                ((lon >= domain[0]) | (lon <= domain[2])))[0]  
        f_ext = ext[ll_sub_l]
        f_bsc = bsc[ll_sub_l]
        f_cld = cld[ll_sub_l]
        f_lon = lon[ll_sub_l]
        f_lat = lat[ll_sub_l]
        f_alt = alt[ll_sub_l]
        f_cod = cod[ll_sub_l]	
        f_aod = aod[ll_sub_l]
        f_time = time[ll_sub_l]

        # -- take care of values greater than 1 than 0
        f_ext[ f_ext < 0       ] = np.nan 
        f_ext[ f_cld > max_cldfrac ] = -999

        f_bsc[ f_bsc < 0       ] = np.nan
        f_bsc[ f_cld > max_cldfrac ] = -999

        outdata = dict(ext=f_ext, cod=f_cod, aod=f_aod, bsc=f_bsc, lon=f_lon, lat=f_lat, alt=f_alt, time=f_time)

    else:

        ext[ ext < 0       ] = np.nan
        ext[ cld > max_cldfrac ] = -999

        bsc[ bsc < 0       ] = np.nan
        bsc[ cld > max_cldfrac ] = -999

        outdata = dict(ext=ext, cod=cod, aod=aod, bsc=bsc, lon=lon, lat=lat, alt=alt, time=time)

    return outdata

def l2_hgrid(top=399, bottom=0):
    '''
    return a height grid for caliop level-2 aerosol profile data
    '''
    #
    hgrid = np.zeros(top-bottom+1)
    for i in np.arange(bottom,top+1):
        if ( i < 145  ):
            hgrid[i] = -0.5 + 0.06 * i
        elif( i < 345 ):
            hgrid[i] = 8.2 + 0.06 * (i-145)
        elif( i <=399 ):
            hgrid[i] = 20.2 + 0.18 * (i-345)

    return hgrid

def l2_hcenter(top=399, bottom=0):

    hgrid = l2_hgrid(top=top, bottom=bottom)
    hcenter = ( hgrid[1:] + hgrid[:-1] ) / 2.0

    return hcenter


def narrowDomain_epic(data, domain, var):
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

    uppLat = domain[3] ; lowLat = domain[1]
    lefLon = domain[0] ; rigLon = domain[2]
    result = np.where((lat >= lowLat ) & (lat <= uppLat ) & \
                      (lon >= lefLon ) & (lon <= rigLon ))
    if len(result[0]) == 0:
        return 0
    H_min = np.min(result[0])-10
    H_max = np.max(result[0])+10
    V_min = np.min(result[1])-10
    V_max = np.max(result[1])+10
    if H_min <0:
        H_min = 0

    for key in var:
        data['idx'] = [H_min,(H_max+1),V_min,(V_max+1)]
        data[key] = data[key][H_min:(H_max+1),V_min:(V_max+1)]
        data['latitude'] = data['latitude'][H_min:(H_max+1),V_min:(V_max+1)]
        data['longitude'] = data['longitude'][H_min:(H_max+1),V_min:(V_max+1)]

    return data

#####
def weighted_alh(cal_ec, bsc=False, maxH=10., do_fill_blank=False):
    # define variables
    nfoot = np.size(cal_ec['lon'])
    cal_h = np.empty(nfoot)
    cal_h[:] = np.nan

    # height grids
    zcenter = call2.l2_hcenter()[::-1]

    # start a foot loop
    for i in range(nfoot):

        if bsc:
            ext_i = cal_ec['bsc'][i,:]
            minv  = 0.000001
        else:
            ext_i = cal_ec['ext'][i,:]
            minv  = 0.00001

        #- fill the blank ext values 
        if do_fill_blank:
            ext_i = mask_blank_layers( ext_i, cal_ec['alt'][i],  maxH=maxH )

    # do aerosol/cloud tests
    nl_cld = np.sum( (ext_i < 0.0) )
    nl_aer = np.sum( (ext_i > minv) )

    if ( cal_ec['cod'][i]<=0.02 and cal_ec['aod'][i]>=0.2) and ( nl_aer >= 10 ): 
        xindex = np.logical_and(ext_i > minv, zcenter <= maxH)
        agg_h  = np.sum( zcenter[xindex] * ext_i[xindex] )
        agg_x  = np.sum( ext_i[xindex] )
        cal_h[i] = agg_h / agg_x
    return cal_h

#####
def  mask_blank_layers(ext1d, alt0d, colaod=0.07, scale_h=2., critical_ext=None, maxH=15):

    #-- calculate bacground profile
    hgrid = call2.l2_hgrid()[::-1]
    aod   = np.zeros( len(hgrid) )
    for j in range( len(hgrid) ):
        aod [j] = colaod*np.exp(-hgrid[j]/scale_h)

    background_ext = (aod[1:]-aod[:-1])/(hgrid[:-1]-hgrid[1:])

    #-- center altitude 
    hcent = call2.l2_hcenter()[::-1]

    filled_ext1d = ext1d + 0.0 
    fill_blow = False
    nlayer = np.size(ext1d)
    for j in np.arange(nlayer):
        if hcent[j] < maxH :
            if critical_ext is None:
                fill_blow = True
            else:
                if ext1d[j]>critical_ext:
                    fill_blow = True

        if fill_blow and not (ext1d[j]<0):
            filled_ext1d[j] = np.nanmax( [ext1d[j], background_ext[j]] )
        if hcent[j] < alt0d:
            filled_ext1d[j] = np.nan

    return filled_ext1d


def get_passive_on_caliop_irr_new(h, uvai_h, cal_lat, cal_lon, cal_alt, p_lat, p_lon, radi, correct_surfh=True):
    '''
    Function to calculate mean and std for passive sat alh & uvai through CALIOP track
    h: [array-like], passive sat alh
    cal_lat: [1d-array], latitude of CALIOP
    cal_lon: [1d-array], longitude of CALIOP
    p_lat: [array-like], passive sat lat
    p_lon: [array-like], passive sat lon
    radi: radius for covering near caliop lat, lon
    '''
    # mask out outer regions->is this really needed here?
    maskLon = np.logical_or( p_lon < np.nanmin(cal_lon),
                p_lon > np.nanmax(cal_lon) )
    maskLat = np.logical_or( p_lat < np.nanmin(cal_lat),
                p_lat > np.nanmax(cal_lat) )
    amask   = np.logical_or( maskLat, maskLon )
    ### 

    ixs   = []
    iys   = []
    icals = []
    c_lon   = []
    c_lat   = []
    alh_h = []
    alh_sh = []
    uvai = []

    for i in range(len(cal_lon)):
        dis = np.sqrt((cal_lat[i] - p_lat)**2 + (cal_lon[i] - p_lon)**2)
        i0,j0 = np.where(dis<radi)
        h_0 = h[i0,j0]
        h_0 = np.ma.masked_invalid(h_0)
        uvai_test = uvai_h[i0,j0]
        ixs.append(i0)
        iys.append(j0)
        c_lat.append(cal_lat[i])
        c_lon.append(cal_lon[i])

        if np.ma.count(h_0) / np.ma.count(dis[i0,j0]) > 0.3:
            if (np.ma.std(h_0) < 1):
                icals.append(i)
                alh_sh.append(np.nanstd(h_0))
                if correct_surf == True:
                    alh_h.append(np.ma.mean( h_0 ) + np.nanmean( cal_alt[icals[i]-2:icals[i]+3] ))
                else:
                    alh_h.append(np.nanmean( h_0 ))

                if np.ma.count(uvai_test) > 2:
                    uvai.append(np.nanmean(uvai_test))
                else:
                    uvai.append(np.nan)
    icals = np.array(icals)
    return alh_h, alh_sh, icals, c_lat, c_lon, uvai



### MAIN CODE STARTS FROM HERE ###

plot_all_scatter = False #True
idfile_list = [210328,210426,220410,230310,230519,230520,210331,210810,210811,220409,230326,230417]
tt_hr_list = ['05','05','06','06','06','06','07','06','05','07','07','08']

dust_idfile_list = [210328,210426,220410,230310,230519,230520]
smoke_idfile_list = [210331,210810,210811,220409,230326,230417]

domain_list = [[100,30,135,50],[95,35,120,50],[90,13,110,22],[90,17,110,26]]
epic_td = 1.5
trop_td = 1
qc = False

for idf_ind, idfile in enumerate(idfile_list):
    #-- CALIOP FILE
    tt_hr = tt_hr_list[idf_ind]
    if idfile in dust_idfile_list:
        domain = domain_list[0]
        dust = True
    elif idfile == 210810 or idfile == 210811:
        domain = domain_list[1]
        dust = False
    elif idfile == 230326:
        domain = domain_list[3]
        dust = False
    else:
        domain = domain_list[2]
        dust = False
    if str(idfile)[:2] == '21':
        new = False
        prov = False
    else:
        new = True
        prov = True
        
    dir_data = '/Dedicated/jwang-data/shared_satData/CALIOP/L2/20' + str(idfile)[:2] +'/'
    hdf_f = []

    for file in glob.glob(dir_data+'CAL_LID_L2_05kmAPro-*.20'+str(idfile)[:2]+'-'+str(idfile)[2:4]+'-'+str(idfile)[4:6]+'T'+tt_hr+'-**-**ZD.hdf'):
        hdf_f = file
    print('CALIOP file',hdf_f)
    #-- select nearest time based on CALIOP time
    # calculate the average time of the CALIOP file
    test = call2.get_extinction(hdf_f, domain=domain, QC=qc, prov_data=prov, b1064=False)
    t = np.array(test['time'])
    tt = np.zeros(len(t))
    for i in np.arange(len(t)):
        time_test = test['time'][:,1][i]-int(test['time'][:,1][i])
        tt[i] = time_test*24
    tt_mean = np.mean(tt)
    min = tt_mean - int(tt_mean)
    min = 60*min
    avgtime = []
    if int(tt_mean)<10:
        avg_time = str(int(tt_mean)) + str(int(min))
        if int(min)<10:
            min = '0'+str(min)
        avg_time = str(int(tt_mean)) + str(min)
        avg_time = int(float(avg_time))
        avg_time = '0'+ str(avg_time)
        avg_time = avgtime.append(avg_time)

    #-- get info from HDF file
    cal = test
    ncaliop = np.size(cal['lon'])
    #-- get CALIOP's lat, lon
    cal_lon, cal_lat = cal['lon'],cal['lat']
    cal_alt = cal['alt']


    ## PASSIVE SAT
    # GEMS file (for both AEH & UVAI)
    dir_gems = '/Dedicated/jwang-data/shared_satData/GEMS/L2/V2.0/AEH/20'+str(idfile)[:4]+'/'+str(idfile)[4:6]+'/'
    timelist_gems = []
    for name in sorted(glob.glob(dir_gems +'GK2_GEMS_L2_20'+str(idfile)+'_'+'****'+'_AEH_??_DPRO_ORI.nc')):
        file_list = name
        time_list = file_list[-23:-19]
        timelist_gems.append(time_list)
    timelist_trop = []

    # TROPOMI file
    if dust == True:
        dir_trop = '/Dedicated/jwang-data2/xchen/data/tropomi/ALH/Asia/dust/' #dust
        for name in sorted(glob.glob(dir_trop + 'tropomi_'+str(idfile)+'T*DUST.v3_BlueRAZAOD2H0.2_NBAR_NDVI_slp_NoAgg.nc')):#DUST
            file_list = name
            d = dir_trop + 'tropomi_'+str(idfile)+'T*DUST.v3_BlueRAZAOD2H0.2_NBAR_NDVI_slp_NoAgg.nc'
            time_list = file_list[-52:-48] #dust
            timelist_trop.append(time_list)
    else:
        dir_trop = '/Dedicated/jwang-data2/xchen/data/tropomi/ALH/Asia/smoke/' #smoke
        for name in sorted(glob.glob(dir_trop + 'tropomi_'+str(idfile)+'T*SMK.v3SMK.v4_BlueRAZAOD2H0.*_NBAR_NDVI_slp_NoAgg.nc')):#SMOKE
            file_list = name
            d = dir_trop + 'tropomi_'+str(idfile)+'T*SMK.v3SMK.v4_BlueRAZAOD2H0.2_NBAR_NDVI_slp_NoAgg.nc'
            time_list = file_list[-57:-53] #smoke
            timelist_trop.append(time_list)

    # EPIC file
    timelist_epic = []
    if idfile == 210331:
        dir_epic = '/Dedicated/jwang-data2/zhendlu/ALH/global_operational/20'+str(idfile)[:2]+'/'+str(idfile)[2:4]+'/' #smoke
        for name in sorted(glob.glob(dir_epic +'DSCOVR_EPIC_L2_AOCH_*_20'+str(idfile)+'*.nc')): #smoke
            file_list = name
            time_list = file_list[80+13:84+13]
            timelist_epic.append(time_list)  

    elif dust == True and new == False:
        dir_epic = '/Dedicated/jwang-data2/zhendlu/ALH/global_operational/20'+str(idfile)[:2]+'/'+str(idfile)[2:4]+'/' # dust
        for name in sorted(glob.glob(dir_epic + 'epic_AOCH_20'+str(idfile)+'*.DST.v1_UVAI_Asia_Dust_newcali.nc')): #dust
            file_list = name
            time_list = file_list[-42:-38]
            timelist_epic.append(time_list)

    elif new == True:
        dir_epic = '/Dedicated/jwang-data2/zhendlu/ALH/global_operational/20'+str(idfile)[:2]+'/'+str(idfile)[2:4]+'/' #smoke
        for name in sorted(glob.glob(dir_epic +'DSCOVR_EPIC_L2_AOCH_*_20'+str(idfile)+'*.nc')): #smoke
            file_list = name
            time_list = file_list[80+13:84+13]
            timelist_epic.append(time_list)        
    else:
        dir_epic = '/Dedicated/jwang-data2/zhendlu/ALH/global_operational/20'+str(idfile)[:2]+'/'+str(idfile)[2:4]+'/' #smoke
        for name in sorted(glob.glob(dir_epic +'DSCOVR_EPIC_L2_AOCH_01_20'+str(idfile)+'*.nc')): #smoke
            file_list = name
            time_list = file_list[-24:-20]
            timelist_epic.append(time_list)

    timelist = {}
    timelist['gems'] = timelist_gems
    timelist['trop'] = timelist_trop
    timelist['epic'] = timelist_epic
    sat = ['gems','trop', 'epic']
    near_times = []
    for isat, vsat in enumerate(sat):
        tlist = []
        for i in range(len(timelist[vsat])):
            tlist.append(timelist[vsat][i])
        avgt = avgtime[0]
        min_diff_list =[]
        for i in np.arange(len(tlist)):
            hh = float(str(tlist[i])[:2])*60
            mm = float(str(tlist[i])[2:])
            hhmm = hh+mm
            avgt_hh = float(avgt[:2]) * 60
            avgt_mm = float(avgt[2:])
            avgt_hhmm = avgt_hh+avgt_mm
            diff = np.abs(avgt_hhmm - hhmm)
            min_diff = np.min(diff)
            min_diff = min_diff_list.append(min_diff)
        min_index = min_diff_list.index(np.min(min_diff_list))
        nearest_time = tlist[min_index]
        near_times.append(nearest_time)

    for name in glob.glob(dir_gems+'GK2_GEMS_L2_20'+str(idfile)+'_'+near_times[0]+'_AEH_??_DPRO_ORI.nc'):
        print('GEMS AEH file',name)
    gems_file = Dataset(name, 'r')

    # GEMS UVAI -----------
    dir_gems_UVAI = '/Dedicated/jwang-data/shared_satData/GEMS/L2/V2.0/AERAOD/20'+str(idfile)[:4]+'/'+str(idfile)[4:6]+'/'
    for name in glob.glob(dir_gems_UVAI+'GK2_GEMS_L2_20'+str(idfile)+'_'+near_times[0]+'_AERAOD_??_DPRO_ORI.nc'):
        print('GEMS AOD file',name)
    gems_UVAI_file = Dataset(name, 'r')
    #----------------------
    if dust == True:
        for name in glob.glob(dir_trop + 'tropomi_'+str(idfile)+'T'+near_times[1]+'*DUST.v3_BlueRAZAOD2H0.2_NBAR_NDVI_slp_NoAgg.nc'):
            print(name)

    else:
        for name in glob.glob(dir_trop + 'tropomi_'+str(idfile)+'T'+near_times[1]+'*SMK.v3SMK.v4_BlueRAZAOD2H0.*_NBAR_NDVI_slp_NoAgg.nc'):
            print(name)

    if idfile == 210331:
        name = dir_trop + 'tropomi_210331T053324SMK.v3SMK.v4_BlueRAZAOD2H0.1_NBAR_NDVI_slp_NoAgg.nc' #### NEW
        near_times[1] = timelist_trop[0]
    if idfile == 210426:
        name = dir_trop + 'tropomi_210426T040718DUST.v3_BlueRAZAOD2H0.2_NBAR_NDVI_slp_NoAgg.nc' #### NEW
        near_times[1] = timelist_trop[0]
    if idfile == 230310:
        name = dir_trop + 'tropomi_230310T035719DUST.v3_BlueRAZAOD2H0.2_NBAR_NDVI_slp_NoAgg.nc'
        near_times[1] = timelist_trop[1]
    if idfile == 230519:
        name = dir_trop + 'tropomi_230519T035316DUST.v3_BlueRAZAOD2H0.2_NBAR_NDVI_slp_NoAgg.nc'
        near_times[1] = timelist_trop[1]
    if idfile == 210811:
        name = dir_trop + 'tropomi_210811T040105SMK.v3SMK.v4_BlueRAZAOD2H0.2_NBAR_NDVI_slp_NoAgg.nc'
        near_times[1] = timelist_trop[0]
    if idfile == 220409:
        name = dir_trop + 'tropomi_220409T052221SMK.v3SMK.v4_BlueRAZAOD2H0.1_NBAR_NDVI_slp_NoAgg.nc'
        near_times[1] = timelist_trop[0]
    if idfile == 230417:
        name = dir_trop + 'tropomi_230417T053138SMK.v3SMK.v4_BlueRAZAOD2H0.1_NBAR_NDVI_slp_NoAgg.nc'
        near_times[1] = timelist_trop[0]

    trop_file = Dataset(name, 'r')

    if idfile == 210331:
        for name in glob.glob(dir_epic + 'DSCOVR_EPIC_L2_AOCH_01_20'+str(idfile)+near_times[2]+'*.nc'):
            print(name)
    elif dust == True and new == False:
        for name in glob.glob(dir_epic + 'epic_AOCH_20'+str(idfile)+near_times[2]+'*.DST.v1_UVAI_Asia_Dust_newcali.nc'):
            print(name)
    else:
        for name in glob.glob(dir_epic + 'DSCOVR_EPIC_L2_AOCH_01_20'+str(idfile)+near_times[2]+'*.nc'):
            print(name)
            
    epic_file = Dataset(name, 'r')


    #-- CUT EPIC data
    epic_dict = dict()
    epic_dict['latitude'] = epic_file['lat'][:]
    epic_dict['longitude'] = epic_file['lon'][:]

    if new == True :
        epic_dict['AOD'] = epic_file['AOD'][:]
        epic_dict['height'] = epic_file['AOCH'][:]
    else:
        epic_dict['AOD'] = epic_file['aod'][:]
        epic_dict['height'] = epic_file['height'][:]
    epic_dict['UVAI'] = epic_file['UVAI'][:]
    epic_narrow = narrowDomain_epic(epic_dict, domain, ['height','UVAI','AOD'])
    test = epic_narrow

    #-- data into dictionary & masking
    sat_f = [gems_file, trop_file, epic_file]

    lats = dict()
    lons = dict()
    alh = dict()
    uvai = dict()
    aod = dict()

    for isat, vsat in enumerate(sat):
        # load LUT to convert ALH definition
        npzfile = np.load('./LUT_caliop_aoch_diff2.npz',allow_pickle=True)

        if vsat == 'epic':
            lats[vsat] = epic_file['lat'][test['idx'][0]:test['idx'][1],test['idx'][2]:test['idx'][3]]
            lons[vsat] = epic_file['lon'][:][test['idx'][0]:test['idx'][1],test['idx'][2]:test['idx'][3]]
            alh[vsat] = epic_narrow['height'][:]
            aod[vsat] = epic_narrow['AOD'][:]
            uvai[vsat] = epic_narrow['UVAI'][:]

            # convert EPIC/TROPOMI AOCH to CALIOP AOCH def
            npz_aoch = npzfile['aoch']
            npz_cal = npzfile['caliop']
            interpolator = interp1d(npz_aoch, npz_cal, kind='linear',fill_value='extrapolate')
            for row in range(alh[vsat].shape[0]):
                for col in range(alh[vsat].shape[1]):
                    if np.isnan(alh[vsat][row,col]) == False:
                        alh[vsat][row,col] = interpolator(alh[vsat][row,col])
                    else:
                        alh[vsat][row,col] = np.nan
            
        if vsat == 'gems':
            lons[vsat] = gems_file['Geolocation Fields']['Longitude'][:]
            lats[vsat] = gems_file['Geolocation Fields']['Latitude'][:]
            aeh, flag = gems_file['Data Fields']['AerosolEffectiveHeight'], gems_file['Data Fields']['QualityFlag']
            aeh = np.array(aeh)
            flag = np.array(flag)
            variable = gems_UVAI_file['Data Fields']['UVAerosolIndex'][:]
            uvai[vsat] = variable
            
            # convert GEMS AEH to CALIOP AOCH def
            npz_aeh = npzfile['aeh']
            npz_cal = npzfile['caliop']
            
            interpolator = interp1d(npz_aeh, npz_cal, kind='linear',fill_value='extrapolate')
            aeh_to_aoch = np.zeros(aeh.shape)
            aeh_to_aoch = interpolator(aeh)
            masked_aeh = np.ma.masked_where(flag > 0, aeh_to_aoch)
            alh[vsat] = masked_aeh

        if vsat == 'trop':

            lats[vsat] = sat_f[isat]['lat'][:]
            lons[vsat] = sat_f[isat]['lon'][:]
            alh[vsat] = sat_f[isat]['height'][:,:,0]
            uvai[vsat] = sat_f[isat]['UVAI'][:]
            aod[vsat] = sat_f[isat]['aod'][:]
            
            npz_aoch = npzfile['aoch']
            npz_cal = npzfile['caliop']
            interpolator = interp1d(npz_aoch, npz_cal, kind='linear',fill_value='extrapolate')
            for row in range(alh[vsat].shape[0]):
                for col in range(alh[vsat].shape[1]):
                    if np.isnan(alh[vsat][row,col]) == False:
                        alh[vsat][row,col] = interpolator(alh[vsat][row,col])
                    else:
                        alh[vsat][row,col] = np.nan
            
    # mask EPIC with UVAI, ALH> 9
    aod['epic'] = np.where(aod['epic'] < 6, aod['epic'], np.nan)
    aod['epic'] = np.where(aod['epic'] > 0.2, aod['epic'], np.nan)
    alh['epic'] = np.where(uvai['epic'] > epic_td, alh['epic'], np.nan)
    alh['epic'] = np.where(alh['epic'] < 9, alh['epic'], np.nan)
    alh['epic'] = np.where(aod['epic'] > 0.2, alh['epic'], np.nan)
    alh['epic'] = np.where(aod['epic'] < 6, alh['epic'], np.nan)
    alh['trop'] = np.where(uvai['trop'] > trop_td, alh['trop'], np.nan)
    aod['trop'] = np.where(aod['trop'] < 6, aod['trop'], np.nan)
    aod['trop'] = np.where(aod['trop'] > 0.2, aod['trop'], np.nan)


    #### PLOT VERTICAL PROFILE FIGURE

    fig = plt.figure(figsize=(5.5,2.5))
    cmap    = plt.cm.rainbow
    y_label = 'Altitude (km)'
    title = "    Comparison with CALIOP 532-nm Extinction Profile"
    ax, xdata = call2.plot_calipso(cal, fig, inset=False,
                                   ymax=14, cmap=cmap,
                                   y_label=y_label, title=title,
                                   domain=domain, bsc=False, b1064=False,use_under=True)

    cal_h_filled = weighted_alh(cal, bsc=False, maxH=10.5, do_fill_blank=True)
    label_cal = 'CALIOP 20'+str(idfile)[0:2]+'-'+str(idfile)[2:4]+'-'+str(idfile)[4:6]+' '+str(avgtime[0][0:2])+':'+str(avgtime[0][2:4])

    cal_h_filled_new = np.ones((cal_h_filled.shape))
    for i in range(len(cal_h_filled)-2):
        cal_h_filled_new[0] = np.nan
        cal_h_filled_new[1] = np.nan
        cal_h_filled_new[-1] = np.nan
        cal_h_filled_new[-2] = np.nan
        #cal_h_filled_new[i] = np.nanmean(cal_h_filled[i-2:i+3])
        cal_h_filled_new[i] = np.nanmean(cal_h_filled[i-1:i+2])


    ax.plot(xdata, cal_h_filled_new, 'k', linewidth=2.0, label=label_cal)
    title = ['GEMS AEH 20','TROPOMI AOCH 20','EPIC AOCH 20']
    epic_all_test = []
    #epic_all = []
    colors= ['orange','limegreen','mediumpurple']

    maxdis = [0.2, 0.2, 0.2] #[0.1, 0.2, 0.2] #radi = [0.05, 0.1, 0.2]
    correct_surf = [False,True,True]
    h = dict()
    sh = dict()
    lat = dict()
    lon = dict()
    icals = dict()
    c_lat = dict()
    c_lon = dict()
    UVAI_test = dict()
    isepic = [False,False,True]
    sat = ['gems','trop','epic']
    correct = [False, True, True]

    for isat, vsat in enumerate(sat):
        h[vsat], sh[vsat], icals[vsat], c_lat[vsat], c_lon[vsat], UVAI_test[vsat] = get_passive_on_caliop_irr_new(alh[vsat],
                                        uvai[vsat], cal_lat, cal_lon, cal_h_filled_new, lats[vsat],lons[vsat],radi = maxdis[isat],correct_surfh=correct[isat])
        post_arr = dict()
        post_arr['h'] = np.array(h[vsat])
        post_arr['sh'] = np.array(sh[vsat])
        post_arr['icals'] = np.array( icals[vsat])
        post_arr['lat'] = np.array(c_lat[vsat])
        post_arr['lon'] = np.array(c_lon[vsat])
        post_arr['uvai'] = np.array(UVAI_test[vsat])   
        
        label_h = title[isat]+str(idfile)[0:2]+'-'+str(idfile)[2:4]+'-'+str(idfile)[4:6]+' '+str(near_times[isat][0:2])+':'+str(near_times[isat][2:4])
        if vsat == 'gems':
            if post_arr['h'] !=[]:
                ax.errorbar(xdata[icals[vsat]], h[vsat], yerr=sh[vsat], fmt='o-', linewidth=0.7,color=colors[isat], markersize=0.3, label=label_h)

        if vsat == 'trop':
            if int(near_times[isat][2:4])>5:
                label_h = title[isat]+str(idfile)[0:2]+'-'+str(idfile)[2:4]+'-'+str(idfile)[4:6]+' 0'+str(int(near_times[isat][0:2])+1)+':'+str(int(near_times[isat][2:4])-5)
            else:
                label_h = title[isat]+str(idfile)[0:2]+'-'+str(idfile)[2:4]+'-'+str(idfile)[4:6]+' 0'+str(int(near_times[isat][0:2]))+':'+str(int(near_times[isat][2:4])+55)
            if post_arr['h'] !=[]:

                ax.errorbar(xdata[post_arr['icals']], post_arr['h'], yerr=post_arr['sh'], fmt='^-', linewidth=0.7,color=colors[isat], markersize=0.3, label=label_h)

        if vsat == 'epic':
            if post_arr['h'] !=[]:

                ax.errorbar(xdata[post_arr['icals']], post_arr['h'], yerr=post_arr['sh'], fmt='s-', linewidth=0.7,color=colors[isat], markersize=0.3, label=label_h)

        epic_all_test.append(post_arr)
    ax.legend(loc='upper right')
    plt.savefig('./img_v5/'+str(idfile)+'_ALH_vertical_prof_QC_'+str(qc), bbox_inches='tight', dpi=300)
    plt.close(fig)

    prefix = 'data_v5/all/' +str(idfile)+'_'+str(tt_hr)
    savefile = prefix +'_slp2.npz'
    for isat, vsat in enumerate(sat):
        print(vsat)
        np.savez(savefile+'_'+vsat, cal_h=cal_h_filled, h0_h=epic_all_test[isat]['h'], h0_sh=epic_all_test[isat]['sh'],h0_icals=epic_all_test[isat]['icals'], \
        h0_c_lat=epic_all_test[isat]['lat'],h0_c_lon=epic_all_test[isat]['lon'],uvai_mean=epic_all_test[isat]['uvai'])
    if dust == True:
        prefix = 'data_v5/dust/' +str(idfile)+'_'+str(tt_hr)
        savefile = prefix +'_slp2.npz'
        for isat, vsat in enumerate(sat):
            np.savez(savefile+'_'+vsat, cal_h=cal_h_filled, h0_h=epic_all_test[isat]['h'], h0_sh=epic_all_test[isat]['sh'],h0_icals=epic_all_test[isat]['icals'], \
            h0_c_lat=epic_all_test[isat]['lat'],h0_c_lon=epic_all_test[isat]['lon'],uvai_mean=epic_all_test[isat]['uvai'])
            
    else:
        prefix = 'data_v5/smoke/' +str(idfile)+'_'+str(tt_hr)
        savefile = prefix +'_slp2.npz'
        for isat, vsat in enumerate(sat):
            np.savez(savefile+'_'+vsat, cal_h=cal_h_filled, h0_h=epic_all_test[isat]['h'], h0_sh=epic_all_test[isat]['sh'],h0_icals=epic_all_test[isat]['icals'], \
            h0_c_lat=epic_all_test[isat]['lat'],h0_c_lon=epic_all_test[isat]['lon'],uvai_mean=epic_all_test[isat]['uvai'])
            
    
########################### scatter all###########################
if plot_all_scatter == True:

    sat = ['gems','trop','epic']
    dates = idfile_list

    all_test = dict()

    cal_h = []
    h0_h = []
    h0_sh = []
    h0_icals = []
    h0_c_lat = []
    h0_c_lon = []

    cal_h = dict()
    h0_h = dict()
    h0_sh = dict()
    h0_icals = dict()
    h0_c_lat = dict()
    h0_c_lon = dict()

    for isat, vsat in enumerate(sat):
        all_test[vsat]=dict()
        cal_h = []
        h0_h = []
        h0_sh = []
        h0_icals = []
        h0_c_lat = []
        h0_c_lon = []
        uvai_mean = []
        filess = sorted(glob.glob('data_v5/all/*_slp2.npz_'+str(vsat)+'.npz'))
        for ifile,vfile in enumerate(filess):
            files = np.load(vfile)
            cal_h.append(files['cal_h'])
            h0_h.append(files['h0_h'])
            h0_sh.append(files['h0_sh'])
            h0_icals.append(files['h0_icals'])
            h0_c_lat.append(files['h0_c_lat'])
            h0_c_lon.append(files['h0_c_lon'])
            uvai_mean.append(files['uvai_mean'])
            files.close()
        all_test[vsat]['cal_h'] = cal_h
        all_test[vsat]['h0_h'] = h0_h
        all_test[vsat]['h0_sh'] = h0_sh
        all_test[vsat]['uvai'] = uvai_mean
        all_test[vsat]['h0_icals'] = h0_icals
        all_test[vsat]['h0_c_lat'] = h0_c_lat
        all_test[vsat]['h0_c_lon'] = h0_c_lon


    dust_dates = dust_idfile_list
    smoke_dates = smoke_idfile_list

    dust_test = dict()

    keys = ['cal_h', 'h0_h', 'h0_sh', 'h0_icals', 'h0_c_lat', 'h0_c_lon']
    data = {key: {} for key in keys}

    for isat, vsat in enumerate(sat):
        dust_test[vsat]=dict()

        cal_h = []
        h0_h = []
        h0_sh = []
        h0_icals = []
        h0_c_lat = []
        h0_c_lon = []
        uvai_mean = []
        filess = sorted(glob.glob('data_v5/dust/*_slp2.npz_'+str(vsat)+'.npz'))
        for ifile,vfile in enumerate(filess)
            files = np.load(vfile)
            cal_h.append(files['cal_h'])
            h0_h.append(files['h0_h'])
            h0_sh.append(files['h0_sh'])
            h0_icals.append(files['h0_icals'])
            h0_c_lat.append(files['h0_c_lat'])
            h0_c_lon.append(files['h0_c_lon'])
            uvai_mean.append(files['uvai_mean'])
            files.close()
        dust_test[vsat]['cal_h'] = cal_h
        dust_test[vsat]['h0_h'] = h0_h
        dust_test[vsat]['h0_sh'] = h0_sh
        dust_test[vsat]['uvai'] = uvai_mean
        dust_test[vsat]['h0_icals'] = h0_icals
        dust_test[vsat]['h0_c_lat'] = h0_c_lat
        dust_test[vsat]['h0_c_lon'] = h0_c_lon




    smoke_test = dict()

    keys = ['cal_h', 'h0_h', 'h0_sh', 'h0_icals', 'h0_c_lat', 'h0_c_lon']
    data = {key: {} for key in keys}

    for isat, vsat in enumerate(sat):
        smoke_test[vsat]=dict()

        cal_h = []
        h0_h = []
        h0_sh = []
        h0_icals = []
        h0_c_lat = []
        h0_c_lon = []
        uvai_mean = []
        filess = sorted(glob.glob('data_v5/smoke/*_slp2.npz_'+str(vsat)+'.npz'))
        for ifile,vfile in enumerate(filess):
            files = np.load(vfile)
            cal_h.append(files['cal_h'])
            h0_h.append(files['h0_h'])
            h0_sh.append(files['h0_sh'])
            h0_icals.append(files['h0_icals'])
            h0_c_lat.append(files['h0_c_lat'])
            h0_c_lon.append(files['h0_c_lon'])
            uvai_mean.append(files['uvai_mean'])
            files.close()
        smoke_test[vsat]['cal_h'] = cal_h
        smoke_test[vsat]['h0_h'] = h0_h
        smoke_test[vsat]['h0_sh'] = h0_sh
        smoke_test[vsat]['uvai'] = uvai_mean
        smoke_test[vsat]['h0_icals'] = h0_icals
        smoke_test[vsat]['h0_c_lat'] = h0_c_lat
        smoke_test[vsat]['h0_c_lon'] = h0_c_lon


    label = ['GEMS','TROPOMI','EPIC']
    colors= ['orange','limegreen','mediumpurple']

    #plot scatter plot
    fig = plt.figure(figsize=(3,3))
    ntot    = []
    nhalfkm = []
    nonekm  = []
    y0 = [[2.7,2.2,1.7],[1.2,0.7,0.2]]
    y1 = [0.8,0.25]
    textx = [0,0,0]
    texty = [0.95, 0.95, 0.95]
    textx0 = [1,1,1]
    texty0 = [0.62, 0.62, 0.62]


    boxstyle = BoxStyle("Round", pad=0.3)
    props = {'boxstyle': boxstyle,
             'facecolor': 'white',
             'linestyle': 'solid',
             'linewidth': 0.5,
             'alpha': 0.9}
    fontsize = 7

    for iepic in range(len(all_test)):
        vsat = sat[iepic]
        xdata = []
        ydata = []
        zdata = []
        xerr  = []
        yerr  = []

        for j in range(len(idfile_list)):
            for i in range(len(all_test[vsat]['h0_icals'][j])):
                ic = all_test[vsat]['h0_icals'][j][i]
                usex = all_test[vsat]['cal_h'][j]
                if all_test[vsat]['h0_h'][j][i] > 0 and usex[ic] > 0:
                    xdata.append(usex[ic])
                    xerr.append(np.nanstd(usex[ic-2:ic+3]))
                    ydata.append(all_test[vsat]['h0_h'][j][i])
                    yerr.append(all_test[vsat]['h0_sh'][j][i])

        xdata = np.array(xdata)
        ydata = np.array(ydata)
        xyerr = np.abs( ydata - xdata )
        sorterr = np.sort(xyerr)
        if np.size(xdata) < 2 or np.size(ydata) <2 :
            uncert = 0
            ntot.append(0)
            nhalfkm.append(0)
            nonekm.append(0)
        else:
            uncert = sorterr[int(np.size(xdata) * 0.68)]
            plt.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='o', linewidth=0.1,ecolor=colors[iepic], mfc=colors[iepic],mew=0.1, mec='k', color=colors[iepic], alpha=0.8, markersize=2, zorder=2 )

            #-- liner regression
            slope, intercept, correlation, p_value_slope, std_error = stats.linregress(xdata, ydata)
            #-- Calculates a Pearson correlation coefficient and the p-value for testing non-correlation
            r, p_value = stats.pearsonr(xdata, ydata)
            rmse   = np.sqrt(np.mean((ydata-xdata)**2))
            mean_x = np.nanmean(xdata)
            mean_y = np.nanmean(ydata)
            std_x  = np.nanstd(xdata, ddof=1)
            std_y  = np.nanstd(ydata, ddof=1)
            x = np.linspace(0, 10.9, num=1000, endpoint=True)
            y = (slope * x) + intercept
            plt.plot(x, y, '-', color=colors[iepic], linewidth=1.5)
            ntot.append( np.size(xdata) )
            nhalfkm.append( np.size( np.where(abs(xdata-ydata)<0.5)[0] ) )
            nonekm.append( np.size( np.where(abs(xdata-ydata)<1.0)[0] ) )

        #-- create strings for equations in the plot
        correlation_string = "R = {:.2f}".format(r)
        sign = " + "
        if intercept < 0:
            sign = " - "
        lineal_eq = "y = " + "{:.3f}".format(slope) + "x" + sign + "{:.3f}".format(abs(intercept))
        rmse_coef = "RMSE = {:.2f} km".format(rmse)
        bias      = np.nanmean(ydata-xdata)
        bias_coef = "bias = {:.2f} km".format(bias)

        if p_value >= 0.05:
            p_value_s = "(p > 0.05)"
        else:
            if p_value < 0.01:
                p_value_s = "(p < 0.01)"
            else:
                p_value_s = "(p < 0.05)"

        if np.size(xdata) == 0 or np.size(ydata) == 0:
            n_collocations = 'N={:d},EE={:.1f}km'.format(ntot[iepic],uncert)
            x_mean_std = "x: {:.1f}".format(mean_x) + " $\pm$ " + "{:.1f}".format(abs(std_x))
            y_mean_std = "y: {:.1f}".format(mean_y) + " $\pm$ " + "{:.1f}".format(abs(std_y))
            equations0 = label[iepic]+ ' height' + '\n' + \
                    n_collocations
        else:
            n_collocations = 'N={:d},EE={:.1f}km'.format(ntot[iepic],uncert)
            x_mean_std = "x: {:.1f}".format(mean_x) + " $\pm$ " + "{:.1f}".format(abs(std_x))
            y_mean_std = "y: {:.1f}".format(mean_y) + " $\pm$ " + "{:.1f}".format(abs(std_y))
            equations0 = label[iepic]+ ' height' + '\n' + \
                    n_collocations + '\n' + \
                    rmse_coef + '\n' + \
                    lineal_eq  + '\n' + \
                    x_mean_std + '\n' + \
                    y_mean_std + '\n' + \
                    correlation_string + ' ' + p_value_s
                    #bias_coef + '\n' + \
            props = {'boxstyle': boxstyle,
             'facecolor': 'white',
             'linestyle': 'solid',
             'linewidth': 1.5,
             'alpha': 0.9,
             'edgecolor': colors[iepic]}
            
            if len(label) > 2:
                if iepic > 1: # iepic < 2
                    plt.annotate(equations0, xy=(0,0), xytext=(0.05, 0.95), textcoords='axes fraction',va='top', bbox=props, fontsize=7, color='k')
                if iepic == 1: # iepic < 2
                    # lower right
                    posXY0      = (textx[iepic], texty[iepic])
                    posXY_text0 = (160, -150)
                    plt.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='bottom',ha='right',bbox=props, \
                            xycoords='axes fraction', textcoords='offset points', fontsize=7, color='k')
                if iepic < 1:
                    posXY0      = (textx0[iepic], texty0[iepic])
                    posXY_text0 = (-4, -5)
                    plt.annotate(equations0, xy=(0,0), xytext=(0.95, 0.95), textcoords='axes fraction',va='top', ha='right', bbox=props,\
                            xycoords='axes fraction', fontsize=7, color='k')     
            if uncert <= 2.0:
                plt.plot([uncert,10.9],[0,10.9-uncert], c=colors[iepic], linestyle='--', linewidth=0.8)
                plt.plot([0,10.9-uncert],[uncert,10.9], c=colors[iepic], linestyle='--', linewidth=0.8)


    plt.xlim([0,10.9])
    plt.ylim([0,10.9])
    plt.yticks([0,2,4,6,8,10],fontsize=8)
    plt.xticks([0,2,4,6,8,10],fontsize=8)
    plt.title('(a) Passive vs CALIOP ALH', fontsize=11)
    plt.xlabel('CALIOP 532 nm AOCH (km)', fontsize=8)
    plt.ylabel('ALH of passive sensors (km)', fontsize=8)
    plt.plot([0,10.9],[0,10.9], 'k-', zorder=1)
    plt.savefig('./img_v5/all_scatter_noqc.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
