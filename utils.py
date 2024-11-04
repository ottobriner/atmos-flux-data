import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import Point, LineString
from geographiclib.geodesic import Geodesic
from matplotlib import pyplot as plt
# import pywt
from functools import partial, reduce
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

def read_l3(filepath, datecol='DATE'):
    df = (pd.read_csv(filepath, header=0, low_memory=False)
      .replace(-9999, np.nan).replace(99999, np.nan)
      .assign(date = lambda x: pd.to_datetime(x[datecol]))
      .set_index('date', drop=False)
     )
    df.index = df.index.set_names(None)
    return df

def plot_diag(df_in):
    df = (
        df_in.replace(-9999, np.nan)
         )
    print(df.describe())
    axs = df.plot(subplots=True)
    # print([ax.get_ylim() for ax in axs])
    for n, col in enumerate(df.columns):
        axs[n].fill_between(df.index, axs[n].get_ylim()[0], axs[n].get_ylim()[1], where=df[col].isna(), 
                            facecolor='red', alpha=0.3)
    plt.show()
    return axs

def save_ameriflux(filepath, save_as=None, save_file=True):
    
    if save_as is None:
        save_as = filepath[:-4] + '_fluxnet.csv'
    
    df = (pd.read_csv(filepath, header=0, low_memory=False)
          .replace(-9999, np.nan).replace(99999, np.nan) # set missing to np.nan for unit conversions
          .rename(columns = {
              'co2_mole_fraction': 'CO2', # umol mol-1
              'co2_mixing_ratio': 'CO2_MIXING_RATIO', # umol mol-1
              'ch4_mole_fraction': 'CH4', # umol mol-1 (convert to nmol below)
              'ch4_mixing_ratio': 'CH4_MIXING_RATIO', # umol mol-1 (convert to nmol below)
              'co2_flux': 'FC', # umol m-2 s-1
              'ch4_flux': 'FCH4', # umol m-2 s-1 (convert to nmol below)
              'h2o_mole_fraction': 'H2O', # mmol mol-1
              'h2o_mixing_ratio': 'H2O_MIXING_RATIO', # mmol mol-1
              'h2o_flux': 'FH2O', # mmol m-2 s-1
              'co2_strg': 'SC', # umol s-1 m-2
              'ch4_strg': 'SCH4', # umol s-1 m-2 (convert to nmol below)
              'H_strg': 'SH', # W m-2
              'LE_strg': 'SLE', # W m-2
              'air_pressure': 'PA', # Pa (convert to kPa below)
              'sonic_temperature': 'T_SONIC', # K (convert to degC below)
              'air_temperature': 'TA', # K (convert to degC below)
              'Precip': 'P', # mm
              'RN': 'NETRAD', # W m-2 (is met tower giving W m-2?)
              'RS': 'SW_IN', # W m-2
              'L': 'MO_LENGTH', # m
              'qc_Tau': 'TAU_SSITC_TEST', 
              'un_Tau': 'TAU_UNCORR',
              'Tau': 'TAU', # kg m-2 s-1
              'u.': 'USTAR',
              'wind_dir': 'WD',
              'wind_speed': 'WS',
              'max_wind_speed': 'WS_MAX',
              'X.z.d..L': 'ZL',
              'Reco': 'RECO',
              # 'Tsoil': 'TS', 
              'x_peak': 'FETCH_MAX', # m
              'x_90.': 'FETCH_90', # m
              'x_70.': 'FETCH_70', # m
              # 'x_55.': 'FETCH_55', # m
              # 'x_40.': 'FETCH_40', # m
              'qc_co2_flux': 'FC_SSITC_TEST',
              'un_co2_flux': 'FC_UNCORR',
              'qc_ch4_flux': 'FCH4_SSITC_TEST',
              'un_ch4_flux': 'FCH4_UNCORR',
              'qc_H': 'H_SSITC_TEST',
              'un_H': 'H_UNCORR',
              'qc_LE': 'LE_SSITC_TEST',
              'un_LE': 'LE_UNCORR',
              'qc_Tau': 'TAU_SSITC_TEST',
          }, errors = 'raise')
          .assign(TIMESTAMP_END = lambda x: pd.to_numeric(pd.to_datetime(x['DATE']).dt.strftime('%Y%m%d%H%M'), 
                                                          errors='coerce').fillna(-9999).astype(np.int64),
                  TIMESTAMP_START = lambda x: x['TIMESTAMP_END'].shift(1, fill_value=int(202205241000)).astype(np.int64),
#                   H2O = lambda x: x['H2O'] / 1000, # umol mol-1 to mmol mol-1
                  CH4 = lambda x: x['CH4'] * 1000, # umol mol-1 to nmol mol-1
                  CH4_MIXING_RATIO = lambda x: x['CH4_MIXING_RATIO'] * 1000, # umol mol-1 to nmol mol-1
                  FCH4 = lambda x: x['FCH4'] * 1000, # umol m-2 s-1 to nmol m-2 s-1
                  SCH4 = lambda x: x['SCH4'] * 1000, # umol s-1 m-2 to nmol s-1 m-2
                  PA = lambda x: x['PA'] * 0.001, # Pa to kPa
                  VPD = lambda x: x['VPD'] * 0.01, # Pa to hPa
                  TA = lambda x: x['TA'], # K to degC
                  T_SONIC = lambda x: x['T_SONIC'], # K to degC
                  # RH_1 = lambda x: x['RH_1'].mask(x['RH_1']>250, other=np.nan),
                 )
          # drop extra time vars, unknown RH_1, two 30cm and 50cm soil sensors , and secondary soil sensor vars
          .drop(columns=[
                  'date', 'DATE', 'TIME', 'YEAR', 'datetime', 'time', 'time_local', 'time_local.met',
                  'RH_1',
                  # 'Tsoil_30cm', 'Tsoil_50cm', 'SWC_30cm', 'SWC_50cm', # leaving deeper soil sensors in for now
                  'SWC_rmean', 'SWC_rstd', 'SWC_mean',
                  'Tsoil_rmean', 'Tsoil_rstd', 'Tsoil_rmean_pre', 'Tsoil_rstd_pre', 'Tsoil_mean',
          ], errors='ignore')
          # .replace(to_replace=np.nan, value=np.int64(-9999)) # set np.nan to -9999 for FLUXNET formatting
          
         )
    df.index = df.index.set_names(None)
    # print(df.loc[:, df.columns.str.contains('SWC_')])
    df[df.columns[df.columns.str.contains('SWC_')]] = df[df.columns[df.columns.str.contains('SWC_')]] * 100 # convert SWC to percent 0-100
    
    df.columns = df.columns.str.replace('Reco_', 'RECO_')
    df.columns = df.columns.str.replace('Tsoil', 'TS')
    
    cols_to_move = ['TIMESTAMP_START', 'TIMESTAMP_END']
    df = df[ cols_to_move + [ col for col in df.columns if col not in cols_to_move ] ]
    df = df.replace(np.nan, np.int64(-9999)) # set np.nan to -9999 after all unit conversions for FLUXNET formatting
    # df[df.columns.isna().any()] = df.fillna(np.int64(-9999))
    
    if save_file: 
        if save_as is not None:
            df.to_csv(save_as, index = False)
        else:
            df.to_csv(f'US-AMS_HH_{df['TIMESTAMP_START'].iloc[0]}_{df['TIMESTAMP_START'].iloc[-1]}.csv', index = False)
#     df.to_csv(save_as=f'./output/csv/ATMOS_L3_{datetime.now().strftime("%Y-%m-%d")}_fluxnet.csv', index=False)
    
    return df

def get_datalogger_locs():
    '''Returns GeoDataFrame of datalogger locations and tuple of origin (lat, lon)'''
    
    # filepath = '/home/otto/data/atmos-flux-data/input/ATMOS_SoilSensors_layout.xlsx - Sheet1.csv'
    filepath = r"C:\Users\ottob\data\atmos-flux-data\ATMOS_SoilSensors_layout.xlsx - Sheet1.csv"
    dataloggers = pd.read_csv(filepath, usecols=[0, 1, 2, 3, 4], header=0, 
                              names=['logger', 'serial', 'status', 'lat', 'lon']).dropna()
    dataloggers = gpd.GeoDataFrame(dataloggers, 
                                   geometry=gpd.points_from_xy(dataloggers['lon'], dataloggers['lat']),
                                   crs=4326)
    # dataloggers.plot(column='logger')
    
    geod = Geodesic.WGS84
    
    A = dataloggers.geometry[3]
    B = dataloggers.geometry[0]
    inv = geod.Inverse(A.y,A.x,B.y,B.x)
    azib = inv['azi1']
    print('Azimuth from logger 0 to 3 = ' + str(azib))
    
    A = dataloggers.geometry[10]
    B = dataloggers.geometry[7]
    inv = geod.Inverse(A.y,A.x,B.y,B.x)
    azit = inv['azi1']
    print('Azimuth from logger 7 to 10 = ' + str(azit))
    azi = np.mean([azib, azit])
    
    center = dataloggers.geometry[5] # projecting from center, logger 6
    
    s = 50 - 6.5 # m
    direct = geod.Direct(center.y,center.x,azi,s)
    leftmidpt = (direct['lat2'],direct['lon2'])
    
    direct = geod.Direct(leftmidpt[0],leftmidpt[1],azi-90,s)
    origin = (direct['lat2'],direct['lon2'])
    print('Origin = ' + str(origin))
    return dataloggers, origin

def mra(data, wavelet, level=None, axis=-1, transform='swt',
        mode='periodization'):
    """Forward 1D multiresolution analysis.
    It is a projection onto the wavelet subspaces.
    Parameters
    ----------
    data: array_like
        Input data
    wavelet : Wavelet object or name string
        Wavelet to use
    level : int, optional
        Decomposition level (must be >= 0). If level is None (default) then it
        will be calculated using the `dwt_max_level` function.
    axis: int, optional
        Axis over which to compute the DWT. If not given, the last axis is
        used. Currently only available when ``transform='dwt'``.
    transform : {'dwt', 'swt'}
        Whether to use the DWT or SWT for the transforms.
    mode : str, optional
        Signal extension mode, see `Modes` (default: 'symmetric'). This option
        is only used when transform='dwt'.
    Returns
    -------
    [cAn, {details_level_n}, ... {details_level_1}] : list
        For more information, see the detailed description in `wavedec`
    See Also
    --------
    ``imra``, ``swt``
    Notes
    -----
    This is sometimes referred to as an additive decomposition because the
    inverse transform (``imra``) is just the sum of the coefficient arrays
    [1]_. The decomposition using ``transform='dwt'`` corresponds to section
    2.2 while that using an undecimated transform (``transform='swt'``) is
    described in section 3.2 and appendix A.
    This transform does not share the variance partition property of ``swt``
    with `norm=True`. It does however, result in coefficients that are
    temporally aligned regardless of the symmetry of the wavelet used.
    The redundancy of this transform is ``(level + 1)``.
    References
    ----------
    .. [1] Donald B. Percival and Harold O. Mofjeld. Analysis of Subtidal
        Coastal Sea Level Fluctuations Using Wavelets. Journal of the American
        Statistical Association Vol. 92, No. 439 (Sep., 1997), pp. 868-880.
        https://doi.org/10.2307/2965551
    """
    if transform == 'swt':
        if mode != 'periodization':
            raise ValueError(
                "transform swt only supports mode='periodization'")
        kwargs = dict(wavelet=wavelet, norm=True)
        forward = partial(pywt.swt, level=level, trim_approx=True, **kwargs)
        if axis % data.ndim != data.ndim - 1:
            raise ValueError("swt only supports axis=-1")
        inverse = partial(pywt.iswt, **kwargs)
        is_swt = True
    elif transform == 'dwt':
        kwargs = dict(wavelet=wavelet, mode=mode, axis=axis)
        forward = partial(pywt.wavedec, level=level, **kwargs)
        inverse = partial(pywt.waverec, **kwargs)
        is_swt = False
    else:
        raise ValueError("unrecognized transform: {}".format(transform))

    wav_coeffs = forward(data)

    mra_coeffs = []
    nc = len(wav_coeffs)

    if is_swt:
        # replicate same zeros array to save memory
        z = np.zeros_like(wav_coeffs[0])
        tmp = [z, ] * nc
    else:
        # zero arrays have variable size in DWT case
        tmp = [np.zeros_like(c) for c in wav_coeffs]

    for j in range(nc):
        # tmp has arrays of zeros except for the jth entry
        tmp[j] = wav_coeffs[j]

        # reconstruct
        rec = inverse(tmp)
        if rec.shape != data.shape:
            # trim any excess coefficients
            rec = rec[tuple([slice(sz) for sz in data.shape])]
        mra_coeffs.append(rec)

        # restore zeros
        if is_swt:
            tmp[j] = z
        else:
            tmp[j] = np.zeros_like(tmp[j])
    return mra_coeffs


def imra(mra_coeffs):
    """Inverse 1D multiresolution analysis via summation.
    Parameters
    ----------
    mra_coeffs : list of ndarray
        Multiresolution analysis coefficients as returned by `mra`.
    Returns
    -------
    rec : ndarray
        The reconstructed signal.
    See Also
    --------
    ``mra``
    References
    ----------
    .. [1] Donald B. Percival and Harold O. Mofjeld. Analysis of Subtidal
        Coastal Sea Level Fluctuations Using Wavelets. Journal of the American
        Statistical Association Vol. 92, No. 439 (Sep., 1997), pp. 868-880.
        https://doi.org/10.2307/2965551
    """
    return reduce(lambda x, y: x + y, mra_coeffs)

def mra8(data, wavelet='sym8', level=None, axis=-1, transform='dwt',
         mode='symmetric'):
    '''wrapper for 'mra' defaults LA8 wavelet and symmetric extension mode
    Parameters
    ----------
    data : array_like
    	data series
    wavelet, level, axis, transform, mode
    	passed to 'mra'    
    Notes
    -----
    Tries to read several series as arrays, or reads single series as array.
    '''
    kwargs = dict(wavelet=wavelet, axis=axis, transform=transform)
    f = partial(mra, level=level, **kwargs)
    
    try:
        c = [f(data[:, i]) for i in range(data.shape[-1])]
    except:
        c = f(data)
    
    return c

def sum_adjacent_scales(c):
    '''Sums wavelet coefficients from adjacent time scales.
    Parameters
    ----------
    c : list of ndarrays
        wavelet coefficients as returned by 'pywt.wavedec' or 'mra'
    Notes
    -----
    Scales are summed pair-wise. an odd len(c) throws error
    '''
    
    if (len(c) % 2) ==0:
        csum = [c[i] + c[i+1] for i in range(0, len(c), 2)]
    else:
        raise ValueError('Cannot pair wavelet scales, number of scales is not even!')
    
    scales = [int(len(csum[i-1])/(i*4)) for i in range(1, len(csum))]    
    scales.insert(0, len(c[0]))
    
    return csum, scales