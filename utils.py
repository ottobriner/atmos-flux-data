import pandas as pd
import numpy as np
import geopandas as gpd
from geographiclib.geodesic import Geodesic

# import pywt
from functools import partial, reduce
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

def read_l3(filepath, datecol='DATE'):
    df = (pd.read_csv(filepath, header=0, low_memory=False)
      .replace(-9999, np.nan)
      .assign(date = lambda x: pd.to_datetime(x[datecol]))
      .set_index('date', drop=False)
     )
    df.index = df.index.set_names(None)
    return df

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