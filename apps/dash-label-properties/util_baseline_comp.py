import numpy as np
import pandas as pd


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.") from exec
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.") from exec
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'") from exec
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def baseline_edges(row):
    N = int(0.05 * row.shape[0])
    sm = smooth(row.values, window_len=N)
    sm = sm[int(N/2):]
    sm = sm[:row.shape[0]]
    sm = pd.Series(sm)
    out = sm.rename(str(row.name))
    return out


def baseline_of_the_row(row):
    x = row.index
    y = row.values
    y = y.astype(float)
    y[1:-1] = np.nan  #y[10:-10] = np.nan
    y_ = pd.Series(y)
    y = y_.values
    idx = np.isfinite(x) & np.isfinite(y)
    model = np.polyfit(x[idx], y[idx], 1)  # 1 or 3
    po = np.polyval(model, x)
    po = pd.Series(po)
    end = po.rename(str(row.name))
    return end