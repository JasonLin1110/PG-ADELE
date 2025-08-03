import torch
import numpy as np
from scipy.optimize import curve_fit
import warnings
warnings.simplefilter("ignore", UserWarning)

def curve_func(x, a, b, c):
    return a *(1-np.exp( -1/c * x**b  ))

def fit(func, x, y):
    popt, pcov = curve_fit(func, x, y, p0 =(1,1,1), method= 'trf', sigma = np.geomspace(1,.1,len(y)), absolute_sigma=True, bounds= ([0,0,0],[1,1,np.inf]) )
    return tuple(popt)

def derivation(x, a, b, c):
    x = x + 1e-6 # numerical robustness
    return a * b * 1/c * np.exp(-1/c * x**b) * (x**(b-1))

def label_update_epoch(ydata_fit, n_epoch = 16, threshold = 0.9, eval_interval = 20, num_iter_per_epoch= 10581/10):
    xdata_fit = np.linspace(0, len(ydata_fit)*eval_interval/num_iter_per_epoch, len(ydata_fit))
    a, b, c = fit(curve_func, xdata_fit, ydata_fit)
    epoch = np.arange(1, n_epoch)
    y_hat = curve_func(epoch, a, b, c)
    relative_change = abs(abs(derivation(epoch, a, b, c)) - abs(derivation(1, a, b, c)))/ abs(derivation(1, a, b, c))
    relative_change[relative_change > 1] = 0
    update_epoch = np.sum(relative_change <= threshold) + 1
    return update_epoch#, a, b, c

def if_update(parameters, iou_value, current_epoch, n_epoch = 16, threshold = 0.90, eval_interval=1, num_iter_per_epoch=1):
    # check iou_value
    start_iter = 0
    for k in range(len(iou_value)-1):
        if iou_value[k+1]-iou_value[k] < parameters["effect_learn"]:
            start_iter = max(start_iter, k + 1)
        else:
            break
    shifted_epoch = start_iter*eval_interval/num_iter_per_epoch
    #cut out the first few entries
    iou_value = iou_value[start_iter: ]
    update_epoch = label_update_epoch(iou_value, n_epoch = n_epoch, threshold=threshold, eval_interval=eval_interval, num_iter_per_epoch=num_iter_per_epoch)
    # Shift back
    update_epoch = shifted_epoch + update_epoch
    return current_epoch >= update_epoch