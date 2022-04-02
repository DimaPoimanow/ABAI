import numpy as np
from scipy.optimize import curve_fit


def approximate_oil(x_init, y_init, n):
    best_error = 10**10
    name = 'exp'
    best_params = [0,0]
    def linear(x,a):
        return a*x + ( -a*x[::-1][0] + y[::-1][0])
    
    l_end = n
    for l in range(n+1, x_init.shape[0]):
        x = x_init[::-1][:l]
        y = y_init[::-1][:l]
        a, _ = curve_fit(linear, x, y, maxfev = 10**8)
        err_1 = np.mean(np.power(y - a*x - (-a*x[::-1][0] + y[::-1][0]), 2))
        if err_1 < best_error:
            best_error = err_1
            name = 'linear'
            best_params = [a]
            l_end = l
    return best_error, name, best_params, l_end