### IMPORTATION
import numpy as np
from matplotlib.cbook import get_sample_data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import sys
# optimization function
from scipy.optimize import minimize
from scipy.optimize import curve_fit


def Func_line(x, a0, b0):
    return a0*x+b0

def Fitlogerror(binf,bsup,x,y,func=Func_line):
    n1 = np.abs(x-10**binf).argmin() # starting point
    n2 = np.abs(x-10**bsup).argmin()  # stopping point
    xdata = np.log(x[n1:n2])
    ydata = np.log(y[n1:n2])
    popt, pcov = curve_fit(func, xdata, ydata)
    return x[n1:n2], popt, pcov

### EXPORT 
# In file Redaction/data
def Export_cvg_curv(name,xdata,ydata):
    Npoint = np.size(ydata)
    with open('../../Redaction/data/'+name+'.txt', 'w') as f:
        f.writelines('eps norm \n')
        for i in range(Npoint):
            web_browsers = ['{0}'.format(xdata[i]),' ','{0} \n'.format(ydata[i])]
            f.writelines(web_browsers)
