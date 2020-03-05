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
# local function
from fefunc import FlowP1
from posttreatment import Func_line
from posttreatment import Fitlogerror
from analysis import Error_computation
from posttreatment import Export_cvg_curv

################################################################################
################################################################################

### PHYSICAL PARAMETERS
b = 2.0  # Depolymerisation speed
L = 10.0  # Domain size
T = 5.0  # Integration time

### NUMERICAL PARAMETERS
NT = 400  # Number of time steps
NX = 1000  # Initial number of grid points
h = L/(NX+1)
deltat = T/NT

### INITIAL CONDITION
x = np.linspace(0.0,L,NX)
cmax = 0.1 # Maximum concentration of polymers
imax = 5
sigma = L/20
def Gaussienne(x,c,i,s):
    gaussienne = c*np.exp(-(x-i)**2/(2*s**2)) # Gaussienne
    return gaussienne


### VECTOR INITIALISATION
# Initial condition set as a gaussian
state_init = Gaussienne(x,cmax,imax,sigma)

################################################################################
################################################################################

### COMPARAISON PARAMETERS
Npoint = 25
epsilon = np.logspace(-3,0,Npoint, endpoint=False)

### ERROR COMPUTATIOM
# normt_l2, normn_l2, normd_l2 = Error_computation('l2',h,NX,deltat,NT,b,epsilon,Npoint,state_init)
# normt_h1, normn_h1, normd_h1 = Error_computation('h1',h,NX,deltat,NT,b,epsilon,Npoint,state_init)
normt_h2, normn_h2, normd_h2 = Error_computation('h2',h,NX,deltat,NT,b,epsilon,Npoint,state_init)
# normt_0, normn_0, normd_0 = Error_computation('x=0',h,NX,deltat,NT,b,epsilon,Npoint,state_init)

################################################################################
################################################################################

### POST TRAITEMENT
### FITTING CURVE
# xdatal2, poptl2, pcovl2 = Fitlogerror(-3,-1.1,epsilon,normt_l2)
# ydatal2 = np.exp(Func_line(np.log(xdatal2), *poptl2))
# # xdata, popt0, pcov0 = Fitlogerror(-3,0,epsilon,normt_0)
# xdatah1, popth1, pcovh1 = Fitlogerror(-3,-1,epsilon,normt_h1)
# ydatah1 = np.exp(Func_line(np.log(xdatah1), *popth1))
xdatah2, popth2, pcovh2 = Fitlogerror(-2,0,epsilon,normt_h2)
ydatah2 = np.exp(Func_line(np.log(xdatah2), *popth2))

################################################################################
################################################################################


# ### PLOT L^2
# fig3, ax3 = plt.subplots()
# ax3.loglog(epsilon, normt_l2, 'k+', label= "Transparent")
# ax3.loglog(epsilon, normn_l2, 'gx', label="Neumann")
# ax3.loglog(epsilon, normd_l2, 'rx', label="Dirichlet")
# ax3.loglog(xdatal2, ydatal2, 'b-', label = "fit y = {0}logx + {1}"\
#             .format(round(poptl2[0],1),round(poptl2[1],1)))
# # Legend and labels
# legend = ax3.legend(loc='lower right', shadow=True, fontsize='x-large')
# legend.get_frame().set_facecolor('C0')
# ax3.set(xlabel='epsilon', ylabel='erreur',
#        title='L2 Error for different value of diffusion coefficient')

# ### PLOT U(0,t)
# fig0, ax0 = plt.subplots()
# ax0.loglog(epsilon, normt_0, 'k+', label= "Transparent")
# ax0.loglog(epsilon, normn_0, 'gx', label="Neumann")
# ax0.loglog(epsilon, normd_0, 'rx', label="Dirichlet")
# ax0.loglog(xdata, np.exp(Func_line(np.log(xdata), *popt0)), 'b-', label = "fit y = {0}logx + {1}"\
#             .format(round(popt0[0],1),round(popt0[1],1)))
# # Legend and labels
# legend = ax0.legend(loc='lower right', shadow=True, fontsize='x-large')
# legend.get_frame().set_facecolor('C1')
# ax0.set(xlabel='epsilon', ylabel='erreur',
#        title='L2 Error fo u(0,t) for different value of diffusion coefficient')

# ### PLOT H^1
# figh, axh = plt.subplots()
# axh.loglog(epsilon, normt_h1, 'k+', label= "Transparent")
# axh.loglog(epsilon, normn_h1, 'g--', label="Neumann")
# axh.loglog(epsilon, normd_h1, 'rx', label="Dirichlet")
# axh.loglog(xdatah1, ydatah1, 'b-', label = "fit y = {0}logx + {1}"\
#             .format(round(popth1[0],1),round(popth1[1],1)))
# legend = axh.legend(loc='lower right', shadow=True, fontsize='x-large')
# legend.get_frame().set_facecolor('C0')
# axh.set(xlabel='epsilon/b', ylabel='erreur',
#        title='H1 Error for different value of diffusion coefficient')

# ### PLOT H^2
# figh2, axh2 = plt.subplots()
# axh2.loglog(epsilon, normt_h2, 'k+', label= "Transparent")
# axh2.loglog(epsilon, normn_h2, 'g--', label="Neumann")
# # axh2.loglog(epsilon, normd_h2, 'rx', label="Dirichlet")
# axh2.loglog(xdatah2, ydatah2, 'b-', label = "fit y = {0}logx + {1}"\
#             .format(round(popth2[0],1),round(popth2[1],1)))
# legend = axh2.legend(loc='lower right', shadow=True, fontsize='x-large')
# legend.get_frame().set_facecolor('C1')
# axh2.set(xlabel='epsilon/b', ylabel='erreur',
#        title='H2 Error for different value of diffusion coefficient')

# # Show plot
# plt.show()

################################################################################
################################################################################

# ### EXPORTATION
# # Norm L2
# Export_cvg_curv("transp_cvg_l2",epsilon,normt_l2)
# Export_cvg_curv("neumann_cvg_l2",epsilon,normn_l2)
# Export_cvg_curv("dirichlet_cvg_l2",epsilon,normd_l2)
# Export_cvg_curv("line_cvg_l2",xdatal2,ydatal2)
# # Norm H1
# Export_cvg_curv("transp_cvg_h1",epsilon,normt_h1)
# Export_cvg_curv("neumann_cvg_h1",epsilon,normn_h1)
# Export_cvg_curv("dirichlet_cvg_h1",epsilon,normd_h1)
# Export_cvg_curv("line_cvg_h1",xdatah1,ydatah1)
# # Norm H2
# Export_cvg_curv("transp_cvg_h2",epsilon,normt_h2)
# Export_cvg_curv("neumann_cvg_h2",epsilon,normn_h2)
Export_cvg_curv("line_cvg_h2",xdatah2,ydatah2)

################################################################################
################################################################################