### IMPORTATION
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import sys
# optimization function
from scipy.optimize import minimize
from scipy.optimize import curve_fit
### LOCAL FUNCTION
from plotdynamic import PlotDymamicSolution
from fefunc import FlowP2

################################################################################
################################################################################

### PHYSICAL PARAMETERS
b = 2.0  # Depolymerisation speed
epsilon = 0.5
L = 100.0  # Domain size
T = 50.0  # Integration time

### NUMERICAL PARAMETERS
NT = 150  # Number of time steps
N = 200  # Initial number of grid points
NX = 2*N
h = L/(N+1)
deltat = T/NT

### INITIAL CONDITION
x = np.linspace(0.0,L,NX)
cmax = 0.1 # Maximum concentration of polymers
imax = 1/2*L
sigma = L/20
def Gaussienne(No,c0,i0,s0):
    x = np.linspace(0.0,L,No)
    gaussienne = c0*np.exp(-(x-i0)**2/(2*s0**2)) # Gaussienne
    gaussienne[0]=0
    return gaussienne
### VECTOR INITIALISATION
# Initial condition set as a gaussian
state_init = Gaussienne(NX,cmax,imax,sigma)


################################################################################
################################################################################

### COMPUTATION OF THE EXACT SOLUTION
### Solution for eps =0
sv = np.zeros((NX,NT))
sv[:,0] = state_init.copy()
flow_v = FlowP2(4,h,N,deltat,b,0)
for i in range(0,NT-1):
    sv[:,i+1] = flow_v.dot(sv[:,i])

Npoint = 40
epsilon = np.logspace(-12,-1,Npoint, endpoint=False)
k=0
st = np.zeros((NX,NT,Npoint))
sn = np.zeros((NX,NT,Npoint))
sd = np.zeros((NX,NT,Npoint))
# Transparent
normt_l2=np.zeros(Npoint) 
normt_h1=np.zeros(Npoint)
normt_h2=np.zeros(Npoint)
# Neuman
normn_l2=np.zeros(Npoint) 
normn_h1=np.zeros(Npoint)
normn_h2=np.zeros(Npoint) 
# Dirichlet
normd_l2=np.zeros(Npoint) 
normd_h1=np.zeros(Npoint) 
normd_h2=np.zeros(Npoint)
############################
# Mass matrix
M = h*np.array([[2/15,1/15,-1/30],[1/15,8/15,1/15],[-1/30,1/15,2/15]])
Mi = np.zeros((2*N,2*N))
for i in range(0,N-1):
    Mi[2*i:2*i+3,2*i:2*i+3] += M
# méthode par élimination en x=L
Mi[2*N-2:,2*N-2:] += M[:-1,:-1]
# Diffusion matrix
K = 1/h*np.array([[7/3,-8/3,1/3],[-8/3,16/3,-8/3],[1/3,-8/3,7/3]])
Ki = np.zeros((2*N,2*N))
for i in range(0,N-1):
    Ki[2*i:2*i+3,2*i:2*i+3] += K
Ki[2*N-2:,2*N-2:] += K[:-1,:-1]
# Norm matrix of second order derivative
N2 = 1/h**3*np.array([[16,-32,16],[-32,64,-32],[16,-32,16]])
N2i = np.zeros((2*N,2*N))
for i in range(0,N-1):
    N2i[2*i:2*i+3,2*i:2*i+3] += N2
N2i[2*N-2:,2*N-2:] += N2[:-1,:-1]


for eps in epsilon:
    ### RESOLUTION : CL TRANSPARENT
    # state initialisation
    st[:,0,k] = state_init.copy()
    A = FlowP2(0,h,N,deltat,b,eps)
    for i in range(0,NT-1):
        st[:,i+1,k] = A.dot(st[:,i,k])
        # L^2 NORM
        normt_l2[k] += T/NT*np.sum(Mi*np.square(st[:,i,k]- sv[:,i]))
        # H^1 NORM
        normt_h1[k] += T/NT*np.sum(Ki*np.square(st[:,i,k]- sv[:,i]))
        # H^2 NORM
        normt_h2[k] += T/NT*np.sum((N2i)*np.square(st[:,i,k]- sv[:,i]))
    ### RESOLUTION : CL NEUMANN
    # state initialisation
    sn[:,0,k] = state_init.copy()
    A = FlowP2(2,h,N,deltat,b,eps)
    for i in range(0,NT-1):
        sn[:,i+1,k] = A.dot(sn[:,i,k])
        # L^2 NORM
        normn_l2[k] += T/NT*np.sum(Mi*np.square(sn[:,i,k]- sv[:,i]))
        # H^1 NORM
        normn_h1[k] += T/NT*np.sum((Ki)*np.square(sn[:,i,k]- sv[:,i]))
        # H^2 NORM
        normn_h2[k] += T/NT*np.sum((N2i)*np.square(sn[:,i,k]- sv[:,i]))
    ### INCREMENT
    k+=1

################################################################################
################################################################################

### POST TRAITEMENT
### FITTING CURVE
def Func_line(x, a0, b0):
    return a0*x+b0
n1 = np.abs(epsilon-10**-12).argmin() # starting point
n2 = np.abs(epsilon-10**-3).argmin()  # stopping point
npt = n2-n1
xdata = np.log(epsilon[n1:n2])
ydata = np.log(normt_h2[n1:n2])
popth2, pcovh2 = curve_fit(Func_line, xdata, ydata)

################################################################################
################################################################################

### PLOT H^2
figh, axh = plt.subplots()
axh.loglog(epsilon, normt_h2, 'k+', label= "Transparent")
axh.loglog(epsilon, normn_h2, 'g+', label="Neumann")
axh.loglog(epsilon[n1:n2], np.exp(Func_line(xdata, *popth2)), 'b-', label = "fit y = {0}logx + {1}"\
            .format(round(popth2[0],1),round(popth2[1],1)))
legend = axh.legend(loc='lower right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('C0')
axh.set(xlabel='epsilon', ylabel='erreur',
       title='H2 Error for different value of diffusion coefficient')

fig0, ax0 = plt.subplots()
ax0.plot(np.linspace(0,1,NT), st[0,:,35]-sn[0,:,35], 'k+', label= "Transparent")
#ax0.loglog(np.linspace(0,1,NT), , 'g+', label= "Neumann")
legend = axh.legend(loc='lower right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('C0')
ax0.set(xlabel='epsilon', ylabel='u(0,t)',
       title='Comparaison on x=0 for eps ={}'.format(epsilon[35]))

# Show plot
plt.show()



################################################################################
################################################################################
# #     ESSAI CLINIQUE

# ### CL Transparent
# statet = np.zeros((NX,NT))
# statet[:,0] = state_init.copy()
# staten = np.zeros((NX,NT))
# staten[:,0] = state_init.copy()
# flowt = FlowP2(0,h,N,deltat,b,1)
# flown = FlowP2(2,h,N,deltat,b,1)
# for i in range(0,NT-1):
#     statet[:,i+1] = flowt.dot(statet[:,i])
#     staten[:,i+1] = flown.dot(staten[:,i])

# ### DATA VIZUALISATION ##########################################################

# # First set up the figure, the axis, and the plot element we want to animate
# transport1d = PlotDymamicSolution(L,1.1*np.amax(statet),np.linspace(0, L, NX),statet-staten,\
#                                   NT,np.linspace(0,T,NT))
# plt.show()

# print(statet-staten)
################################################################################
################################################################################