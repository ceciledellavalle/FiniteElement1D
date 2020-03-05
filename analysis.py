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
from fefunc import Mass_matrixP1
from fefunc import Mass_matrixP2
from fefunc import Rigidity_matrixP1
from fefunc import Rigidity_matrixP2
from fefunc import Deriv_matrixP1
from fefunc import FlowP1


def Error_computation(opt,h,NX,deltat,NT,b,epsilon,Npoint,state_init,\
    N=0,elt='P1',Flow_func=FlowP1):
    ### COMPUTATION OF THE EXACT SOLUTION
    ### Solution for eps =0
    sv = np.zeros((NX,NT))
    sv[:,0] = state_init.copy()
    flow_v = Flow_func(4,h,NX,deltat,b,0)
    for i in range(0,NT-1):
        sv[:,i+1] = flow_v.dot(sv[:,i])
    ### COMPUTATION OF THE SOLUTION WITH MODEL ERROR
    ### INITIALISATION
    # Transparent
    st = np.zeros((NX,NT,Npoint))
    normt=np.zeros(Npoint) 
    # Neuman
    sn = np.zeros((NX,NT,Npoint))
    normn=np.zeros(Npoint) 
    # Dirichlet
    sd = np.zeros((NX,NT,Npoint))
    normd=np.zeros(Npoint) 
    ############################
    if elt=='P1':
        # Mass matrix
        M = Mass_matrixP1(h,NX)
        # Diffusion matrix
        K = Rigidity_matrixP1(h,NX)
    elif elt=='P2':
        # Mass matrix
        M = Mass_matrixP2(h,N,NX)
        # Diffusion matrix
        K = Rigidity_matrixP2(h,N,NX)
    ##############################
    k=0
    for eps in epsilon:
        ### RESOLUTION : CL TRANSPARENT
        # state initialisation
        st[:,0,k] = state_init.copy()
        A = Flow_func(0,h,NX,deltat,b,eps)
        for i in range(0,NT-1):
            st[:,i+1,k] = A.dot(st[:,i,k])
            # L^2 NORM
            if opt == 'l2':
                norm_temp = np.transpose(st[:,i,k]- sv[:,i]).dot(M).dot(st[:,i,k]- sv[:,i])
                normt[k] += deltat*norm_temp
            # H^1 NORM
            if opt == 'h1':
                norm_temp = np.transpose(st[:,i,k]- sv[:,i]).dot(K).dot(st[:,i,k]- sv[:,i])
                normt[k] += deltat*norm_temp
            # H^2 NORM
            if opt == 'h2':
                v = -np.linalg.inv(M).dot(K).dot(st[:,i,k]- sv[:,i])
                norm_temp = np.transpose(v).dot(K).dot(v)
                normt[k] += deltat*norm_temp
        # NORM x=0
        if opt == 'x=0':
            normt[k] = np.amax((st[0,:,k]- sv[0,:])**2)
        ### RESOLUTION : CL NEUMANN
        # state initialisation
        sn[:,0,k] = state_init.copy()
        A = Flow_func(2,h,NX,deltat,b,eps)
        for i in range(0,NT-1):
            sn[:,i+1,k] = A.dot(sn[:,i,k])
            # L^2 NORM
            if opt == 'l2':
                norm_temp = np.transpose(sn[:,i,k]- sv[:,i]).dot(M).dot(sn[:,i,k]- sv[:,i])
                normn[k] += deltat*norm_temp
            # H^1 NORM 
            if opt == 'h1':
                norm_temp = np.transpose(sn[:,i,k]- sv[:,i]).dot(K).dot(sn[:,i,k]- sv[:,i])
                normn[k] += deltat*norm_temp
            # H^2 NORM
            if opt == 'h2':
                v = -np.linalg.inv(M).dot(K).dot(sn[:,i,k]- sv[:,i])
                norm_temp = np.transpose(v).dot(K).dot(v)
                normn[k] += deltat*norm_temp
        # NORM x=0
        if opt == 'x=0':
            normn[k] = np.amax((sn[0,:,k]- sv[0,:])**2)
        ### RESOLUTION : CL DIRICHLET
        # state initialisation
        sd[:,0,k] = state_init.copy()
        A = Flow_func(1,h,NX,deltat,b,eps)
        for i in range(0,NT-1):
            sd[:,i+1,k] = A.dot(sd[:,i,k])
            # L^2 NORM
            if opt == 'l2':
                norm_temp = np.transpose(sd[:,i,k]- sv[:,i]).dot(M).dot(sd[:,i,k]- sv[:,i])
                normd[k] += deltat*norm_temp
            # H^1 NORM
            if opt == 'h1':
                norm_temp = np.transpose(sd[:,i,k]- sv[:,i]).dot(K).dot(sd[:,i,k]- sv[:,i])
                normd[k] += deltat*norm_temp
            # H^2 NORM
            if opt == 'h2':
                v = -np.linalg.inv(M).dot(K).dot(sd[:,i,k]- sv[:,i])
                norm_temp = np.transpose(v).dot(K).dot(v)
                normd[k] += deltat*norm_temp
        # NORM x=0
        if opt == 'x=0':
            normd[k] = np.amax((sd[0,:,k]- sv[0,:])**2)
        ### INCREMENT
        k+=1
    #
    return normt, normn, normd