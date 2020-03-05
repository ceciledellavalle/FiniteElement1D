### IMPORTATION
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import sys


################################################################################
###                            ELEMENTS FINIS - P1

def FlowP1(nc,h,NX,deltat,b,epsilon):
    # We ignore the last element for which u(L,t) =0
    # 0 -- Transparent
    # 1 -- Dirichlet
    # 2 -- Neumann
    # 3 -- Robin
    # 4 -- Limit Model (transport)
    # Matrice de Masse
    M =h/6*(4*np.eye(NX)\
    +np.diag(np.ones(NX-1),1)\
    +np.diag(np.ones(NX-1),-1))
    if nc == 0 :
        M[0,0] = h/3
    if nc == 1 :
        pass
    if nc == 2 :
        M[0,0] = h/3
    if nc == 3 :
        M[0,0] = h/3
    if nc == 4 :
        M[0,0] = h/3
    # Matrice de rigidité 1
    # Diffusion
    K = 1/h*(2*np.eye(NX)\
    -np.diag(np.ones(NX-1),1)\
    -np.diag(np.ones(NX-1),-1))
    if nc == 0 :
        K[0,0] = 1/h
    if nc == 1 :
        pass
    if nc == 2 :
        K[0,0] = 1/h
    if nc == 3 :
        K[0,0] =-1
    if nc == 4 :
        K = np.zeros((NX,NX))
    # Matrice de rigidité 2
    # Transport
    D = 1/2*(\
    +np.diag(np.ones(NX-1),1)\
    -np.diag(np.ones(NX-1),-1))
    if nc == 0 :
        D[0,0] = -1/2
    if nc == 1 :
        pass
    if nc == 2 :
        D[0,0] = -1/2
    if nc == 4 :
        D[0,0] = -1/2
    # Matrice CL en x=0
    Do = np.zeros((NX,NX))
    if nc == 0:
        Do[0,0]=1
    #Flow computation
    flow = np.linalg.inv(M -deltat*b*D +deltat*epsilon*K +epsilon/b*Do).dot(M+epsilon/b*Do)
    return flow

    
def Mass_matrixP1(h,NX):
    # Mass matrix
    M =h/6*(4*np.eye(NX)\
    +np.diag(np.ones(NX-1),1)\
    +np.diag(np.ones(NX-1),-1))
    M[0,0] = h/3
    return M

def Rigidity_matrixP1(h,NX):
    # Diffusion matrix
    K = 1/h*(2*np.eye(NX)\
    -np.diag(np.ones(NX-1),1)\
    -np.diag(np.ones(NX-1),-1))
    K[0,0]=1/h
    return K

def Deriv_matrixP1(h,NX):
    # Transport matrix
    D = 1/2*(\
    +np.diag(np.ones(NX-1),1)\
    -np.diag(np.ones(NX-1),-1))
    D[0,0]= -1/2
    return D



################################################################################
###                            ELEMENTS FINIS - P2

def FlowP2(nc,h,N,deltat,b,epsilon):
    # We ignore the last element for which u(L,t) =0
    # 0 -- Transparent
    # 2 -- Neumann
    # 4 -- Limit Model (transport)
    # ==================================================
    # Matrice de Masse
    M = h*np.array([[2/15,1/15,-1/30],[1/15,8/15,1/15],[-1/30,1/15,2/15]])
    # Matrice de taille 2N 
    # Et non 2N+1 car on élimine le point en x=L
    Mi = np.zeros((2*N,2*N))
    for i in range(0,N-1):
        Mi[2*i:2*i+3,2*i:2*i+3] += M
    # méthode par élimination en x=L
    Mi[2*N-2:,2*N-2:] += M[:-1,:-1]
    if nc == 0 :
        pass
    if nc == 2 :
        pass
    if nc == 4 :
        pass
    # ==================================================
    # Matrice de rigidité 1
    K = 1/h*np.array([[7/3,-8/3,1/3],[-8/3,16/3,-8/3],[1/3,-8/3,7/3]])
    Ki = np.zeros((2*N,2*N))
    for i in range(0,N-1):
        Ki[2*i:2*i+3,2*i:2*i+3] += K
    Ki[2*N-2:,2*N-2:] += K[:-1,:-1]
    if nc == 0 :
        pass
    if nc == 2 :
        pass
    if nc == 4 :
        Ki = np.zeros((2*N,2*N))
    # Matrice de rigidité 2
    D = np.array([[-1/2,2/3,-1/6],[-2/3,0,2/3],[1/6,-2/3,1/2]])
    Di = np.zeros((2*N,2*N))
    for i in range(0,N-1):
        Di[2*i:2*i+3,2*i:2*i+3] += D
    Di[2*N-2:,2*N-2:] += D[:-1,:-1]
    # Matrice CL en x=0
    Do = np.zeros((2*N,2*N))
    if nc == 0:
        Do[0,0]=1
    #Flow computation
    flow = np.linalg.inv(Mi +epsilon/b*Do -deltat*b*Di +deltat*epsilon*Ki).dot(Mi+epsilon/b*Do)
    return flow


def Mass_matrixP2(h,N,NX):
    # Matrice de Masse
    M = h*np.array([[2/15,1/15,-1/30],[1/15,8/15,1/15],[-1/30,1/15,2/15]])
    # Matrice de taille 2N 
    # Et non 2N+1 car on élimine le point en x=L
    Mi = np.zeros((NX,NX))
    for i in range(0,N-1):
        Mi[2*i:2*i+3,2*i:2*i+3] += M
    # méthode par élimination en x=L
    Mi[2*N-2:,2*N-2:] += M[:-1,:-1]
    return Mi

def Rigidity_matrixP2(h,N,NX):
    # Diffusion matrix
    K = 1/h*np.array([[7/3,-8/3,1/3],[-8/3,16/3,-8/3],[1/3,-8/3,7/3]])
    Ki = np.zeros((2*N,2*N))
    for i in range(0,N-1):
        Ki[2*i:2*i+3,2*i:2*i+3] += K
    Ki[2*N-2:,2*N-2:] += K[:-1,:-1]
    return Ki