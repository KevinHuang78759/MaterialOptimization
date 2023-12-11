# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:28:57 2023

@author: kh787
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from numba import jit, cuda
import matplotlib.animation as animation


#Local Force Vector
local_F = np.array([0, 1./2, 1./12, 0, 1./2, -1./12])
local_F_ax = np.array([1/2, 0, 0, 1/2, 0, 0])

#Shape functions for result interpolation of forces
axial_shape = np.array([-1, 0, 0, 1, 0, 0])
shear_shape = 8 * np.array([0, 3./2, 3./4, 0, -3./2, 3./4])
#load function
@jit(nopython = True)
def load_func(x):
    return -np.abs(np.sin(x * np.pi))*10

@jit(nopython = True)
def load_func_backloaded(x):
    return -100/(1 + np.exp(-10 *(x - 0.5)))
#Creates initialization state for single length beam
@jit(nopython = True)
def create_init_single(NumEls = 25):
    
    node_x = np.zeros(NumEls + 1)
    node_y = np.zeros(NumEls + 1)
    for i in range(0, NumEls + 1):
        node_y[i] = 0.0
        node_x[i] = i/NumEls
    #Connection Array
    ElConn = np.zeros((NumEls, 2), dtype = np.int16)
    
    load_distr = np.zeros(NumEls)
    for i in range(0, NumEls):
        ElConn[i,0] = i
        ElConn[i,1] = i + 1
        x_p = i/NumEls + 1/(2*NumEls)
        load_distr[i] = load_func(x_p)
    
    #Return the nodes connectivity, loads, start condition, fixed nodes, and end node
    start_cond = np.ones(NumEls) * 5
    
    fixed = np.zeros(2, dtype = np.int16)
    fixed[1] = NumEls
    dvars = np.arange(0, NumEls)
    return node_x, node_y, ElConn, load_distr, start_cond, fixed, NumEls, dvars

def create_bracket(ElW = 4):
    n = (ElW+1)**2
    node_x = np.zeros(n)
    node_y = np.zeros(n)
    dvars = []
    fixed = []
    for i in range(0, n):
        node_x[i] = i % (ElW + 1)
        node_y[i] = i // (ElW + 1)
        if i % (ElW + 1) == 0:
            fixed.append(i)
    
    node_x /= ElW
    node_y /= ElW
    numEls = ElW * (ElW + 1) * 2 + ElW * ElW
    
    load_distr = np.zeros(numEls)
    start_A = np.ones(numEls)*5
    ElConn = np.zeros((numEls, 2), dtype = np.int16)
    
    for i in range(0, numEls):
        if i < ElW*ElW:
            #Diagonal Element designation
            offset = i // ElW
            ElConn[i, 0] = i + offset
            ElConn[i, 1] = ElConn[i, 0] + ElW + 2
            dvars.append(i)
        elif i < ElW * ElW + ElW * (ElW + 1):
            #Vertical Elements
            vert_i = i - ElW * ElW
            ElConn[i, 0] = vert_i
            ElConn[i, 1] = vert_i + ElW + 1
            
            dvars.append(i)
        else:
            #Horizontal Elements
            h_i = int(i - ElW * ElW - ElW * (ElW + 1))
            offset = h_i // ElW
            ElConn[i, 0] = h_i + offset
            ElConn[i, 1] = ElConn[i, 0] + 1 
            if h_i + offset + ElW >= n-1:
                
                xp = (node_x[h_i + offset] + node_x[ElConn[i, 0] + 1])/2
                load_distr[i] = load_func_backloaded(xp)
            else:
                dvars.append(i)
    dvars = np.array(dvars)
    fixed = np.array(fixed)
    fixed = np.sort(fixed)
    
    return node_x, node_y, ElConn, load_distr, start_A, fixed, numEls, dvars
    
    

def create_lcl_prob(x1,y1,x2,y2, Area, Load):
    
    L = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    c = (x2 - x1)/L
    s = (y2 - y1)/L
    
    rot_mat = np.array([[ c, s, 0, 0, 0, 0],
                        [-s, c, 0, 0, 0, 0],
                        [ 0, 0, 1, 0, 0, 0],
                        [ 0, 0, 0, c, s, 0],
                        [ 0, 0, 0,-s, c, 0],
                        [ 0, 0, 0, 0, 0, 1]])
    
    
    f_lcl = Load * L * local_F
    
    f_lcl = rot_mat.T @ f_lcl
    
    el_const = Area/L
    def_const = Area*Area/(12 * L**3)
    
    #Elongation Local Stiffness
    local_stiff_elo = np.array([[ 1, 0, 0,-1, 0, 0],
                                             [ 0, 0, 0, 0, 0, 0],
                                             [ 0, 0, 0, 0, 0, 0],
                                             [-1, 0, 0, 1, 0, 0],
                                             [ 0, 0, 0, 0, 0, 0],
                                             [ 0, 0, 0, 0, 0, 0]],dtype = np.float64)
    
    ke_el = el_const * local_stiff_elo
    sdd =  np.array([[ 0, 0, 0, 0, 0, 0],
                                         [ 0,12, 6*L, 0,-12,6*L],
                                         [ 0, 6*L, 4*L*L, 0,-6*L, 2*L*L],
                                         [ 0, 0, 0, 0, 0, 0],
                                         [ 0,-12,-6*L, 0,12,-6*L],
                                         [ 0, 6*L, 2*L*L, 0,-6*L, 4*L*L]],dtype = np.float64)
    ke_def = def_const * sdd
    
    k_all = ke_el + ke_def
    
    k_all = rot_mat.T @ k_all @ rot_mat
    
    return k_all, f_lcl
    
def create_lcl_dK(x1,y1,x2,y2, Area, Load):
    
    L = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    c = (x2 - x1)/L
    s = (y2 - y1)/L
    
    rot_mat = np.array([[ c, s, 0, 0, 0, 0],
                        [-s, c, 0, 0, 0, 0],
                        [ 0, 0, 1, 0, 0, 0],
                        [ 0, 0, 0, c, s, 0],
                        [ 0, 0, 0,-s, c, 0],
                        [ 0, 0, 0, 0, 0, 1]])
    
    
    el_const = 1/L
    def_const = Area/(6*L**3)
    
    #Elongation Local Stiffness
    local_stiff_elo = np.array([[ 1, 0, 0,-1, 0, 0],
                                             [ 0, 0, 0, 0, 0, 0],
                                             [ 0, 0, 0, 0, 0, 0],
                                             [-1, 0, 0, 1, 0, 0],
                                             [ 0, 0, 0, 0, 0, 0],
                                             [ 0, 0, 0, 0, 0, 0]],dtype = np.float64)
    
    ke_el = el_const * local_stiff_elo
    sdd =  np.array([[ 0, 0, 0, 0, 0, 0],
                                         [ 0,12, 6*L, 0,-12,6*L],
                                         [ 0, 6*L, 4*L*L, 0,-6*L, 2*L*L],
                                         [ 0, 0, 0, 0, 0, 0],
                                         [ 0,-12,-6*L, 0,12,-6*L],
                                         [ 0, 6*L, 2*L*L, 0,-6*L, 4*L*L]],dtype = np.float64)
    ke_def = def_const * sdd
    
    k_all = ke_el + ke_def
    
    k_all = rot_mat.T @ k_all @ rot_mat
    
    return k_all

def create_globals(nx, ny, ElConn, load_distr, condition, fixed):
    num_nodes = nx.shape[0]
    global_K = np.zeros((3*num_nodes, 3*num_nodes))
    global_F = np.zeros((3*num_nodes))
    
    #Now loop through elements and populate the globals
    for e in range(0, ElConn.shape[0]):
        x1 = nx[ElConn[e, 0]]
        y1 = ny[ElConn[e, 0]]
        x2 = nx[ElConn[e, 1]]
        y2 = ny[ElConn[e, 1]]
        A = condition[e]
        Load = load_distr[e]
        
        K_e, F_e = create_lcl_prob(x1, y1, x2, y2, A, Load)
        
        global_K[ElConn[e, 0]*3:ElConn[e, 0]*3 + 3, ElConn[e, 0]*3:ElConn[e, 0]*3 + 3] += K_e[:3,:3]
        
        global_K[ElConn[e, 1]*3:ElConn[e, 1]*3 + 3, ElConn[e, 1]*3:ElConn[e, 1]*3 + 3] += K_e[3:,3:]
        
        global_K[ElConn[e, 0]*3:ElConn[e, 0]*3 + 3, ElConn[e, 1]*3:ElConn[e, 1]*3 + 3] += K_e[:3,3:]
        
        global_K[ElConn[e, 1]*3:ElConn[e, 1]*3 + 3, ElConn[e, 0]*3:ElConn[e, 0]*3 + 3] += K_e[3:,:3]
        
        
        global_F[ElConn[e, 0]*3:ElConn[e, 0]*3 + 3] += F_e[:3]
        
        global_F[ElConn[e, 1]*3:ElConn[e, 1]*3 + 3] += F_e[3:]
    
    #Remove rows and columns from fixed ends
    deletion = []
    for i in range(fixed.shape[0]):
        deletion.append(fixed[i]*3)
        deletion.append(fixed[i]*3 + 1)
        deletion.append(fixed[i]*3 + 2)
    global_K = np.delete(global_K, deletion, axis = 0)
    global_K = np.delete(global_K, deletion, axis = 1)
    global_F = np.delete(global_F, deletion)
    
    return global_K, global_F


#binary search to find the number of fixed node indices before a target node number
@jit(nopython = True)
def bs(arr, target):
    left, right = 0, arr.shape[0] - 1
    count = 0
    
    while left <= right:
        mid = (left + right) // 2
        mid_value = arr[mid]
        
        if mid_value == target:
            return -1  # Target found, return -1
        elif mid_value < target:
            count = mid + 1
            left = mid + 1
        else:
            right = mid - 1
    
    return count  # Return the count of values less than the target


def get_result_list(nx, ny, fixed, d, ElConn):
    el_list = []
    for i in range(0, ElConn.shape[0]):
        n1 = ElConn[i, 0]
        n2 = ElConn[i, 1]
        
        lcl_sln = np.zeros(6)
        c1 = bs(fixed, n1)
        c2 = bs(fixed, n2)
        
        if c1 >= 0:
            rn1 = n1 - c1
            lcl_sln[:3] = d[rn1*3:rn1*3+3]
            
        if c2 >= 0:
            rn2 = n2 - c2
            lcl_sln[3:] = d[rn2*3:rn2*3+3]
        
        x1 = nx[n1]
        y1 = ny[n1]
        x2 = nx[n2]
        y2 = ny[n2]
        
        
        lcl_sln = lcl_sln
        
        line = [x1 + lcl_sln[0],
                y1 + lcl_sln[1],
                x2 + lcl_sln[3],
                y2 + lcl_sln[4]]
        el_list.append(line)
    return el_list
        
def plot_list(el_list, areas):
    #normalize areas
    
    max_ab = np.max(np.abs(areas))
    if max_ab > 0:
        areas /= max_ab
    
    areas *= 1/2
    fig, ax = plt.subplots(1,1,dpi = 300)
    idx = 0
    for lst in el_list:
        ax.plot([lst[0], lst[2]], [lst[1], lst[3]], color = (0.5 - areas[idx],0.5 + areas[idx],0))
        
        idx += 1
    plt.show()

def restricted_SGD(generator, iters = 10, rate = 0.1, batch_size = 5, disp = False):
    nx, ny, con, ld, ar, fi, poi, dvars = generator()
    ar_org = ar.copy()
    #Material Restraint
    max_material = np.sum(ar[dvars])
    
    norms = np.zeros(iters)
    for i in range(iters):
        ar_diff = ar - ar_org
        K, F = create_globals(nx, ny, con, ld, ar, fi)
        
        K_inv = np.linalg.inv(K)
        d = K_inv @ F
        
        if disp:
            r_list = get_result_list(nx, ny, fi, d, con)
            
            
            plot_list(r_list,ar_diff)
        
        #Perform the gradient descent step
        
        #First compute the gradient
        
        du2dAi = np.zeros(dvars.shape[0])
        
        for dv in range(0, dvars.shape[0]):
            
            #First we need to compute dK/dA for this design variable
            dKdA = np.zeros_like(K)
            el = dvars[dv]
            n1 = con[el, 0]
            n2 = con[el, 1]
            
            x1 = nx[n1]
            y1 = ny[n1]
            
            x2 = nx[n2]
            y2 = ny[n2]
            
            lcldKdA = create_lcl_dK(x1, y1, x2, y2, ar[el], ld[el])
            c1 = bs(fi, n1)
            c2 = bs(fi, n2)
            
            if c1 >= 0:
                rn1 = n1 - c1
                dKdA[rn1*3:rn1*3 +3,rn1*3:rn1*3 +3] = lcldKdA[:3,:3]
                
            if c2 >= 0:
                rn2 = n2 - c2
                dKdA[rn2*3:rn2*3 +3,rn2*3:rn2*3 +3] = lcldKdA[3:,3:]
            
            if c1 >= 0 and c2 >= 0:
                rn1 = n1 - c1
                rn2 = n2 - c2
                dKdA[rn1*3:rn1*3 +3,rn2*3:rn2*3 +3] = lcldKdA[:3,3:]
                dKdA[rn2*3:rn2*3 +3,rn1*3:rn1*3 +3] = lcldKdA[3:,:3]
            
            #For now F doesn't depend on the areas
            dudA = -K_inv @ dKdA @ d
            du2dAi[dv] = 2 * d.T @ dudA
        
        #generate batch of randomly sampled new areas
        batches = np.zeros((batch_size, dvars.shape[0]))
        xt = ar[dvars] - rate*du2dAi
        std = np.ones(dvars.shape[0]) * rate
        for j in range(0, batch_size):
            batches[j] = np.random.normal(xt, std)
        
        #clean the batches to project back into feasible region
        #first make all areas non negative
        batches = np.clip(batches, 0, np.inf)
        #now make sure they sum to at most the total material
        for j in range(0, batch_size):
            s = np.sum(batches[j])
            
            #scale by projection
            batches[j] = batches[j]*max_material/s
        
        #Now find the best performing vector of the batch
        batch_p = np.zeros(batch_size)
        for j in range(0, batch_size):
            temp_ar = ar
            temp_ar[dvars] = batches[j]
            
            K, F = create_globals(nx, ny, con, ld, temp_ar, fi)
            
            K_inv = np.linalg.inv(K)
            d = K_inv @ F
            
            batch_p[j] = np.linalg.norm(d)**2
        
        #take the best
        take = np.argmin(batch_p)
        ar[dvars] = batches[take]
        norms[i] = batch_p[take]
        
    fig, ax = plt.subplots(1,1, dpi = 400)
    ax.plot(norms)
    plt.show()
    #create an animation

#Line bridge
restricted_SGD(create_init_single, iters = 120, batch_size=20, disp = True)

#restricted_SGD(create_bracket, iters = 200, batch_size=15, disp = True)