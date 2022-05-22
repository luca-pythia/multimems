import numpy as np
import pandas as pd
#get 2D force array
import ndsplines
from scipy import interpolate

def add_boundaries(pos_el,fe_el,n1=1,n2=1):
    nonzero_idx = np.where((np.abs(fe_el) > 1))
    #nonzero_idx = np.where((fe_el < 0))
    if nonzero_idx[0].size != 0:

        pos_nonzero = pos_el[nonzero_idx]
        fe_nonzero = fe_el[nonzero_idx]

        zero_idx_left = np.where(pos_el <np.min(pos_nonzero))
        zero_idx_right = np.where(pos_el >np.max(pos_nonzero))

        if zero_idx_left[0].size != 0:

            pos_zero_left = pos_el[zero_idx_left]
            fe_zero_left = fe_el[zero_idx_left]
            fe_zero_left = (pos_zero_left-np.min(pos_nonzero))**2/(pos_el[1]-pos_el[0])**2
            fe_el[:len(zero_idx_left[0])] -= n1*fe_zero_left

        if zero_idx_right[0].size != 0:

            pos_zero_right = pos_el[zero_idx_right]
            fe_zero_right = fe_el[zero_idx_right]
            fe_zero_right = (pos_zero_right-np.max(pos_nonzero))**2/(pos_el[1]-pos_el[0])**2
            fe_el[zero_idx_right[0][0]:] += n2*fe_zero_right
            fe_el[-1] = 2*fe_el[-2]
        
    return fe_el

def get_du_nd(pos,fe,n_dim,kT=2.494,bounds= True,degrees=1):
    #fe*=kT
    #add quadratic boundaries (only works for 2d)      
    # generate grid of independent variables
    pos_list = pos.T.tolist()
    mesh= np.meshgrid(*pos_list,indexing='ij')
    mesh = np.vstack(([x.ravel() for x in mesh]))
    mesh = np.array(mesh).T
    bins = pos.shape[0]
    dim_list = []
    for i in range(n_dim):
        dim_list.append(bins)
    dim_list.append(n_dim)
    #mesh = mesh.reshape((*dim_list))
    mesh = mesh.reshape((dim_list))
    
    interp = ndsplines.make_interp_spline(mesh, fe,degrees=degrees)
    force_funcs = []
    #force_array = np.zeros((len(pos.T[0]),n_dim))
    
    pos_fine = np.zeros((n_dim,bins*2))
    delta = (np.mean((pos[1:] - pos[:-1]))/2)
    
    for i in range(n_dim):

        pos_fine[i] = np.linspace(np.min(pos.T[i])-delta,np.max(pos.T[i])+delta,bins*2)
    
    #bins_array= np.meshgrid(*pos_fine.tolist(),indexing='ij')
    #bins_array =np.array([bins_array]).reshape((n_dim,(bins*2)**n_dim)).T
    #print(xx.shape,bins_array.shape)
    #bins_array = np.array(sorted(bins_array.tolist(),key=lambda x:sum(x),reverse=False))
    
    for i in range(n_dim):
        deriv_interp = interp.derivative(i) #derivative of potential in ith direction
        force_funcs.append(deriv_interp)
            
    force_matrix = np.zeros((n_dim,len(pos_fine.T),len(pos_fine.T)))
    if n_dim > 1:
        
        for i in range(n_dim):
            for j in range(len(pos_fine.T)):
                for k in range(len(pos_fine.T)):
                    force_matrix[i][j][k] = force_funcs[i]([pos_fine[0][j],pos_fine[1][k]])
        
        if bounds and n_dim == 2:
        
            force_full = force_matrix.copy()
            for j in range(len(force_full[0])):
                force_full[0].T[j] = add_boundaries(pos_fine[0],force_full[0].T[j],n1=10,n2=10)
                force_full[1][j] = add_boundaries(pos_fine[1],force_full[1][j],n1=10,n2=10)
            force_matrix = force_full.copy()
    else:
        force_matrix[0] = force_funcs[0](pos_fine[0])
        
    return pos, fe, force_funcs, pos_fine, force_matrix


def extract_free_energy(x): 
    #one-dimensional
    hist,edges=np.histogram(x, bins=100, density=True)
    pos =(edges[1:]+edges[:-1])/2
    hist = hist[np.nonzero(hist)]
    pos = pos[np.nonzero(hist)]
    fe=-np.log(hist[np.nonzero(hist)]) #in units of kT!

    fe_spline=interpolate.splrep(pos, fe, s=0, per=0)
    force_array=interpolate.splev(x, fe_spline, der=1)

    return force_array

def spline_fe_for_sim(pos,fe,dx = 5, add_bounds=True,start=0):
    dxf=(pos[1]-pos[0])/dx
    fe_spline=interpolate.splrep(pos, fe, s=0, per=0)
    force_bins=np.arange(pos[0],pos[-1],dxf)
    force_matrix=interpolate.splev(force_bins, fe_spline, der=1)

    if add_bounds:
        #add quadratic boundary at right end
        xfine_new = np.arange(pos[0],pos[-1]*1.5,dxf)
        xfine_right = xfine_new[len(force_bins)-start*dx:]
        force_fine_right = (xfine_right-pos[-start])**2*(dxf)*0.001
        force_fine_right +=force_matrix[-start*dx]
        force_matrix = np.append(force_matrix[:-start*dx],force_fine_right)
        force_bins = xfine_new
    
    return fe_spline,force_bins,force_matrix
