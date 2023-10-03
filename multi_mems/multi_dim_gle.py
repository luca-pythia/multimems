import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, integrate, optimize

from multi_mems import *
from einsumt import einsumt as einsum

#import memtools as mt #for 1D force extraction (more robust in some cases)
import ndsplines #install for MV Potential Landscape https://pypi.org/project/ndsplines/

class multi_dim_gle:
    """Class for extracting correlation functions 
       and memory kernel matrix 
       for a network of trajectories (xva-dataframes)
    """
    
    def __init__(self, trunc, bins = 32, kT =2.494, free_energy = 'MV', verbose = True,plot = False,symm_matr = False,physical = True,diagonal_mass =False,force_funcs = None):
        
        self.trunc = trunc #depth of memory kernel entries
        self.bins = bins #number of bins for the histogram (potential-extraction)
  
        self.kT = kT #thermal energy of system in atomic units
        self.physical = physical
        self.symm_matr = symm_matr #we assume that v_acf on off-diagonal entries are the same (12 = 21, etc.)
        #self.kT_init = kT #ony for physical
        self.free_energy = free_energy #enables inclusion of mean force team in GLE, #if MV: Conditional Probability, #if ADD: Additional Assumption U(x1,x2,...) = U(x1) + U(x2) + ...
        self.verbose = verbose
        self.plot = plot #option to plot the resuls by matplotlib
        self.diagonal_mass = diagonal_mass #avoids additional coupling between observables
        self.force_funcs = force_funcs #if we already know the analytical shape of the potential landscape, we do not need to compute and interpolate it via the data
        
        
        
        
    def compute_free_energy_landscape(self,xvaf):
        self.n_dim = xvaf.filter(regex='^x',axis=1).shape[1]

        if self.free_energy == 'ADD': #1D Potentials
            pos_arrays = np.zeros((self.n_dim,self.bins))
            fe_arrays = np.zeros((self.n_dim,self.bins))
            for i in range(0, self.n_dim): 
                    pos, hist, fe, xfine, fe_fine, force_array = self.extract_free_energy(xvaf['x_' + str(i+1)])
                    pos_arrays[i] = pos
                    fe_arrays[i] = fe
                    
                    if self.plot:
                        plt.title(r'$x_' + str(i+1)+'$')
                        plt.plot(pos,fe,color='k')
                        plt.xlabel(r'$x_' + str(i+1)+'$')
                        plt.ylabel('$U/k_BT$')
                        plt.show()
                        
        elif self.free_energy == 'MV':
            xx = xvaf.filter(regex='^x',axis=1)
            xx = xx.to_numpy()
            pos,hist,fe,mesh,interp,mesh_fine,fe_fine,force_funcs = self.extract_free_energy_nD(xx)
            pos_arrays = pos
            fe_arrays = fe
            
            if self.plot and self.n_dim == 2:
                pos1 = pos.T[0]
                pos2 = pos.T[1]
                plt.imshow(fe-np.min(fe), cmap=plt.cm.jet, interpolation='spline16', extent = [np.min(pos1) , np.max(pos1), np.min(pos2) , np.max(pos2)])
                plt.colorbar()
                plt.xlabel(r'$x_1$')
                plt.ylabel(r'$x_2$')

                plt.show()    
        else:
            print('no free energy landscape calculated')
            pos_arrays = None
            fe_arrays = None
        return pos_arrays,fe_arrays
    
    #https://www.pnas.org/doi/abs/10.1073/pnas.2023856118
    def compute_correlations_G(self,xvaf): #all columns of xvaf has to include the same time steps!!
        #if mkl: #uses faster correlation function, but need to installed first
            #correlation = correlation_fast
        #self.n_dim = int(xvaf.shape[1]/4)
        self.n_dim = xvaf.filter(regex='^x',axis=1).shape[1]
        
        self.dt = xvaf.index[1] - xvaf.index[0]
        
        if self.verbose:
            print('dimension of system: ' + str(self.n_dim))
            print('found dt = ' + str(self.dt))
            
        tmax=int(self.trunc/self.dt)
        
        if self.verbose:
            print('calculate all correlation functions...')
        
        v_corr_matrix = np.zeros([tmax,self.n_dim,self.n_dim])
        xU_corr_matrix = np.zeros([tmax,self.n_dim,self.n_dim])
        
        for i in range(0,self.n_dim): #could be possibly accelerated by list comprehension
            for j in range(0,self.n_dim):
                #v_corr_matrix.T[i][j] = correlation(xvaf['v_' + str(i+1)],xvaf['v_' + str(j+1)])[:tmax]
                v_corr_matrix.T[j][i] = correlation(xvaf['v_' + str(i+1)],xvaf['v_' + str(j+1)])[:tmax] #attention i and j are switched in off-diagonal!
        
        force_funcs = []
        if self.free_energy == 'ADD': #additive composition
            
            self.kT_1D = 1
            for i in range(0,self.n_dim):
                for j in range(0, self.n_dim): 
                    
                    #xva = xvaf.filter(regex=str(i+1),axis=1)
                    #xva.columns = ['t', 'x', 'v', 'a']
                    #xva['t'] = np.arange(len(xva))*self.dt
                    
                    # in principle we could use memtools for this, since it should be more stable
                    #xf=mt.xframe(xvaf['x_' + str(j+1)],xvaf.index,fix_time=True)
                    #xva=mt.compute_va(xf,correct_jumps=True)
                    #mem = mt.Igle(xva, kT = self.kT_1D, trunc = self.trunc,saveall = False,verbose = False)    
                    #mem.compute_fe(bins=self.bins)

                    #dU=mem.dU
                    #force_funcs.append(dU)
                    #force_array = dU(xvaf['x_' + str(j+1)])
                    
                    pos, hist, fe, xfine, fe_fine, force_array = self.extract_free_energy(xvaf['x_' + str(i+1)])
                    xU_corr_matrix.T[j][i] = correlation(force_array,xvaf['x_' + str(j+1)])[:tmax]
                    #pos, hist, fe, xfine, fe_fine, force_array = self.extract_free_energy(xvaf['x_' + str(j+1)])
                    #xU_corr_matrix.T[j][i] = correlation(xvaf['x_' + str(i+1)],force_array)[:tmax]

        elif self.free_energy == 'MV':
            
            if self.verbose:
                print('calculate multi-variate free energy landscape')
            
                
            xx = xvaf.filter(regex='^x',axis=1)
            xx = xx.to_numpy()
            
            #if self.n_dim == 2: #not fixed yet!!
                #pos1,pos2,hist,fe,force1,force2,force_fine1,force2,force_array = self.extract_free_energy_2D(xvaf['x_1'].values,xvaf['x_2'].values)
                   
            #else:
            
            if self.force_funcs is None:
                #this nd splines function is from https://github.com/kb-press/ndsplines

                pos,hist,fe,mesh,interp,mesh_fine,fe_fine,force_funcs = self.extract_free_energy_nD(xx)
            else: #use a previously calculated free energy landscape
                #mesh,interp,mesh_fine,fe_fine,force_funcs= self.get_du_nD(self.fe_array[0],self.fe_array[1])
                force_funcs = self.force_funcs
            dU = force_funcs
            force_array = np.zeros((len(xvaf),self.n_dim))
            
            for j in range(self.n_dim): #maybe memory issues in ndsplines
                force_array.T[j] = dU[j](xx).reshape(len(xx))
            
            
            for i in range(0,self.n_dim):
                for j in range(0, self.n_dim): 
                    #xU_corr_matrix.T[i][j] = correlation(xvaf['x_' + str(i+1)],force_array.T[j])[:tmax]
                    #xU_corr_matrix.T[j][i] = correlation(force_array.T[j],xvaf['x_' + str(i+1)])[:tmax]
                    xU_corr_matrix.T[j][i] = correlation(force_array.T[i],xvaf['x_' + str(j+1)])[:tmax]
                    #xU_corr_matrix.T[j][i] = correlation(xvaf['x_' + str(i+1)],force_array.T[j])[:tmax]

        elif self.free_energy == "Mori":
            if self.verbose:
                print('calculate linear forces in Mori-GLE...')
            
            for i in range(0,self.n_dim): #could be possibly accelerated by list comprehension
                for j in range(0,self.n_dim):
                    xU_corr_matrix.T[j][i] = correlation(xvaf['x_' + str(i+1)]-np.mean(xvaf['x_' + str(i+1)]),xvaf['x_' + str(j+1)]-np.mean(xvaf['x_' + str(j+1)]))[:tmax]

            self.k_matrix = np.dot(v_corr_matrix[0],np.linalg.inv(xU_corr_matrix[0]))
            if self.verbose:
                print('constant k-matrix:') #note we do not use position-dependent masses!!
                print(self.k_matrix)

            for i in range(tmax):
                xU_corr_matrix[i] = np.dot(self.k_matrix,xU_corr_matrix[i])
                         
        t = np.arange(0,len(v_corr_matrix)*self.dt,self.dt)
        
        if self.symm_matr:
            for i in range(0,self.n_dim):
                for j in range(0, self.n_dim): 
                    if i != j:
                        v_corr_matrix.T[i][j] = (v_corr_matrix.T[i][j] + v_corr_matrix.T[j][i])/2
                        v_corr_matrix.T[j][i] = v_corr_matrix.T[i][j]

                        
        if self.plot:
            if self.verbose:
                print('plot correlation matrices...')
            if self.n_dim == 1:
                plt.plot(t,v_corr_matrix.T[0][0], color = 'k')
                plt.xscale('log')
                plt.ylabel(r'$C^{vv}$ [a.u.]')
                plt.xlabel('t')
                plt.axhline(y = 0, linestyle = '--', color = 'k')
                plt.show()
                
                plt.plot(t,xU_corr_matrix.T[0][0], color = 'k')
                plt.xscale('log')
                plt.ylabel(r'$C^{Fx}$ [a.u.]')
                plt.xlabel('t')
                plt.axhline(y = 0, linestyle = '--', color = 'k')
                plt.show()

            else:
                fig, ax = plt.subplots(self.n_dim,self.n_dim, figsize=(5*self.n_dim,3*self.n_dim))
                plt.subplots_adjust(wspace = 0.4)
                plt.subplots_adjust(hspace = 0.3)
                for i in range(0,self.n_dim):
                    for j in range(0, self.n_dim): 
                        ax[i][j].plot(t,v_corr_matrix.T[j][i], color = 'k')
                        ax[i][j].set_xscale('log')
                        ax[i][j].set_ylabel(r'$C^{vv}_{%s%s}$ [a.u.]' % (i+1, j+1))
                        ax[i][j].set_xlabel('t')

                        ax[i][j].axhline(y = 0, linestyle = '--', color = 'k')

                plt.show()
                
                fig, ax = plt.subplots(self.n_dim,self.n_dim, figsize=(5*self.n_dim,3*self.n_dim))
                plt.subplots_adjust(wspace = 0.4)
                plt.subplots_adjust(hspace = 0.3)
                for i in range(0,self.n_dim):
                    for j in range(0, self.n_dim): 
                        ax[i][j].plot(t,xU_corr_matrix.T[j][i], color = 'k')
                        ax[i][j].set_xscale('log')
                        ax[i][j].set_ylabel(r'$C^{Fx}_{%s%s}$ [a.u.]' % (i+1, j+1))
                        ax[i][j].set_xlabel('t')

                        ax[i][j].axhline(y = 0, linestyle = '--', color = 'k')

                plt.show()
                    
        return t, v_corr_matrix,xU_corr_matrix, force_funcs
    
    
    #the memory matrix extraction
    #https://www.pnas.org/doi/abs/10.1073/pnas.2023856118
    def compute_kernel_mat_G(self,v_corr_matrix,xU_corr_matrix,first_kind = True, d  = 0,multiprocessing=1):
        tmax=int(self.trunc/self.dt)
        ikernel_matrix = np.zeros([tmax,self.n_dim,self.n_dim])
        kernel_matrix = np.zeros([tmax,self.n_dim,self.n_dim])
        t = np.arange(0,len(kernel_matrix)*self.dt,self.dt)
        
        
        self.mass_matrix = self.kT*np.linalg.inv(v_corr_matrix[0])
        if self.physical == False:
            
            #self.mass_matrix/=self.mass_matrix
            self.mass_matrix = np.eye(self.n_dim)
            self.kT = v_corr_matrix[0]
            if self.verbose:
                print('use kT from v-acf')
        
        if self.verbose:
            print('used kT')
            print(self.kT)
            
        if self.diagonal_mass : #avoid additional coupling
            if self.verbose:
                print('use diagonal mass matrix')
            for i in range(0,self.n_dim):
                for j in range(0, self.n_dim):
                    if i != j:
                        self.mass_matrix[i][j] = 0
        
        if self.verbose:
            print('constant mass matrix:') #note we do not use position-dependent masses!!
            print(self.mass_matrix)
        
        
        prefac_mat = np.linalg.inv(v_corr_matrix[0])
       
        if self.verbose:
            print('extract memory kernel entries ...')
        for i in range(1,len(ikernel_matrix)):


                if first_kind:#recommended!!

                    if multiprocessing == 1:
                        ikernel_matrix[i] -= np.einsum('ijk,ikl->jl',ikernel_matrix[1:i+1],v_corr_matrix[:i][::-1])
                        #ikernel_matrix[i] -= np.einsum('ijk,ikl->jl',ikernel_matrix[1:i],v_corr_matrix[1:i][::-1])

                    else:
                        ikernel_matrix[i] -= einsum('ijk,ikl->jl',ikernel_matrix[1:i+1],v_corr_matrix[:i][::-1],pool=multiprocessing) #faster through parallelization
                        #ikernel_matrix[i] -= einsum('ijk,ikl->jl',ikernel_matrix[1:i],v_corr_matrix[1:i][::-1],pool=multiprocessing) #faster through parallelization

                    ikernel_matrix[i] -= np.dot(self.mass_matrix,v_corr_matrix[i]-v_corr_matrix[0])/self.dt


                    ikernel_matrix[i] -= np.dot(self.kT,xU_corr_matrix[0])/self.dt


                    ikernel_matrix[i] += np.dot(self.kT,xU_corr_matrix[i])/self.dt


                    ikernel_matrix[i] = np.matmul(ikernel_matrix[i],prefac_mat)*2


                else:
                    if multiprocessing == 1:
                        ikernel_matrix[i] -= np.einsum('ijk,ikl->jl',ikernel_matrix[1:i+1],v_corr_matrix[:i][::-1])
                        #ikernel_matrix[i] -= np.einsum('ijk,ikl->jl',ikernel_matrix[1:i],v_corr_matrix[1:i][::-1])

                    else:
                        ikernel_matrix[i] -= einsum('ijk,ikl->jl',ikernel_matrix[1:i+1],v_corr_matrix[:i][::-1],pool=multiprocessing) #faster through parallelization
                        #ikernel_matrix[i] -= einsum('ijk,ikl->jl',ikernel_matrix[1:i],v_corr_matrix[1:i][::-1],pool=multiprocessing) #faster through parallelization

                    ikernel_matrix[i] -= np.dot(np.dot(self.kT,xU_corr_matrix[0]),np.dot(np.linalg.inv(v_corr_matrix[0]),v_corr_matrix[i]))/self.dt


                    ikernel_matrix[i] += np.dot(self.kT,xU_corr_matrix[i])/self.dt


                    ikernel_matrix[i] = np.matmul(ikernel_matrix[i],prefac_mat)*2

        #calculate memory matrix from ikernel
        for i in range(0,self.n_dim):
                    for j in range(0, self.n_dim): 
                        if d == 1:
                            for n in range(len(kernel_matrix)-1):
                                kernel_matrix.T[i][j][n] = (ikernel_matrix.T[i][j][n+1] - ikernel_matrix.T[i][j][n])/self.dt
                        
                        
                        elif d==2: #predictor-corrector + central difference (https://www.nature.com/articles/s42005-020-0389-0)
                            #if self.verbose:
                                #print('use predictor-corrector scheme...')
                            ikernel_matrix_corrected = ikernel_matrix
                            for n in range(1,len(kernel_matrix)-1):
                                ikernel_matrix_corrected.T[i][j][n] = (ikernel_matrix_corrected.T[i][j][n-1] + 3*ikernel_matrix.T[i][j][n] + ikernel_matrix.T[i][j][n+1])/5
                            
                            kernel_matrix.T[i][j]  = np.gradient(ikernel_matrix_corrected.T[i][j],self.dt)  
                            ikernel_matrix = ikernel_matrix_corrected
                        else: #central difference
                            kernel_matrix.T[i][j]  = np.gradient(ikernel_matrix.T[i][j],self.dt)  




        if self.plot:
            if self.verbose:
                print('plot kernel entries...')
            if self.n_dim == 1:
                plt.plot(t[:-1],kernel_matrix.T[0][0][:-1], color = 'k')
                plt.xscale('log')
                plt.ylabel(r'$\Gamma$ [a.u.]')
                plt.xlabel('t')

                plt.axhline(y = 0, linestyle = '--', color = 'k')

                plt.show()

            else:
                fig, ax = plt.subplots(self.n_dim,self.n_dim, figsize=(5*self.n_dim,3*self.n_dim))
                plt.subplots_adjust(wspace = 0.4)
                plt.subplots_adjust(hspace = 0.3)
                for i in range(0,self.n_dim):
                    for j in range(0, self.n_dim): 
                        ax[i][j].plot(t[:-1],kernel_matrix.T[j][i][:-1], color = 'k')
                        ax[i][j].set_xscale('log')
                        ax[i][j].set_ylabel(r'$\Gamma_{%s%s}$ [a.u.]' % (i+1, j+1))
                        ax[i][j].set_xlabel('t')

                        ax[i][j].axhline(y = 0, linestyle = '--', color = 'k')

                plt.show()

        return t, ikernel_matrix, kernel_matrix

    #https://github.com/kb-press/ndsplines
    def extract_free_energy_nD(self,xx):
        self.n_dim = xx.shape[1]

        #multi_d histogram
        hist = np.histogramdd(xx, bins=self.bins,density=True)
        pos = np.zeros((self.bins,self.n_dim))
        for i in range(self.n_dim):

            pos.T[i] =(hist[1][i][1:]+hist[1][i][:-1])/2
        hist = hist[0]

        fe=-np.log(hist)
        fe[np.where(fe == np.inf)] = np.nanmax(fe[fe != np.inf]) #in general zero
        fe-=np.min(fe)

        # generate grid of independent variables
        pos_list = pos.T.tolist()
        mesh= np.meshgrid(*pos_list,indexing='ij')
        
        mesh = np.vstack(([x.ravel() for x in mesh]))
        mesh = np.array(mesh).T


        dim_list = []
        for i in range(self.n_dim):
            dim_list.append(self.bins)
        dim_list.append(self.n_dim)


        #mesh = mesh.reshape((*dim_list))
        mesh = mesh.reshape((dim_list))

        # create the interpolating spline
        interp = ndsplines.make_interp_spline(mesh, fe)

        # generate denser grid of independent variables to interpolate (not implemented yet...)
        mesh_fine = 0
        fe_fine = 0
        

        # evaluate derivative at point (x,y) (at x-direction)
        force_funcs = []
        for i in range(self.n_dim):
            deriv_interp = interp.derivative(i) #derivative of potential in ith direction
            force_funcs.append(deriv_interp)

        return pos,hist,fe,mesh,interp,mesh_fine,fe_fine,force_funcs
    
    def get_du_nD(self,pos,fe):
        self.n_dim = len(pos)

        fe[np.where(fe == np.inf)] = np.nanmax(fe[fe != np.inf]) #in general zero

        # generate grid of independent variables
        pos_list = pos.T.tolist()
        mesh= np.meshgrid(*pos_list,indexing='ij')
        mesh = np.vstack(([x.ravel() for x in mesh]))
        mesh = np.array(mesh).T


        dim_list = []
        for i in range(self.n_dim):
            dim_list.append(len(pos)[0])
        dim_list.append(self.n_dim)


        #mesh = mesh.reshape((*dim_list))
        mesh = mesh.reshape((dim_list))

        # create the interpolating spline
        interp = ndsplines.make_interp_spline(mesh, fe)

        # generate denser grid of independent variables to interpolate (not implemented yet...)
        mesh_fine = 0
        fe_fine = 0
        

        # evaluate derivative at point (x,y) (at x-direction)
        force_funcs = []
        for i in range(self.n_dim):
            deriv_interp = interp.derivative(i) #derivative of potential in ith direction
            force_funcs.append(deriv_interp)

        return mesh,interp,mesh_fine,fe_fine,force_funcs
    
    def extract_free_energy(self,x): 
        #one-dimensional, we assume the potential landscape to be additive, i.e. U(x) = U(x1) + ..+ U(xn)
        hist,edges=np.histogram(x, bins=self.bins, density=True)
        pos =(edges[1:]+edges[:-1])/2
        
        pos = pos[np.nonzero(hist)]
        hist = hist[np.nonzero(hist)]
        
        fe=-np.log(hist[np.nonzero(hist)])
        

        fe_spline=interpolate.splrep(pos, fe, s=0, per=0)

        dxf=pos[1]-pos[0]
        xfine=np.arange(pos[0],pos[-1],dxf/10.)
        fe_fine=interpolate.splev(xfine, fe_spline)
        force_fine=interpolate.splev(xfine, fe_spline, der=1)
        force_array=interpolate.splev(x, fe_spline, der=1)

        return pos, hist, fe, xfine, fe_fine, force_array
    
    
    def extract_free_energy_2D(self,x1,x2):
        
        hist, pos1, pos2 = np.histogram2d(x1, x2, bins=(self.bins, self.bins))

        pos1 =(pos1[1:]+pos1[:-1])/2

        pos2 =(pos2[1:]+pos2[:-1])/2
        #hist = hist[np.nonzero(hist)]
        #pos = pos[np.nonzero(hist)]
        fe=-np.log(hist)#*self.kT
        fe[np.where(fe == np.inf)] = np.nanmax(fe[fe != np.inf])

        forces = []
        grad = np.array(np.gradient(fe))

        grad1 = grad[0]
        grad2 = grad[1]
        dx1 = pos1[1] - pos1[0]
        dx2 = pos2[1] - pos2[0]

        force1 = grad1/dx1
        force2 = grad2/dx2



        #pos1_grid,pos2_grid = np.mgrid[np.min(pos1):np.max(pos1)+dx1:dx1, np.min(pos2):np.max(pos2)+dx2:dx2]
        pos1_grid,pos2_grid = np.meshgrid(pos1, pos2)     


        force_func1 = interpolate.bisplrep(pos1_grid, pos2_grid, force1, s=0)
        force_func2 = interpolate.bisplrep(pos1_grid, pos2_grid, force2, s=0)

        force_array = np.zeros((len(x1),self.n_dim))

        for i in range(len(force_array)):
        
            force_array[i][0] = interpolate.bisplev(x1[i], x2[i], force_func1)
            force_array[i][1] = interpolate.bisplev(x1[i], x2[i], force_func2)

        return pos1,pos2,hist,fe,force1,force2,force_func1,force_func2,force_array

            
        if self.plot:
            plt.imshow(fe, cmap=plt.cm.jet, interpolation='spline16', extent = [np.min(pos1) , np.max(pos1), np.min(pos2) , np.max(pos2)])
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.colorbar()
            plt.show()

        return pos1,pos2,hist,fe,fe_spline, xfine1,xfine2, fe_fine,force_funcs
    
    
    def smooth_data(self,t,data,start=0,end = -1,step =1):
        x = data.copy()
        if end == -1:
            end = len(x)
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                xx = x.T[i][j][start:end]
                
                tt = t[start:end]

                x_points = np.zeros(int(len(xx)/step))
                x_points[0] = xx[0]

                t_points = np.zeros(int(len(tt)/step))
                t_points[0] = tt[0]
                for k in range(1,len(x_points)):
                    x_points[k] = np.mean(xx[int(step*(1+2*(k-1)/2)):int(step*(1+2*(k)/2))])
                    t_points[k] = np.mean(tt[int(step*(1+2*(k-1)/2)):int(step*(1+2*(k)/2))])

                spl = interpolate.UnivariateSpline(t_points, x_points,k=2,s=0)
                
                x.T[i][j][start:end]= spl(tt)
                
        return x


    #Jan Daldrop's method 
    #https://www.pnas.org/doi/abs/10.1073/pnas.1722327115

    def compute_correlations_direct(self,xvaf): #all columns of xvaf has to include the same time steps!!
        #if mkl: #uses faster correlation function, but need to installed first
            #correlation = correlation_fast
        #self.n_dim = int(xvaf.shape[1]/4)
        self.n_dim = xvaf.filter(regex='^x',axis=1).shape[1]
        
        self.dt = xvaf.index[1] - xvaf.index[0]
        
        if self.verbose:
            print('dimension of system: ' + str(self.n_dim))
            print('found dt = ' + str(self.dt))
            
        tmax=int(self.trunc/self.dt)
        
        if self.verbose:
            print('calculate all correlation functions...')
        
        v_corr_matrix = np.zeros([tmax,self.n_dim,self.n_dim])
        a_corr_matrix = np.zeros([tmax,self.n_dim,self.n_dim])
        va_corr_matrix = np.zeros([tmax,self.n_dim,self.n_dim])
        
        vU_corr_matrix = np.zeros([tmax,self.n_dim,self.n_dim])
        aU_corr_matrix = np.zeros([tmax,self.n_dim,self.n_dim])
        
        for i in range(0,self.n_dim): #could be possibly accelerated by list comprehension
            for j in range(0,self.n_dim):

                v_corr_matrix.T[j][i] = correlation(xvaf['v_' + str(i+1)],xvaf['v_' + str(j+1)])[:tmax] #attention i and j are switched in off-diagonal!
                a_corr_matrix.T[j][i] = correlation(xvaf['a_' + str(i+1)],xvaf['a_' + str(j+1)])[:tmax] #attention i and j are switched in off-diagonal!
                #va_corr_matrix.T[j][i] = correlation(xvaf['a_' + str(i+1)],xvaf['v_' + str(j+1)])[:tmax] #attention i and j are switched in off-diagonal!
                va_corr_matrix.T[j][i] = correlation(xvaf['v_' + str(i+1)],xvaf['a_' + str(j+1)])[:tmax] #attention i and j are switched in off-diagonal!

        force_funcs = []
        if self.free_energy == 'ADD': #additive composition
            
            self.kT_1D = 1
            for i in range(0,self.n_dim):
                for j in range(0, self.n_dim): 
                    
                    #pos, hist, fe, xfine, fe_fine, force_array = self.extract_free_energy(xvaf['x_' + str(i+1)])
                    #vU_corr_matrix.T[j][i] = correlation(force_array,xvaf['v_' + str(j+1)])[:tmax]
                    #aU_corr_matrix.T[j][i] = correlation(force_array,xvaf['a_' + str(j+1)])[:tmax]

                    pos, hist, fe, xfine, fe_fine, force_array = self.extract_free_energy(xvaf['x_' + str(j+1)])
                    vU_corr_matrix.T[j][i] = correlation(xvaf['v_' + str(i+1)],force_array)[:tmax]
                    aU_corr_matrix.T[j][i] = correlation(xvaf['a_' + str(i+1)],force_array)[:tmax]

                   

        elif self.free_energy == 'MV':
            
            if self.verbose:
                print('calculate multi-variate free energy landscape')
            
                
            xx = xvaf.filter(regex='^x',axis=1)
            xx = xx.to_numpy()
            
            #if self.n_dim == 2: #not fixed yet!!
                #pos1,pos2,hist,fe,force1,force2,force_fine1,force2,force_array = self.extract_free_energy_2D(xvaf['x_1'].values,xvaf['x_2'].values)
                   
            #else:
            
            if self.force_funcs is None:
                #this nd splines function is from https://github.com/kb-press/ndsplines

                pos,hist,fe,mesh,interp,mesh_fine,fe_fine,force_funcs = self.extract_free_energy_nD(xx)
            else: #use a previously calculated free energy landscape
                #mesh,interp,mesh_fine,fe_fine,force_funcs= self.get_du_nD(self.fe_array[0],self.fe_array[1])
                force_funcs = self.force_funcs
            dU = force_funcs
            force_array = np.zeros((len(xvaf),self.n_dim))
            
            for j in range(self.n_dim): #maybe memory issues in ndsplines
                force_array.T[j] = dU[j](xx).reshape(len(xx))
            
            
            for i in range(0,self.n_dim):
                for j in range(0, self.n_dim): 
                    #vU_corr_matrix.T[j][i] = correlation(force_array.T[i],xvaf['v_' + str(j+1)])[:tmax]
                    #aU_corr_matrix.T[j][i] = correlation(force_array.T[i],xvaf['a_' + str(j+1)])[:tmax]
                    vU_corr_matrix.T[j][i] = correlation(xvaf['v_' + str(i+1)],force_array.T[j],)[:tmax]
                    aU_corr_matrix.T[j][i] = correlation(xvaf['a_' + str(i+1)],force_array.T[j],)[:tmax]

        elif self.free_energy == "Mori":

            x_corr_matrix = np.zeros([tmax,self.n_dim,self.n_dim])

            if self.verbose:
                print('calculate linear forces in Mori-GLE...')
            
            for i in range(0,self.n_dim): #could be possibly accelerated by list comprehension
                for j in range(0,self.n_dim):
                    x_corr_matrix.T[j][i] = correlation(xvaf['x_' + str(i+1)]-np.mean(xvaf['x_' + str(i+1)]),xvaf['x_' + str(j+1)]-np.mean(xvaf['x_' + str(j+1)]))[:tmax]
                    vU_corr_matrix.T[j][i] = correlation(xvaf['v_' + str(i+1)],xvaf['x_' + str(j+1)]-np.mean(xvaf['x_' + str(j+1)]))[:tmax]
                    aU_corr_matrix.T[j][i] = correlation(xvaf['a_' + str(i+1)],xvaf['x_' + str(j+1)]-np.mean(xvaf['x_' + str(j+1)]))[:tmax]

            self.k_matrix = np.dot(v_corr_matrix[0],np.linalg.inv(x_corr_matrix[0]))
            if self.verbose:
                print('constant k-matrix:') #note we do not use position-dependent masses!!
                print(self.k_matrix)

            for i in range(tmax):
                vU_corr_matrix[i] = np.dot(self.k_matrix,vU_corr_matrix[i])
                aU_corr_matrix[i] = np.dot(self.k_matrix,aU_corr_matrix[i])
   
        t = np.arange(0,len(v_corr_matrix)*self.dt,self.dt)
        
        if self.symm_matr:
            for i in range(0,self.n_dim):
                for j in range(0, self.n_dim): 
                    if i != j:
                        v_corr_matrix.T[i][j] = (v_corr_matrix.T[i][j] + v_corr_matrix.T[j][i])/2
                        a_corr_matrix.T[i][j] = (a_corr_matrix.T[i][j] + a_corr_matrix.T[j][i])/2
                        va_corr_matrix.T[i][j] = (va_corr_matrix.T[i][j] + va_corr_matrix.T[j][i])/2

                        v_corr_matrix.T[j][i] = v_corr_matrix.T[i][j] 
                        a_corr_matrix.T[j][i] = a_corr_matrix.T[i][j] 
                        va_corr_matrix.T[j][i] = va_corr_matrix.T[i][j] 
                        
                        
        if self.plot:
            if self.verbose:
                print('plot correlation matrices...')
            if self.n_dim == 1:
                plt.plot(t,v_corr_matrix.T[0][0], color = 'k')
                plt.xscale('log')
                plt.ylabel(r'$C^{vv}$ [a.u.]')
                plt.xlabel('t')
                plt.axhline(y = 0, linestyle = '--', color = 'k')
                plt.show()
                
                plt.plot(t,a_corr_matrix.T[0][0], color = 'k')
                plt.xscale('log')
                plt.ylabel(r'$C^{aa}$ [a.u.]')
                plt.xlabel('t')
                plt.axhline(y = 0, linestyle = '--', color = 'k')
                plt.show()

                plt.plot(t,va_corr_matrix.T[0][0], color = 'k')
                plt.xscale('log')
                plt.ylabel(r'$C^{va}$ [a.u.]')
                plt.xlabel('t')
                plt.axhline(y = 0, linestyle = '--', color = 'k')
                plt.show()

                plt.plot(t,vU_corr_matrix.T[0][0], color = 'k')
                plt.xscale('log')
                plt.ylabel(r'$C^{Fv}$ [a.u.]')
                plt.xlabel('t')
                plt.axhline(y = 0, linestyle = '--', color = 'k')
                plt.show()

                plt.plot(t,aU_corr_matrix.T[0][0], color = 'k')
                plt.xscale('log')
                plt.ylabel(r'$C^{Fa}$ [a.u.]')
                plt.xlabel('t')
                plt.axhline(y = 0, linestyle = '--', color = 'k')
                plt.show()

            else:
                fig, ax = plt.subplots(self.n_dim,self.n_dim, figsize=(5*self.n_dim,3*self.n_dim))
                plt.subplots_adjust(wspace = 0.4)
                plt.subplots_adjust(hspace = 0.3)
                for i in range(0,self.n_dim):
                    for j in range(0, self.n_dim): 
                        ax[i][j].plot(t,v_corr_matrix.T[j][i], color = 'k')
                        ax[i][j].set_xscale('log')
                        ax[i][j].set_ylabel(r'$C^{vv}_{%s%s}$ [a.u.]' % (i+1, j+1))
                        ax[i][j].set_xlabel('t')

                        ax[i][j].axhline(y = 0, linestyle = '--', color = 'k')

                plt.show()

                fig, ax = plt.subplots(self.n_dim,self.n_dim, figsize=(5*self.n_dim,3*self.n_dim))
                plt.subplots_adjust(wspace = 0.4)
                plt.subplots_adjust(hspace = 0.3)
                for i in range(0,self.n_dim):
                    for j in range(0, self.n_dim): 
                        ax[i][j].plot(t,va_corr_matrix.T[j][i], color = 'k')
                        ax[i][j].set_xscale('log')
                        ax[i][j].set_ylabel(r'$C^{va}_{%s%s}$ [a.u.]' % (i+1, j+1))
                        ax[i][j].set_xlabel('t')

                        ax[i][j].axhline(y = 0, linestyle = '--', color = 'k')

                plt.show()

                fig, ax = plt.subplots(self.n_dim,self.n_dim, figsize=(5*self.n_dim,3*self.n_dim))
                plt.subplots_adjust(wspace = 0.4)
                plt.subplots_adjust(hspace = 0.3)
                for i in range(0,self.n_dim):
                    for j in range(0, self.n_dim): 
                        ax[i][j].plot(t,a_corr_matrix.T[j][i], color = 'k')
                        ax[i][j].set_xscale('log')
                        ax[i][j].set_ylabel(r'$C^{aa}_{%s%s}$ [a.u.]' % (i+1, j+1))
                        ax[i][j].set_xlabel('t')

                        ax[i][j].axhline(y = 0, linestyle = '--', color = 'k')

                plt.show()
                
                fig, ax = plt.subplots(self.n_dim,self.n_dim, figsize=(5*self.n_dim,3*self.n_dim))
                plt.subplots_adjust(wspace = 0.4)
                plt.subplots_adjust(hspace = 0.3)
                for i in range(0,self.n_dim):
                    for j in range(0, self.n_dim): 
                        ax[i][j].plot(t,vU_corr_matrix.T[j][i], color = 'k')
                        ax[i][j].set_xscale('log')
                        ax[i][j].set_ylabel(r'$C^{vF}_{%s%s}$ [a.u.]' % (i+1, j+1))
                        ax[i][j].set_xlabel('t')

                        ax[i][j].axhline(y = 0, linestyle = '--', color = 'k')

                plt.show()

                fig, ax = plt.subplots(self.n_dim,self.n_dim, figsize=(5*self.n_dim,3*self.n_dim))
                plt.subplots_adjust(wspace = 0.4)
                plt.subplots_adjust(hspace = 0.3)
                for i in range(0,self.n_dim):
                    for j in range(0, self.n_dim): 
                        ax[i][j].plot(t,aU_corr_matrix.T[j][i], color = 'k')
                        ax[i][j].set_xscale('log')
                        ax[i][j].set_ylabel(r'$C^{aF}_{%s%s}$ [a.u.]' % (i+1, j+1))
                        ax[i][j].set_xlabel('t')

                        ax[i][j].axhline(y = 0, linestyle = '--', color = 'k')

                plt.show()
                    
        return t, v_corr_matrix,va_corr_matrix,a_corr_matrix,vU_corr_matrix,aU_corr_matrix, force_funcs

    def compute_kernel_mat_direct(self,v_corr_matrix,va_corr_matrix,a_corr_matrix,vU_corr_matrix,aU_corr_matrix,first_kind = True,multiprocessing=1,correct=True):
        
    
        tmax=int(self.trunc/self.dt)
        ikernel_matrix = np.zeros([tmax,self.n_dim,self.n_dim])
        kernel_matrix = np.zeros([tmax,self.n_dim,self.n_dim])
        t = np.arange(0,len(kernel_matrix)*self.dt,self.dt)
        
        
        self.mass_matrix = self.kT*np.linalg.inv(v_corr_matrix[0])
        if self.physical == False:
            
            #self.mass_matrix/=self.mass_matrix
            self.mass_matrix = np.eye(self.n_dim)
            self.kT = v_corr_matrix[0]
            if self.verbose:
                print('use kT from v-acf')
        
        if self.verbose:
            print('used kT')
            print(self.kT)
            
        if self.diagonal_mass : #avoid additional coupling
            if self.verbose:
                print('use diagonal mass matrix')
            for i in range(0,self.n_dim):
                for j in range(0, self.n_dim):
                    if i != j:
                        self.mass_matrix[i][j] = 0
        
        if self.verbose:
            print('constant mass matrix:') #note we do not use position-dependent masses!!
            print(self.mass_matrix)
        
            
        #aU_corr_matrix*=-1
        kernel_matrix[0] = np.dot(np.dot(self.mass_matrix,a_corr_matrix[0]) + np.dot(self.kT,aU_corr_matrix[0]),np.linalg.inv(v_corr_matrix[0]))
        #kernel_matrix[0] = np.dot(np.dot(self.mass_matrix,a_corr_matrix[0]) - np.dot(self.kT,aU_corr_matrix[0]),np.linalg.inv(v_corr_matrix[0]))

        if self.verbose:
            print('initial memory matrix:')
            print(kernel_matrix[0])

        prefac_mat = np.linalg.inv(v_corr_matrix[0])
        if self.verbose:
            print('extract memory kernel entries...')
        for i in range(1,len(kernel_matrix)):
            if first_kind:
                if multiprocessing == 1:
                    kernel_matrix[i] += np.einsum('ijk,ikl->jl',kernel_matrix[1:i+1],v_corr_matrix[:i][::-1])
                    #kernel_matrix[i] += np.einsum('ijk,ikl->jl',kernel_matrix[1:i],v_corr_matrix[1:i][::-1])

                else:
                    kernel_matrix[i] += einsum('ijk,ikl->jl',kernel_matrix[1:i+1],v_corr_matrix[:i][::-1],pool=multiprocessing) #faster through parallelization
                    #kernel_matrix[i] += einsum('ijk,ikl->jl',kernel_matrix[1:i],v_corr_matrix[1:i][::-1],pool=multiprocessing) #faster through parallelization

                kernel_matrix[i] += np.dot(self.mass_matrix,va_corr_matrix[i])/self.dt 

                kernel_matrix[i] += np.dot(self.kT,vU_corr_matrix[i])/self.dt 
                kernel_matrix[i] +=np.dot(kernel_matrix[0],v_corr_matrix[i])/2

                kernel_matrix[i] = np.dot(kernel_matrix[i],prefac_mat)*-2
            else:#recommended!!
                prefac_mat = np.linalg.inv(v_corr_matrix[0] - self.dt*0.5*va_corr_matrix[0])
                if multiprocessing == 1:
                    kernel_matrix[i] += np.einsum('ijk,ikl->jl',kernel_matrix[1:i+1],v_corr_matrix[:i][::-1])
                    #kernel_matrix[i] += np.einsum('ijk,ikl->jl',kernel_matrix[1:i],v_corr_matrix[1:i][::-1])

                else:
                    kernel_matrix[i] += einsum('ijk,ikl->jl',kernel_matrix[1:i+1],v_corr_matrix[:i][::-1],pool=multiprocessing) #faster through parallelization
                    #kernel_matrix[i] += einsum('ijk,ikl->jl',kernel_matrix[1:i],v_corr_matrix[1:i][::-1],pool=multiprocessing) #faster through parallelization

                kernel_matrix[i] += self.dt*np.dot(kernel_matrix[0],va_corr_matrix[i])/2
                kernel_matrix[i] += np.dot(self.mass_matrix,a_corr_matrix[i]) 
                kernel_matrix[i] += np.dot(self.kT,aU_corr_matrix[i])

                kernel_matrix[i] = np.dot(kernel_matrix[i],prefac_mat)
        t = np.arange(0,len(kernel_matrix)*self.dt,self.dt)

        for i in range(0,self.n_dim):
                for j in range(0, self.n_dim):
                    ikernel_matrix.T[i][j] = integrate.cumtrapz(kernel_matrix.T[i][j],t,initial=0)

        if correct:
            for i in range(0,self.n_dim):
                for j in range(0, self.n_dim):
                    kernel_matrix.T[i][j] = np.gradient(ikernel_matrix.T[i][j],self.dt)
        
        if self.plot:
            if self.verbose:
                print('plot kernel entries...')
            if self.n_dim == 1:
                plt.plot(t[:-1],kernel_matrix.T[0][0][:-1], color = 'k')
                plt.xscale('log')
                plt.ylabel(r'$\Gamma$ [a.u.]')
                plt.xlabel('t')

                plt.axhline(y = 0, linestyle = '--', color = 'k')

                plt.show()
            
            else:
                fig, ax = plt.subplots(self.n_dim,self.n_dim, figsize=(5*self.n_dim,3*self.n_dim))
                plt.subplots_adjust(wspace = 0.4)
                plt.subplots_adjust(hspace = 0.3)
                for i in range(0,self.n_dim):
                    for j in range(0, self.n_dim): 
                        ax[i][j].plot(t[:-1],kernel_matrix.T[j][i][:-1], color = 'k')
                        ax[i][j].set_xscale('log')
                        ax[i][j].set_ylabel(r'$\Gamma_{%s%s}$ [a.u.]' % (i+1, j+1))
                        ax[i][j].set_xlabel('t')

                        ax[i][j].axhline(y = 0, linestyle = '--', color = 'k')

                plt.show()
            
        return t, ikernel_matrix, kernel_matrix
    