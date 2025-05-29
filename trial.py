'''
The code is inpired by https://github.com/befelix/SafeOpt/tree/master
The random RKHS fucntion generation is inspired by https://github.com/Data-Science-in-Mechanical-Engineering/UncertaintyBounds21
Both corresponding works are cited in the main text.
---
Anonymous authors.
'''


import torch
import torch.nn as nn
import gpytorch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from tqdm import tqdm
import time
import warnings
import copy
import multiprocessing
import tikzplotlib
import pickle
import dill
from matplotlib.patches import Ellipse
import torch.multiprocessing as mp
from scipy.special import comb
from plot import plot_1D, plot_2D_contour, plot_1D_SafeOpt_with_sets, plot_gym, plot_gym_together
# from ground_truth_experiment import ground_truth_experiment
# import gym
import sys
import os
import time
import random


class MultiLayerRNN(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes):
        super(MultiLayerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Define the first input branch RNN
        self.rnn1 = nn.LSTM(input_size=50, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Define the second input branch RNN
        self.rnn2 = nn.LSTM(input_size=50, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Merge layer
        self.merge_layer = nn.Linear(hidden_size * 2, hidden_size)
        # Output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x1, x2):
        # Forward pass for the first input branch
        out1, _ = self.rnn1(x1)
        # Forward pass for the second input branch
        out2, _ = self.rnn2(x2)
        # Concatenate the outputs of both branches
        out = torch.cat((out1, out2), dim=2) ### CHANGED
        # Merge layer
        out = self.merge_layer(out)
        # Output layer
        out = self.fc(out)
        return out
    
def load_model(model_path, hidden_size, num_layers, num_classes):
    model = MultiLayerRNN(hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

def predict(model, input1, input2):
    model.eval()
    if isinstance(input1, list):
        input1 = input1[-1]
    if isinstance(input2, list):
        input2 = input2[-1]
    input1_tensor = torch.tensor(input1).unsqueeze(0).unsqueeze(0)  # shape [1, 1, 1]
    input2_tensor = torch.tensor(input2).unsqueeze(0).unsqueeze(0)  # shape [1, 1, 1]
    # repeat dims 50 times to match the expected input ssize
    input1_tensor = input1_tensor.repeat(1, 1, 50)  # Now shape: [1, 1, 50]
    input2_tensor = input2_tensor.repeat(1, 1, 50)  # Now shape: [1, 1, 50]

    with torch.no_grad():
        output = model(input1_tensor, input2_tensor)
        print(f"output: {output}")
    return output.item()


class GPRegressionModel(gpytorch.models.ExactGP):  # this model has to be build "new"
    def __init__(self, train_x, train_y, noise_std, n_devices=1, output_device=torch.device('cpu'), lengthscale=0.1):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.tensor(noise_std**2)
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        self.kernel.lengthscale = lengthscale
        # self.base_kernel.lengthscale.requires_grad = False; somehow does not work
        if output_device.type != 'cpu':
            self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                self.kernel, device_ids=range(n_devices), output_device=output_device)
        else:
            self.covar_module = self.kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def reshape_with_extra_dims(tensor, num_dims):
    # Calculate the number of extra dimensions needed
    extra_dims = [1] * (num_dims - tensor.dim())  # Adjust extra dimensions based on the tensor's shape
    # Reshape the tensor with extra dimensions
    reshaped_tensor = tensor.unsqueeze(*extra_dims).float()  # Convert tensor to floating-point
    return reshaped_tensor


def convert_to_hashable(item):
    if isinstance(item, (torch.Tensor, np.ndarray)):
        return tuple(map(tuple, item.tolist()))
    elif isinstance(item, tuple):
        return tuple(convert_to_hashable(i) for i in item)
    else:
        return item


def compute_X_plot(n_dimensions, points_per_axis):
    X_plot_per_domain = torch.linspace(0, 1, points_per_axis)
    X_plot_per_domain_nd = [X_plot_per_domain] * n_dimensions
    X_plot = torch.cartesian_prod(*X_plot_per_domain_nd).reshape(-1, n_dimensions)
    return X_plot


def initial_safe_samples(gt, num_safe_points):  # for toy examples and introductory example
    fX = gt.fX
    num_safe_points = num_safe_points
    # sampling_logic = fX > gt.safety_threshold  # alternative option
    sampling_logic = torch.logical_and(fX > np.quantile(fX, 0.4), fX < np.quantile(fX, 0.50))
    random_indices_sample = torch.randint(high=X_plot[sampling_logic].shape[0], size=(num_safe_points,))
    X_sample = X_plot[sampling_logic][random_indices_sample]
    Y_sample = fX[sampling_logic][random_indices_sample] + torch.tensor(np.random.normal(loc=0, scale=noise_std, size=X_sample.shape[0]), dtype=torch.float32)
    return X_sample, Y_sample


class ground_truth():
    def __init__(self, num_center_points, X_plot, RKHS_norm, seed=None):
        def fun(kernel, alpha):
            return lambda X: kernel(X.reshape(-1, self.X_center.shape[1]), self.X_center).detach().numpy() @ alpha
        # For ground truth
        self.X_plot = X_plot
        self.RKHS_norm = RKHS_norm
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        random_indices_center = torch.randint(high=self.X_plot.shape[0], size=(num_center_points,))
        self.X_center = self.X_plot[random_indices_center]
        alpha = np.random.uniform(-1, 1, size=self.X_center.shape[0])
        self.kernel = gpytorch.kernels.MaternKernel(nu=3/2)
        self.kernel.lengthscale = 0.1  # used in all runs except for Furuta hardware. There, this will get over-written.
        RKHS_norm_squared = alpha.T @ self.kernel(self.X_center, self.X_center).detach().numpy() @ alpha
        alpha /= np.sqrt(RKHS_norm_squared)/RKHS_norm  # scale to RKHS norm
        self.f = fun(self.kernel, alpha)
        self.fX = torch.tensor(self.f(self.X_plot), dtype=torch.float32)
        self.safety_threshold = np.quantile(self.fX, 0.3)  # np.quantile(self.fX, np.random.uniform(low=0.15, high=0.5))  # alternative option

    def conduct_experiment(self, x, noise_std):
        return torch.tensor(self.f(x) + np.random.normal(loc=0, scale=noise_std, size=1), dtype=x.dtype)

    def local_RKHS_norm(self, lb, ub, X_plot_local=None):  # needed for getting training labels
        nugget_factor = 1e-4  # regularization
        # Returns RKHS norm of ground truth on local domain between lb and ub; heuristic kernel interpolation approach
        if X_plot_local is None:
            local_gt_indices = torch.all(torch.logical_and(self.X_plot >= lb, self.X_plot <= ub), axis=1)
            if sum(local_gt_indices) > 10000:  # the problem is that the matrix gets very high dimensional and we cannot explicitly compute it
                subset_boolean = torch.randperm(sum(local_gt_indices)) < 10000
                X_local = self.X_plot[local_gt_indices][subset_boolean]
                fX_local = self.fX[local_gt_indices][subset_boolean]
            else:
                X_local = self.X_plot[local_gt_indices]
                fX_local = self.fX[local_gt_indices]
        else:  # use the local X_plots
            if X_plot.shape[0] > 10000:
                X_local = X_plot_local[torch.randperm(X_plot_local.shape[0]) < 10000]
                fX_local = torch.tensor(self.f(X_local), dtype=X_local.dtype)
            else:
                X_local = X_plot_local
                fX_local = torch.tensor(self.f(X_local), dtype=X_local.dtype)
        K_local = self.kernel(X_local, X_local).evaluate() + torch.eye(X_local.shape[0])*nugget_factor  # add a small nugget factor
        local_RKHS_norm_value = torch.sqrt(fX_local.reshape(1, -1) @ torch.inverse(K_local) @ fX_local.reshape(-1, 1)).flatten()  # inverse computation takes a lot of time but does not need to be done often. True RKHS norm given eigentlich once. And again also just for training
        while not local_RKHS_norm_value > 0:
            nugget_factor *= 10
            K_local = self.kernel(X_local, X_local).evaluate() + torch.eye(X_local.shape[0])*nugget_factor  # add a small nugget factor
            local_RKHS_norm_value = torch.sqrt(fX_local.reshape(1, -1) @ torch.inverse(K_local) @ fX_local.reshape(-1, 1)).flatten()
        return local_RKHS_norm_value


class safe_BO():
    def __init__(self, delta_confidence, delta_cube, noise_std, grid_coords, X_plot, X_sample,
                Y_sample, safety_threshold, exploration_threshold, gt, compute_local_X_plot, compute_all_sets=False):
        def compute_X_plot_locally(n_dimensions, points_per_axis, lb, ub):  # for scalability; discretization within each sub-domain separately
            X_plot = []
            for i in range(n_dimensions):
                X_plot_per_domain = torch.linspace(lb[i], ub[i], points_per_axis)
                X_plot.append(X_plot_per_domain)
            X_plot = torch.cartesian_prod(*X_plot).reshape(-1, n_dimensions)
            return X_plot
            
        self.compute_all_sets = compute_all_sets
        self.gt = gt
        self.exploration_threshold = exploration_threshold
        self.delta_confidence = delta_confidence
        self.X_plot = X_plot
        self.noise_std = noise_std
        self.n_dimensions = X_plot.shape[1]
        self.best_lower_bound_local = -np.infty
        self.safety_threshold = safety_threshold
        self.grid_coords = grid_coords  # tuple of grid coordinates (e.g., (0,0) for first cube in 2D)
        self.lambda_bar = max(self.noise_std, 1)
        
        # Initialize RKHS norm and variance integral lists
        self.RKHS_norm_mean_function_list = []
        self.vi_frac_list = []
        
        # Calculate bounds based on grid coordinates
        if grid_coords == (-1, -1):  # Global domain
            self.lb = torch.tensor([0]*X_plot.shape[1])
            self.ub = torch.tensor([1]*X_plot.shape[1])
            self.x_sample = X_sample
            self.y_sample = Y_sample
            self.discr_domain = X_plot
        else:
            # Calculate bounds for local cube based on grid coordinates
            self.lb = torch.tensor([coord * delta_cube for coord in grid_coords])
            self.ub = torch.tensor([(coord + 1) * delta_cube for coord in grid_coords])
            self.ub[self.ub > 1] = 1  # clipping
            
            # Get samples within this cube
            sample_indices = torch.all(torch.logical_and(X_sample >= self.lb, X_sample <= self.ub), axis=1)
            self.x_sample = X_sample[sample_indices].clone().detach()
            self.y_sample = Y_sample[sample_indices].clone().detach()
            
            if not compute_local_X_plot:
                self.discr_domain = X_plot[torch.all(torch.logical_and(self.X_plot >= self.lb, self.X_plot <= self.ub), axis=1)]
            else:
                self.discr_domain = compute_X_plot_locally(
                    n_dimensions=self.X_plot.shape[1],
                    points_per_axis=int(np.round(self.X_plot.shape[0]**(1/self.X_plot.shape[1]))),
                    lb=self.lb,
                    ub=self.ub
                )

    def compute_model(self, dict_reuse_GPs, gpr):
        if convert_to_hashable(self.x_sample) in dict_reuse_GPs.keys():
            self.model, self.K = dict_reuse_GPs[convert_to_hashable(self.x_sample)]
        else:
            
            self.model = gpr(train_x=self.x_sample, train_y=self.y_sample, noise_std=self.noise_std, lengthscale=0.1)
            self.K = self.model(self.x_sample).covariance_matrix
            dict_reuse_GPs[convert_to_hashable(self.x_sample)] = [self.model, self.K]

    def compute_mean_var(self):  # GP model predictions
        self.model.eval()
        self.f_preds = self.model(self.discr_domain)
        self.mean = self.f_preds.mean
        self.var = self.f_preds.variance

    def compute_confidence_intervals_training(self, dict_local_RKHS_norms={}):
        if self.grid_coords in dict_local_RKHS_norms:
            self.B = dict_local_RKHS_norms[self.grid_coords]
        else:
            self.B = self.compute_RKHS_norm_true()
            dict_local_RKHS_norms[self.grid_coords] = self.B
        self.compute_beta()
        self.lcb = self.mean - self.beta*torch.sqrt(self.var)  # we have to use standard deviation instead of variance
        self.ucb = self.mean + self.beta*torch.sqrt(self.var)
        return dict_local_RKHS_norms

    def compute_confidence_intervals_evaluation(self, RNN_model=None, m_PAC=None, alpha_bar=None, PAC=True, RKHS_norm_guessed=None):  # PAC is a boolean that decides whether we are in the outer loop or inner loop
        if RKHS_norm_guessed is None:
            if len(self.RKHS_norm_mean_function_list) > 0:
                self.B = predict(RNN_model, self.RKHS_norm_mean_function_list, self.vi_frac_list)
            else:
                # If we don't have any RKHS norm data yet, use the true RKHS norm
                self.B = self.compute_RKHS_norm_true()
                
            if PAC:
                list_random_RKHS_norms = []
                N_hat = int(max(torch.round((torch.max(self.ub-self.lb))*500), len(self.y_sample) + 10))
                print(f'Getting PAC bounds now for cube {self.grid_coords}.')
                x_interpol = self.x_sample
                y_interpol = self.y_sample
                for _ in range(m_PAC):
                    X_c = (torch.min(self.discr_domain) - torch.max(self.discr_domain))*torch.rand(N_hat, self.x_sample.shape[1]) + torch.max((self.discr_domain))
                    X_c_tail = X_c[x_interpol.shape[0]:]
                    X_c[:self.x_sample.shape[0]] = x_interpol
                    alpha_tail = -2*alpha_bar*torch.rand(N_hat-len(y_interpol), 1) + alpha_bar
                    y_tail = self.model.kernel(x_interpol, X_c_tail).evaluate() @ alpha_tail
                    y_head = (y_interpol - torch.squeeze(y_tail)).reshape(-1, 1)
                    # matrix inversion with nugget factor for regularization
                    # Following line satisfies interpolation propoert
                    alpha_head = torch.inverse(self.model.kernel(x_interpol, x_interpol).evaluate()+torch.eye(len(y_interpol))*1e-3) @ y_head
                    alpha = torch.cat((alpha_head, alpha_tail))
                    random_RKHS_norm = torch.sqrt(alpha.T @ self.model.kernel(X_c, X_c).evaluate() @ alpha)
                    list_random_RKHS_norms.append(random_RKHS_norm)
                numpy_list = [tensor.item() for tensor in list_random_RKHS_norms]
                numpy_list.sort()
                r_final = 0
                for r in range(m_PAC):
                    summ = 0
                    for i in range(r):
                        summ += comb(m_PAC, i)*gamma_PAC**(i)*(1-gamma_PAC)**(m_PAC-i)
                    if summ > kappa_PAC or self.B > numpy_list[-1-r]:
                        break
                    else:
                        r_final = r
                self.B = max(self.B, numpy_list[-1-r_final])  # Algorithm 3; scenario approach with PAC bounds
        elif RKHS_norm_guessed is not None:
            self.B = RKHS_norm_guessed
        self.compute_beta()
        self.lcb = self.mean - self.beta*torch.sqrt(self.var)  # we have to use standard deviation instead of variance
        self.ucb = self.mean + self.beta*torch.sqrt(self.var)

    def compute_safe_set(self):
        self.S = self.lcb > self.safety_threshold

        # Auxiliary objects of potential maximizers M and potential expanders G
        self.G = self.S.clone()
        self.M = self.S.clone()

    def maximizer_routine(self, best_lower_bound_others):
        self.M[:] = False  # initialize
        self.max_M_var = 0  # initialize
        if not torch.any(self.S):  # no safe points
            return
        self.best_lower_bound_local = max(self.lcb[self.S])
        self.M[self.S] = self.ucb[self.S] >= max(best_lower_bound_others, self.best_lower_bound_local)
        self.M[self.M.clone()] = (self.ucb[self.M] - self.lcb[self.M]) > self.exploration_threshold
        if not torch.any(self.M):
            return
        self.max_M_var = torch.max(self.ucb[self.M] - self.lcb[self.M])
        self.max_M_ucb = torch.max(self.ucb[self.M])

    def expander_routine(self):
        self.G[:] = False  # initialize
        if not torch.any(self.S) or torch.all(self.S):  # no safe points or all of them are safe points -> no possible expanders
            return
        # no need to consider points in M
        if self.compute_all_sets:  # for visualization; introductory example
            s = self.S.clone()
        else:
            s = torch.logical_and(self.S, ~self.M)
            s[s.clone()] = (self.ucb[s] - self.lcb[s]) > self.max_M_var
            s[s.clone()] = (self.ucb[s] - self.lcb[s]) > self.exploration_threshold  # only sufficiently uncertain.
        # still same size as the safe set! We are just over-writing the positive ones
        if not torch.any(s):
            return
        potential_expanders = self.discr_domain[s]
        unsafe_points = self.discr_domain[~self.S]
        kernel_distance = self.compute_kernel_distance(potential_expanders, unsafe_points)
        ucb_expanded = self.ucb[s].unsqueeze(1).expand(-1, kernel_distance.size(1))
        s[s.clone()] = torch.any(ucb_expanded - self.B*kernel_distance > self.safety_threshold, dim=1)
        # or go with for loop; might be more scalable, but slower for smaller dimensions
        # boolean_expander = ~s[s.clone()]  # assume that all are NOT expanders and go in the loop
        # for i in range(len(potential_expanders)):
        #     potential_expander = potential_expanders[i]
        #     for unsafe_point in unsafe_points:
        #         if self.ucb[s][i] - self.compute_kernel_distance(potential_expander, unsafe_point) > self.safety_threshold:
        #             boolean_expander[i] = True
        #             break  # we only need one!  
        # s[s.clone()] = boolean_expander  # update the potential expanders on whether they can potentially expand to an unsafe point            
        self.G = s

    def compute_beta(self):
        # New beta computation based on provided formula
        # β^(c)_tc := 2B^(c)_tc + 300 I^(c)_tc log³(tc/δ)
        
        # Get current time step (number of samples)
        t = self.x_sample.shape[0]
        
        # Get dimension
        d = self.n_dimensions
        
        # Matérn kernel parameter (nu = 1.5 in our case)
        mu = 1.5
        
        # Information gain γt = O(t^(d(d+1)/(2ν+d(d+1))) · log t)
        # For the constant, we use a reasonable scaling factor
        exponent = d * (d + 1) / (2 * mu + d * (d + 1))
        gamma_t = (t ** exponent) * torch.log(torch.tensor(t + 1, dtype=torch.float32))  # +1 to avoid log(0)
        
        # Compute log³(t/δ)
        log_term = torch.log(torch.tensor(t / self.delta_confidence, dtype=torch.float32))
        
        # β^(c)_tc := 2B^(c)_tc + 300 I^(c)_tc log³(tc/δ)
        self.beta = torch.sqrt(2 * self.B +  gamma_t * log_term)
        
        # Print self.B specifically for global domain
        if self.grid_coords == (-1, -1):
            print(f"Global domain RKHS norm (B): {self.B}")
        
        print(f"beta: {self.beta}, t: {t}, d: {d}, gamma_t: {gamma_t}, log_term: {log_term}")

    def compute_RKHS_norm_true(self):
        return self.gt.local_RKHS_norm(lb=self.lb, ub=self.ub, X_plot_local=self.discr_domain) if self.grid_coords != (-1, -1) else self.gt.RKHS_norm

    def compute_kernel_distance(self, x, x_prime):  # let us try whether it works without reshaped!
        '''
        k(x,x)+k(x^\prime,x^\prime)-k(x,x^\prime)-k(x^\prime,x)=2-2k(x,x^\prime)
        This holds for all radial kernels with output variance 1, i.e., k(x,x)\equiv 1.
        Both of which are true for our case.
        We have this setting and we exploit it.
        '''
        if self.model.kernel.__class__.__name__ != 'MaternKernel':  # This work only uses Matérn kernels
            raise Exception("Current implementation only works with radial kernels.")
        matrix_containing_kernel_values = self.model.kernel(x, x_prime).evaluate()  # here we can have problems with the size of the matrix
        return torch.sqrt(2-2*matrix_containing_kernel_values)
    
    def save_data_for_RNN_training(self, dict_mean_RKHS_norms, dict_recip_variances, x_last_iteration):
        if convert_to_hashable(self.grid_coords) not in dict_mean_RKHS_norms.keys():
            alpha = torch.inverse(self.K+self.noise_std**2*torch.eye(self.K.shape[0])) @ self.y_sample
            self.RKHS_norm_mean_function_list = [torch.sqrt(alpha.reshape(1, -1) @ self.K @ alpha.reshape(-1, 1)).flatten()]  # RKHS norm of the mean.
            dict_mean_RKHS_norms[self.grid_coords] = self.RKHS_norm_mean_function_list

            variance_integral = sum(self.var)/2*(self.n_dimensions/self.discr_domain.shape[0])
            self.vi_frac_list = [1/variance_integral]  # reciprocal covariance integral
            dict_recip_variances[self.grid_coords] = self.vi_frac_list
        elif x_last_iteration is None:
            pass
        elif torch.all(torch.logical_and(x_last_iteration >= self.lb, x_last_iteration <= self.ub)):
            alpha = torch.inverse(self.K+self.noise_std**2*torch.eye(self.K.shape[0])) @ self.y_sample
            RKHS_norm_mean_function = torch.sqrt(alpha.reshape(1, -1) @ self.K @ alpha.reshape(-1, 1)).flatten()
            dict_mean_RKHS_norms[self.grid_coords].append(RKHS_norm_mean_function)
            self.RKHS_norm_mean_function_list = dict_mean_RKHS_norms[self.grid_coords]

            variance_integral = sum(self.var)/2*(self.n_dimensions/self.discr_domain.shape[0])
            dict_recip_variances[self.grid_coords].append(1/variance_integral)
            self.vi_frac_list = dict_recip_variances[self.grid_coords]
        else:  # Already in the list and new point is not influencing the sub-domain
            self.vi_frac_list = dict_recip_variances[self.grid_coords]
            self.RKHS_norm_mean_function_list = dict_mean_RKHS_norms[self.grid_coords]
        return dict_mean_RKHS_norms, dict_recip_variances

 

def run(args):
    hyperparameters, num_iterations, X_sample, Y_sample, gt, X_plot, global_approach, RNN_model, compute_local_X_plot = args

    compute_all_sets = False
    noise_std, delta_confidence, alpha_bar, m_PAC, gamma_PAC, kappa_PAC,\
    exploration_threshold, delta_cube, num_local_cubes = hyperparameters

    # Create fixed grid of cubes
    n_dimensions = X_plot.shape[1]
    grid_size = int(1/delta_cube)  # Number of cubes per dimension
    all_cubes = set()
    
    # Generate all possible grid coordinates
    if n_dimensions == 1:
        all_cubes = {(i,) for i in range(grid_size)}
    else:
        all_cubes = {(i, j) for i in range(grid_size) for j in range(grid_size)}
    
    # Add global domain
    all_cubes.add((-1, -1))
    
    # Find which cube contains the initial safe point
    initial_cube = None
    boundary_cubes = set()  # Set to store cubes that contain boundary points
    
    # Check if initial point is on boundary
    initial_point = X_sample[0]
    print(f"\nInitial safe point: {initial_point}")
    
    for coords in all_cubes:
        if coords == (-1, -1):
            continue
        lb = torch.tensor([coord * delta_cube for coord in coords])
        ub = torch.tensor([(coord + 1) * delta_cube for coord in coords])
        
        # Check if point is within cube
        is_within = torch.all(torch.logical_and(initial_point >= lb, initial_point <= ub))
        
        # Check if point is on boundary
        is_on_boundary = torch.any(torch.isclose(initial_point, lb)) or torch.any(torch.isclose(initial_point, ub))
        
        if is_within:
            if is_on_boundary:
                boundary_cubes.add(coords)
                print(f"Initial point is on boundary of cube {coords}")
            else:
                initial_cube = coords
                print(f"Initial point is in cube {coords}")
                break
    
    # If initial point is on boundary, randomly select one of the boundary cubes
    if initial_cube is None and len(boundary_cubes) > 0:
        initial_cube = random.choice(list(boundary_cubes))
        boundary_cubes.remove(initial_cube)
        print(f"Randomly selected cube {initial_cube} from boundary cubes")
    
    if initial_cube is None:
        raise Exception("Initial safe point not found in any cube!")
    
    # Start with only the initial cube active
    active_local_cube = initial_cube  # Only one active local cube
    inactive_cubes = all_cubes - {active_local_cube, (-1, -1)} - boundary_cubes
    deactivated_cubes = set()  # Set to track cubes that were deactivated by algorithm
    potential_cubes = set()  # Set to track cubes that had boundary intersections
    print(f"\nStarting with active cube: {active_local_cube}")
    print(f"Number of inactive cubes: {len(inactive_cubes)}")
    print(f"Number of boundary cubes: {len(boundary_cubes)}")
    print(f"Number of deactivated cubes: {len(deactivated_cubes)}")
    
    x_new_last_iteration = torch.tensor([-torch.inf for _ in range(n_dimensions)])
    best_lower_bound_others = -np.infty
    
    # Initialize dictionaries for storing RKHS norm data
    dict_mean_RKHS_norms = {}
    dict_recip_variances = {}
    
    while X_sample.shape[0] <= num_iterations:
        print(f'\nIteration {X_sample.shape[0]}')
        print(f'Active cube: {active_local_cube}')
        print(f'Current best lower bound: {best_lower_bound_others:.4f}')
        
        try:
            del chosen_cube
        except NameError:
            pass
            
        best_lower_bound_others = -np.infty
        max_uncertainty_interesting = 0
        dict_reuse_GPs = {}
        
        # First compute global cube to get best lower bound
        global_cube = safe_BO(delta_confidence=delta_confidence, delta_cube=delta_cube, noise_std=noise_std,
                            grid_coords=(-1, -1), X_plot=X_plot, X_sample=X_sample,
                            Y_sample=Y_sample, safety_threshold=gt.safety_threshold,
                            exploration_threshold=exploration_threshold, gt=gt,
                            compute_local_X_plot=compute_local_X_plot, compute_all_sets=compute_all_sets)
        
        global_cube.compute_model(dict_reuse_GPs, gpr=GPRegressionModel)
        global_cube.compute_mean_var()
        dict_mean_RKHS_norms, dict_recip_variances = global_cube.save_data_for_RNN_training(dict_mean_RKHS_norms, dict_recip_variances, x_new_last_iteration)
        global_cube.compute_confidence_intervals_evaluation(RNN_model=RNN_model, m_PAC=m_PAC, alpha_bar=alpha_bar, PAC=True)
        global_cube.compute_safe_set()
        global_cube.maximizer_routine(best_lower_bound_others=best_lower_bound_others)
        global_cube.expander_routine()
        
        # Update best lower bound from global cube
        if torch.any(global_cube.S):
            best_lower_bound_others = max(global_cube.lcb[global_cube.S])
            print(f"Updated best lower bound from global cube: {best_lower_bound_others:.4f}")
        
        # If no active cube, try to activate a new one using global domain
        if active_local_cube is None:
            print("\nNo active cube, using global domain to activate new cube")
            print(f"Current inactive cubes: {inactive_cubes}")
            print(f"Current boundary cubes: {boundary_cubes}")
            print(f"Current deactivated cubes: {deactivated_cubes}")
            print(f"Potential cubes (had boundary intersections): {potential_cubes}")
            
            # Initialize set of available cubes
            available_cubes = set()
            
            # Add boundary cubes and potential cubes to available cubes, excluding deactivated cubes
            available_cubes.update(boundary_cubes - deactivated_cubes)
            available_cubes.update(potential_cubes - deactivated_cubes)
            print(f"Available cubes (including boundary and potential): {available_cubes}")
            
            if torch.any(torch.logical_or(global_cube.M, global_cube.G)):
                # Get all suggested points
                print("global_cube.M: ", global_cube.M)
                print("global_cube.G: ", global_cube.G)
                suggested_points = global_cube.discr_domain[torch.logical_or(global_cube.M, global_cube.G)]
                print(f"Suggested points: {suggested_points}")
                suggested_variances = global_cube.var[torch.logical_or(global_cube.M, global_cube.G)]
                
                # Find which cubes contain the suggested points
                suggested_cubes = set()
                for point in suggested_points:
                    for coords in all_cubes - {(-1, -1)} - deactivated_cubes:  # Exclude deactivated cubes here
                        lb = torch.tensor([coord * delta_cube for coord in coords])
                        ub = torch.tensor([(coord + 1) * delta_cube for coord in coords])
                        ub[ub > 1] = 1  # clipping
                        
                        # Check if point is in cube (including boundaries)
                        # A point is in a cube if it's either:
                        # 1. Strictly inside the cube, or
                        # 2. On the boundary of the cube
                        is_inside = torch.all(torch.logical_and(point > lb, point < ub))
                        is_on_boundary = torch.any(torch.isclose(point, lb)) or torch.any(torch.isclose(point, ub))
                        is_in_cube = is_inside or is_on_boundary
                        
                        if is_in_cube:
                            suggested_cubes.add(coords)
                            print(f"Point {point} is in cube {coords}")
                            if is_on_boundary:
                                print(f"  (Point is on boundary of cube {coords})")
                
                # Remove deactivated cubes from suggested cubes
                suggested_cubes = suggested_cubes - deactivated_cubes
                print(f"Cubes containing suggested points (excluding deactivated): {suggested_cubes}")
                
                # Add suggested cubes to available cubes
                available_cubes.update(suggested_cubes)
                print(f"Total available cubes: {available_cubes}")
                
                # Only remove current cube from available cubes
                available_cubes = available_cubes - {active_local_cube}
                print(f"Cubes available for random selection: {available_cubes}")
                
                if len(available_cubes) == 0:
                    print("No new cubes to explore, terminating")
                    break
                
                # Randomly select a cube from available cubes
                selected_cube = random.choice(list(available_cubes))
                print(f"Randomly selected cube: {selected_cube}")
                
                # Find a point in the selected cube
                x_new = None
                if selected_cube in boundary_cubes:
                    # For boundary cubes, use the boundary point
                    lb = torch.tensor([coord * delta_cube for coord in selected_cube])
                    ub = torch.tensor([(coord + 1) * delta_cube for coord in selected_cube])
                    x_new = lb  # or ub, depending on which boundary we want to use
                elif selected_cube in suggested_cubes:
                    # For suggested points, find a point in the selected cube
                    for point in suggested_points:
                        lb = torch.tensor([coord * delta_cube for coord in selected_cube])
                        ub = torch.tensor([(coord + 1) * delta_cube for coord in selected_cube])
                        if torch.all(torch.logical_and(point >= lb, point <= ub)):
                            x_new = point
                            break
                elif selected_cube in potential_cubes:
                    # For potential cubes, use the boundary point
                    lb = torch.tensor([coord * delta_cube for coord in selected_cube])
                    ub = torch.tensor([(coord + 1) * delta_cube for coord in selected_cube])
                    x_new = lb  # or ub, depending on which boundary we want to use
                
                if x_new is None:
                    print("No suitable point found in selected cube, terminating")
                    break
                    
                active_local_cube = selected_cube
                if selected_cube in inactive_cubes:
                    inactive_cubes.remove(selected_cube)
                if selected_cube in boundary_cubes:
                    boundary_cubes.remove(selected_cube)
                if selected_cube in potential_cubes:
                    potential_cubes.remove(selected_cube)
                print(f"Activating cube {selected_cube} with point {x_new}")
            elif len(available_cubes) > 0:
                # FORCE EXPLORATION: Even if global domain doesn't suggest promising points,
                # explore available boundary/potential cubes to prevent premature termination
                print("Global domain suggests no promising points, but forcing exploration of available cubes")
                
                # Only remove current cube from available cubes
                available_cubes = available_cubes - {active_local_cube}
                print(f"Forcing exploration of cubes: {available_cubes}")
                
                if len(available_cubes) == 0:
                    print("No new cubes to explore, terminating")
                    break
                
                # Randomly select a cube from available cubes
                selected_cube = random.choice(list(available_cubes))
                print(f"Force-selected cube: {selected_cube}")
                
                # Find a point in the selected cube
                x_new = None
                if selected_cube in boundary_cubes:
                    # For boundary cubes, use the boundary point
                    lb = torch.tensor([coord * delta_cube for coord in selected_cube])
                    ub = torch.tensor([(coord + 1) * delta_cube for coord in selected_cube])
                    x_new = lb  # or ub, depending on which boundary we want to use
                elif selected_cube in potential_cubes:
                    # For potential cubes, use the boundary point
                    lb = torch.tensor([coord * delta_cube for coord in selected_cube])
                    ub = torch.tensor([(coord + 1) * delta_cube for coord in selected_cube])
                    x_new = lb  # or ub, depending on which boundary we want to use
                
                if x_new is None:
                    print("No suitable point found in force-selected cube, terminating")
                    break
                    
                active_local_cube = selected_cube
                if selected_cube in inactive_cubes:
                    inactive_cubes.remove(selected_cube)
                if selected_cube in boundary_cubes:
                    boundary_cubes.remove(selected_cube)
                if selected_cube in potential_cubes:
                    potential_cubes.remove(selected_cube)
                print(f"Force-activating cube {selected_cube} with point {x_new}")
            else:
                print('Algorithm terminated! No more promising points to sample.')
                break
        
        # Try active local cube
        local_cube = safe_BO(delta_confidence=delta_confidence, delta_cube=delta_cube, noise_std=noise_std,
                          grid_coords=active_local_cube, X_plot=X_plot, X_sample=X_sample,
                          Y_sample=Y_sample, safety_threshold=gt.safety_threshold,
                          exploration_threshold=exploration_threshold, gt=gt,
                          compute_local_X_plot=compute_local_X_plot, compute_all_sets=compute_all_sets)
        
        local_cube.compute_model(dict_reuse_GPs, gpr=GPRegressionModel)
        local_cube.compute_mean_var()
        dict_mean_RKHS_norms, dict_recip_variances = local_cube.save_data_for_RNN_training(dict_mean_RKHS_norms, dict_recip_variances, x_new_last_iteration)
        local_cube.compute_confidence_intervals_evaluation(RNN_model=RNN_model, m_PAC=m_PAC, alpha_bar=alpha_bar, PAC=True)
        local_cube.compute_safe_set()
        local_cube.maximizer_routine(best_lower_bound_others=best_lower_bound_others)
        local_cube.expander_routine()
        
        # Update best lower bound from local cube if it's better
        if torch.any(local_cube.S):
            local_best_lower_bound = max(local_cube.lcb[local_cube.S])
            if local_best_lower_bound > best_lower_bound_others:
                best_lower_bound_others = local_best_lower_bound
                print(f"Updated best lower bound from local cube: {best_lower_bound_others:.4f}")
        
        if torch.any(torch.logical_or(local_cube.M, local_cube.G)):
            max_uncertainty_interesting_local = max((local_cube.ucb - local_cube.lcb)[torch.logical_or(local_cube.M, local_cube.G)])
            x_new_current = local_cube.discr_domain[torch.logical_or(local_cube.M, local_cube.G)][torch.argmax(local_cube.var[torch.logical_or(local_cube.M, local_cube.G)])]
            
            # Check if new point is on boundary
            is_boundary_point = False
            for other_coords in all_cubes - {active_local_cube} - {(-1, -1)} - deactivated_cubes:  # Exclude deactivated cubes here
                other_lb = torch.tensor([coord * delta_cube for coord in other_coords])
                other_ub = torch.tensor([(coord + 1) * delta_cube for coord in other_coords])
                other_ub[other_ub > 1] = 1  # clipping
                
                # Check if point is actually within the other cube's bounds (including boundaries)
                is_within_cube = torch.all(torch.logical_and(x_new_current >= other_lb, x_new_current <= other_ub))
                
                if is_within_cube:
                    # Check if it's actually on the boundary between the two cubes
                    current_lb = torch.tensor([coord * delta_cube for coord in active_local_cube])
                    current_ub = torch.tensor([(coord + 1) * delta_cube for coord in active_local_cube])
                    current_ub[current_ub > 1] = 1  # clipping
                    
                    # A point is on the boundary if it's on the edge of the current cube
                    # AND within the other cube
                    is_on_current_boundary = (torch.any(torch.isclose(x_new_current, current_lb)) or 
                                            torch.any(torch.isclose(x_new_current, current_ub)))
                    
                    if is_on_current_boundary:
                        # Add to boundary cubes if the other cube is inactive and not deactivated
                        if other_coords in inactive_cubes and other_coords not in deactivated_cubes:
                            boundary_cubes.add(other_coords)
                            print(f"Found boundary point with inactive cube {other_coords}")
                            is_boundary_point = True
                        # Add to potential cubes if the other cube is not inactive and not deactivated
                        elif other_coords not in deactivated_cubes:
                            potential_cubes.add(other_coords)
                            print(f"Found boundary point with potential cube {other_coords}")
                            is_boundary_point = True
            
            if not torch.any(torch.all(X_sample == x_new_current, axis=1)):
                chosen_cube = local_cube
                x_new = x_new_current
                print(f"Sampling from active cube {active_local_cube}")
        
        # If no promising points in active local cube, mark it as done
        if 'chosen_cube' not in locals():
            print(f"No promising points in active cube {active_local_cube}, marking as done")
            deactivated_cubes.add(active_local_cube)  # Add to deactivated cubes instead of inactive
            active_local_cube = None
            continue
        
        # Sample the new point
        y_new = gt.conduct_experiment(x=x_new, noise_std=noise_std)
        if y_new < gt.safety_threshold:
            raise Exception('Sampled unsafe point!')
            
        X_sample = torch.cat((X_sample, x_new.unsqueeze(0)), dim=0)
        Y_sample = torch.cat((Y_sample, y_new), dim=0)
        print(f"Sampled point: {x_new} with value: {y_new}")
        
        # Update global domain RKHS norm data after local sampling
        global_cube = safe_BO(delta_confidence=delta_confidence, delta_cube=delta_cube, noise_std=noise_std,
                            grid_coords=(-1, -1), X_plot=X_plot, X_sample=X_sample,
                            Y_sample=Y_sample, safety_threshold=gt.safety_threshold,
                            exploration_threshold=exploration_threshold, gt=gt,
                            compute_local_X_plot=compute_local_X_plot, compute_all_sets=compute_all_sets)
        
        global_cube.compute_model(dict_reuse_GPs, gpr=GPRegressionModel)
        global_cube.compute_mean_var()
        dict_mean_RKHS_norms, dict_recip_variances = global_cube.save_data_for_RNN_training(dict_mean_RKHS_norms, dict_recip_variances, x_new)
        
        # Check if current local cube is done using the existing local_cube instance
        if 'chosen_cube' in locals() and chosen_cube.grid_coords != (-1, -1):
            # Update local cube with new sample
            local_cube = safe_BO(delta_confidence=delta_confidence, delta_cube=delta_cube, noise_std=noise_std,
                              grid_coords=active_local_cube, X_plot=X_plot, X_sample=X_sample,
                              Y_sample=Y_sample, safety_threshold=gt.safety_threshold,
                              exploration_threshold=exploration_threshold, gt=gt,
                              compute_local_X_plot=compute_local_X_plot, compute_all_sets=compute_all_sets)
            
            local_cube.compute_model(dict_reuse_GPs, gpr=GPRegressionModel)
            local_cube.compute_mean_var()
            dict_mean_RKHS_norms, dict_recip_variances = local_cube.save_data_for_RNN_training(dict_mean_RKHS_norms, dict_recip_variances, x_new)
            local_cube.compute_confidence_intervals_evaluation(RNN_model=RNN_model, m_PAC=m_PAC, alpha_bar=alpha_bar, PAC=True)
            local_cube.compute_safe_set()
            local_cube.maximizer_routine(best_lower_bound_others=best_lower_bound_others)
            local_cube.expander_routine()
            
            if not torch.any(torch.logical_or(local_cube.M, local_cube.G)):
                # Current cube is done, move to inactive
                print(f"Cube {active_local_cube} is done, moving to inactive")
                deactivated_cubes.add(active_local_cube)
                active_local_cube = None
        
        x_new_last_iteration = copy.deepcopy(x_new)
        del x_new
        
    return X_sample, Y_sample, {active_local_cube} if active_local_cube is not None else set(), gt.safety_threshold


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Hyperparameters
    noise_std = 0.01  # standard deviation of noise, and in GPs
    delta_confidence = 0.01  # yields 99% confidence for safety proof.
    num_safe_points = 1  # singleton safe set
    num_iterations = 1500  # iterations. We used 50 iterations to get training data and for the Gym experiments, 30 otherwise
    exploration_threshold = 0.1  # exploration threshold, see Sui et al. 2015
    n_dimensions = 1
    points_per_axis = 101  # 30 for 4D, 1000 for 1D, 500 for 2D, 100 for 3D, 8 for 6D. Depends on computational resources, also a "hyperparameter"

    # Initialize our algorithm
    X_plot = compute_X_plot(n_dimensions, points_per_axis)
    delta_cube = 0.25 # hyperparameter
    num_local_cubes = 1

    compute_all_sets= False
    # For "evaluating"
    kappa_PAC = 0.01  # confidence PAC bounds
    gamma_PAC = 0.1  # probability PAC bounds
    m_PAC = 1000  # number of random RKHS function created for PAC bounds
    alpha_bar = 1
    RKHS_norm = 5 # np.random.uniform(0.5, 30)
    
    # Set random seed for reproducibility
    seed = 21  # You can change this to any integer you want
    gt = ground_truth(num_center_points=1000, X_plot=X_plot, RKHS_norm=RKHS_norm, seed=seed)
    X_sample_init, Y_sample_init = initial_safe_samples(gt=gt, num_safe_points=num_safe_points)
    X_sample = X_sample_init.clone()
    Y_sample = Y_sample_init.clone()
    if n_dimensions == 1:
            plot_1D(X_sample=X_sample, Y_sample=Y_sample, X_plot=X_plot, fX=gt.fX, title='ours', safety_threshold=gt.safety_threshold, save=False, delta_cube=delta_cube)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        compute_local_X_plot = True
        hyperparameters = [noise_std, delta_confidence, alpha_bar, m_PAC, gamma_PAC, kappa_PAC,
                            exploration_threshold, delta_cube, num_local_cubes]
        model_path = "rnn_model.pt"
        hidden_size = 20
        num_layers = 2
        num_classes = 1
        RNN_model = load_model(model_path, hidden_size, num_layers, num_classes)
        global_approach = False

        # test the args of run function

        X_sample_our, Y_sample_our, active_cubes, safety_threshold = run(args=[hyperparameters, num_iterations, X_sample, Y_sample, gt, X_plot, global_approach, RNN_model, compute_local_X_plot])
        if n_dimensions == 1:
            plot_1D(X_sample=X_sample_our, Y_sample=Y_sample_our, X_plot=X_plot, fX=gt.fX, title='ours', safety_threshold=safety_threshold, save=False, delta_cube=delta_cube)
        elif n_dimensions == 2:
            plot_2D_contour(X_plot, gt.fX, X_sample_our, Y_sample=Y_sample_our, safety_threshold=gt.safety_threshold, title='ours', levels=10, save=False, delta_cube=delta_cube)