import torch
import torch.nn as nn

import numpy as np
import gpytorch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import comb
from scipy.special import gamma



def initial_safe_samples(gt, num_safe_points):  # for toy examples and introductory example
    fX = gt.fX
    num_safe_points = num_safe_points
    # sampling_logic = fX > gt.safety_threshold  # alternative option
    sampling_logic = torch.logical_and(fX > np.quantile(fX, 0.4), fX < np.quantile(fX, 0.50))
    random_indices_sample = torch.randint(high=gt.X_plot[sampling_logic].shape[0], size=(num_safe_points,))
    X_sample = gt.X_plot[sampling_logic][random_indices_sample]
    Y_sample = fX[sampling_logic][random_indices_sample] + torch.tensor(np.random.normal(loc=0, scale=noise_std, size=X_sample.shape[0]), dtype=torch.float32)
    return X_sample, Y_sample


class GroundTruth():
    def __init__(self, num_center_points, n_dimensions,points_per_axis, RKHS_norm):
        X_plot = self.compute_X_plot(n_dimensions, points_per_axis)
        self.points_per_axis = points_per_axis
        def fun(kernel, alpha):
            return lambda X: kernel(X.reshape(-1, self.X_center.shape[1]), self.X_center).detach().numpy() @ alpha
        # For ground truth
        self.X_plot = X_plot
        self.RKHS_norm = RKHS_norm
        random_indices_center = torch.randint(high=self.X_plot.shape[0], size=(num_center_points,))
        self.X_center = self.X_plot[random_indices_center]
        alpha = np.random.uniform(-1, 1, size=self.X_center.shape[0])
        self.kernel = gpytorch.kernels.MaternKernel(nu=3/2)
        self.kernel.lengthscale = 0.1  # used in all runs except for Furuta hardware. There, this will get over-written.
        RKHS_norm_squared = alpha.T @ self.kernel(self.X_center, self.X_center).detach().numpy() @ alpha
        alpha /= np.sqrt(RKHS_norm_squared)/RKHS_norm  # scale to RKHS norm
        self.f = fun(self.kernel, alpha)
        self.fX = torch.tensor(self.f(self.X_plot), dtype=torch.float32)
        self.safety_threshold = np.quantile(self.fX, 0.3) 
    
    def compute_X_plot(self,n_dimensions, points_per_axis):
        X_plot_per_domain = torch.linspace(0, 1, points_per_axis)
        X_plot_per_domain_nd = [X_plot_per_domain] * n_dimensions
        X_plot = torch.cartesian_prod(*X_plot_per_domain_nd).reshape(-1, n_dimensions)
        return X_plot
    
    def plot(self, X_sample=None, Y_sample=None):
        
        if self.X_plot.shape[1] == 1:
            plt.figure()
            plt.plot(self.X_plot, self.fX, color='blue')
            plt.scatter(self.local_regions[:, 0], self.fX[self.local_regions[:, 0].long()], color='black')
            plt.xlabel('$a$')
            plt.ylabel('$y$')
        elif self.X_plot.shape[1] == 2:
            division_points = int(np.sqrt(len(self.fX)))
            x1 = self.X_plot[:, 0].reshape(division_points, division_points)
            x2 = self.X_plot[:, 1].reshape(division_points, division_points)

            plt.figure()
            contour = plt.contour(x1, x2, self.fX.reshape(division_points, division_points), levels=10, cmap='seismic')
            plt.scatter(self.local_regions[:, 0], self.local_regions[:, 1], color='black', s=10)
            plt.colorbar(contour)
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.title('Ground truth')
        else:
            print('Plotting not implemented for this dimension.')

        if X_sample is not None and Y_sample is not None:
            if self.X_plot.shape[1] == 2:
                plt.scatter(X_sample[1:, 0], X_sample[1:, 1], color='black', s=50)
                plt.scatter(X_sample[0, 0], X_sample[0, 1], color='magenta', s=50)
                for i in range(len(Y_sample)):
                    if Y_sample[i] < self.safety_threshold:
                        plt.scatter(X_sample[i, 0], X_sample[i, 1], marker='s', color='red', s=150)
            elif self.X_plot.shape[1] == 1:
                plt.scatter(X_sample[1:], Y_sample[1:], color='black')
                plt.plot(X_sample[0], Y_sample[0], 'd', color='magenta', markersize=10)
                plt.plot(self.X_plot, [self.safety_threshold]*len(self.X_plot), '-r')
                plt.title('Ground truth')
                plt.xlabel('$a$')
                plt.ylabel('$y$')
        plt.show()
    
    def compute_local(self, delta_cube):
        intervals_list = []
        for i in range(self.X_plot.shape[1]):
            intervals = torch.linspace(0, 1, int(1/delta_cube)+1)            
            intervals_list.append(intervals)
        print('Intervals:', intervals_list)
        self.local_regions = torch.cartesian_prod(*intervals_list).reshape(-1, self.X_plot.shape[1])
        print('Local regions:', self.local_regions)

    def detect_region_lower_bound(self, x_tensor, delta_cube):
        # round tensor items to the downward nearest interval
        rounded_X_sample = torch.floor(x_tensor / delta_cube) * delta_cube
        return rounded_X_sample

        

if __name__ == '__main__':
    n_dimensions = 1
    points_per_axis = 1000  # 30 for 4D, 1000 for 1D, 500 for 2D, 100 for 3D, 8 for 6D. Depends on computational resources, also a "hyperparameter"
    RKHS_norm = 5  # np.random.uniform(0.5, 30)
    kappa_PAC = 0.01  # confidence PAC bounds
    gamma_PAC = 0.1  # probability PAC bounds
    m_PAC = 1000  
    alpha_bar = 1

    
    gt = GroundTruth(num_center_points=1000,
                    n_dimensions=n_dimensions,
                    points_per_axis=points_per_axis, 
                    RKHS_norm=RKHS_norm)
    delta_cube = 0.1
    gt.compute_local(delta_cube)
    gt.plot()

    noise_std = 0.1
    num_safe_points = 1
    X_sample, Y_sample = initial_safe_samples(gt, num_safe_points)


    

    num_of_iterations = 0
    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        print(gt.X_plot)
    

        while num_of_iterations <=5:
            print('Iteration:', num_of_iterations)
            gt.plot(X_sample, Y_sample)
            print('X_sample:', X_sample)
            print('Y_sample:', Y_sample)
            lb = gt.detect_region_lower_bound(X_sample[num_of_iterations], delta_cube)
            ub = lb + delta_cube
            print('Local indices:', lb)

        




            num_of_iterations += 1
