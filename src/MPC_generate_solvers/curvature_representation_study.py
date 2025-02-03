import numpy as np
import matplotlib.pyplot as plt
from path_track_definitions import generate_path_data



# racetrack generation
track_choice = 'racetrack_vicon_2' 

s_vals_global_path,\
x_vals_global_path,\
y_vals_global_path,\
s_4_local_path,\
x_4_local_path,\
y_4_local_path,\
dx_ds, dy_ds, d2x_ds2, d2y_ds2,\
k_vals_global_path,\
k_4_local_path,\
heading_angle_4_local_path = generate_path_data(track_choice)



# plot the path 4 local path
figure, ax = plt.subplots(3,1)
ax[0].plot(x_4_local_path, y_4_local_path,label='path')
ax[0].set_xlabel('x [m]')
ax[0].set_ylabel('y [m]')
ax[0].axis('equal')
# now plot the heading 
ax[1].plot(s_4_local_path, heading_angle_4_local_path,label='heading')
ax[1].set_xlabel('s [m]')
ax[1].set_ylabel('heading [rad]')
# now plot the curvature
ax[2].plot(s_4_local_path, k_4_local_path,label='k')
ax[2].set_xlabel('s [m]')
ax[2].set_ylabel('k')





# define path length and k_vec
s_init = 2.75 #6.5
path_length = 4.5 # 3 m/s for 1.5s time horizon


# find the index of the point closest to the initial point
s_init_index = np.argmin(np.abs(s_4_local_path - s_init))
# find the index of the point closest to the final point
s_final_index = np.argmin(np.abs(s_4_local_path - (s_init + path_length)))
len_s = s_final_index - s_init_index




# define k_vec to be 0 until 1.5, then 1.33, then 0 again
s_vec = s_4_local_path[s_init_index:s_final_index] - s_4_local_path[s_init_index] # set it to 0
k_vec = k_4_local_path[s_init_index:s_final_index]


# integrate the k_vec to get the angle and then again to get the position
angle = np.zeros(len_s)
x_path = np.zeros(len_s)
y_path = np.zeros(len_s)

# integrate using the real k values
for i in range(1,len_s):
    angle[i] = angle[i-1] + k_vec[i-1]*(s_vec[i]-s_vec[i-1])
    x_path[i] = x_path[i-1] + np.cos(angle[i-1])*(s_vec[i]-s_vec[i-1])
    y_path[i] = y_path[i-1] + np.sin(angle[i-1])*(s_vec[i]-s_vec[i-1])



# plot the k_vec
figure, ax_k = plt.subplots()
ax_k.plot(s_vec, k_vec, label='k',color='gray',alpha=0.3)
ax_k.set_xlabel('s [m]')
ax_k.set_ylabel('k_vec')




# plot the path
figure, ax_p = plt.subplots()
ax_p.plot(x_path, y_path,label='path')
ax_p.set_xlabel('x [m]')
ax_p.set_ylabel('y [m]')
ax_p.axis('equal')

# show different methods of representing the k_vec and overlay the k_vec plot
# the most thorough thing to do would be to compare the solver perfomance for different radiuses of k_vec with different initial conditions


# plotting the k_vec approximation for different methods
# kernelized linear regression
n_points_kernelized = 40
# define the labels
labels_k = np.interp(np.linspace(0,path_length,n_points_kernelized), s_vec, k_vec)
s_X_vec_normalized = np.expand_dims(np.linspace(0,1,n_points_kernelized),1)
s_vec_normalized = s_vec/path_length

# add datapoints to the plot
ax_k.scatter(s_X_vec_normalized * path_length, labels_k, label='data',color='gray',marker='.')

try:
    from .path_track_definitions import K_RBF_kernel, K_matern2_kernel
except:
    from path_track_definitions import K_RBF_kernel, K_matern2_kernel



# define fixed path quantities
path_lengthscale = 1.3/n_points_kernelized
lambda_val = 0.0001**2

K_xx = K_matern2_kernel(s_X_vec_normalized, s_X_vec_normalized, path_lengthscale,n_points_kernelized,n_points_kernelized)
Kxx_inv = np.linalg.inv(K_xx + lambda_val * np.eye(len(s_X_vec_normalized))) 
k_vec_matern = np.zeros(len_s)
for i in range(len_s):
    s_star = np.array([s_vec_normalized[i]])
    K_x_star = K_matern2_kernel(s_star, s_X_vec_normalized, path_lengthscale,1,n_points_kernelized)      
    left_side = K_x_star @ Kxx_inv
    k_vec_matern[i] = left_side @ labels_k
# add this to the plot
ax_k.plot(s_vec, k_vec_matern, label='matern2', zorder=20,alpha=0.5,color='orange')

# repeat the same for the RBF kernel
K_xx = K_RBF_kernel(s_X_vec_normalized, s_X_vec_normalized, path_lengthscale,n_points_kernelized,n_points_kernelized)
Kxx_inv = np.linalg.inv(K_xx + lambda_val * np.eye(len(s_X_vec_normalized)))
k_vec_RBF = np.zeros(len_s)
for i in range(len_s):
    s_star = np.array([s_vec_normalized[i]])
    K_x_star = K_RBF_kernel(s_star, s_X_vec_normalized, path_lengthscale,1,n_points_kernelized)      
    left_side = K_x_star @ Kxx_inv
    k_vec_RBF[i] = left_side @ labels_k
# add this to the plot
ax_k.plot(s_vec, k_vec_RBF, label='RBF',alpha=0.5)

# # --- spline interpolation ---
from scipy.interpolate import CubicSpline
cs = CubicSpline(s_X_vec_normalized[:,0], labels_k)
k_vec_spline = cs(s_vec_normalized)
# add this to the plot
ax_k.plot(s_vec, k_vec_spline, label='spline',alpha=0.5,color='red')

# using chebyshev polynomials with 20 bases
from numpy.polynomial.chebyshev import Chebyshev
cheb = Chebyshev.fit(s_X_vec_normalized[:,0], labels_k, 20)
k_vec_cheb = cheb(s_vec_normalized)
# add this to the plot
ax_k.plot(s_vec, k_vec_cheb, label='chebyshev',alpha=0.5,color='green')


ax_k.legend()
plt.show()




# --- use an SVGP model  with constant mean---
import torch
import gpytorch

# define troch version of the data
x_data =  torch.unsqueeze(torch.tensor(s_vec_normalized),1).float().cuda()
y_data = torch.tensor(k_vec).float().cuda()



from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        #covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) 
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=5/2))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

inducing_points = torch.unsqueeze(torch.linspace(0, 1, n_points_kernelized),1).cuda()
model = GPModel(inducing_points=inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

# set initial values for the likelihood noise and the lenfthsacle
# print out model parameters
for name, param in model.named_parameters():
    print(f'Parameter name: {name}, value = {param}')


likelihood.noise_covar.register_constraint(
"raw_noise", gpytorch.constraints.Interval(0, 1.0)
)
likelihood.noise = torch.tensor([0]).cuda()
model.covar_module.base_kernel.lengthscale = torch.tensor([path_lengthscale]).cuda()
#model.covar_module.outputscale = torch.tensor([1.0]).cuda()
#model.variational_strategy._variational_distribution.variational_mean.data = torch.tensor(labels_k).float().cuda()
# print out trainable parameters for the model

# train the SVGP
train_iter = 1
learn_rate = 0.01



model.train()
likelihood.train()


trainable_params_model = {'params': model.parameters()}
#trainable_params_model = [{'params': [p for n, p in model.named_parameters() if n != "covar_module.base_kernel.raw_lengthscale"]}]

optimizer = torch.optim.Adam([trainable_params_model], lr=learn_rate) # ,{'params': likelihood.parameters()}


# Our loss object. We're using the VariationalELBO
#mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_data.size(0))
mse_criterion = torch.nn.MSELoss()


print('likelihood noise', likelihood.noise.item())
print('lengthscale', model.covar_module.base_kernel.lengthscale.item())
loss_vec = np.zeros(train_iter)
from tqdm import tqdm
for i in tqdm(range(train_iter)):
    optimizer.zero_grad()
    output = model(x_data)
    #loss = -mll(output, y_data)
    loss = mse_criterion(output.mean, y_data)
    loss.backward()
    optimizer.step()
    #collect loss for plotting
    loss_vec[i] = loss.item()
print('likelihood noise', likelihood.noise.item())
print('lengthscale', model.covar_module.base_kernel.lengthscale.item())
#plot loss
plt.figure()
plt.plot(loss_vec)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('SVGP loss')

# evalaute the model in the inducing points locations
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    y_inducing = model(model.variational_strategy.inducing_points)




# add to the plot
ax_k.plot(s_vec, output.mean.detach().cpu().numpy(), label='SVGP',alpha=0.5,color='blue')
ax_k.scatter(model.variational_strategy.inducing_points.detach().cpu().numpy()*path_length,
             y_inducing.mean.detach().cpu().numpy(),color='blue',marker='.',label='inducing points')






ax_k.legend()
#ax_p.legend()
plt.show()