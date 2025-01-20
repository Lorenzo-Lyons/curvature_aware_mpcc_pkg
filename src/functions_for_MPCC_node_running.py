import numpy as np
import sys
import time
sys.path.insert(0, '/home/lorenzo/OneDrive/PhD/ubuntu_stuff/Forces_pro_extracted/forces_pro_client')
import forcespro.nlp
import casadi
import math
from scipy.optimize import minimize, Bounds
import warnings #turn off low rank waring from chebfit
warnings.simplefilter('ignore', np.RankWarning)





def find_s_of_closest_point_on_global_path(x_y_state, s_vals_global_path, x_vals_global_path, y_vals_global_path, previous_index, estimated_ds):
    min_ds = np.min(np.diff(s_vals_global_path))
    estimated_index_jumps = math.ceil(estimated_ds / min_ds)
    minimum_index_jumps = math.ceil(0.1 / min_ds)

    # in case the vehicle is still, ensure a minimum search space to account for localization error
    if estimated_index_jumps < minimum_index_jumps:
        estimated_index_jumps = minimum_index_jumps

    Delta_indexes = estimated_index_jumps * 3

    start_i = previous_index - Delta_indexes
    finish_i = previous_index + Delta_indexes

    #check if start_i is negative and finish_i is positive
    if start_i < 0:
        s_search_vector = np.concatenate((s_vals_global_path[start_i:], s_vals_global_path[: finish_i]), axis=0)
        x_search_vector = np.concatenate((x_vals_global_path[start_i:], x_vals_global_path[: finish_i]), axis=0)
        y_search_vector = np.concatenate((y_vals_global_path[start_i:], y_vals_global_path[: finish_i]), axis=0)

    elif finish_i > s_vals_global_path.size:
        s_search_vector = np.concatenate((s_vals_global_path[start_i:], s_vals_global_path[: finish_i - s_vals_global_path.size]), axis=0)
        x_search_vector = np.concatenate((x_vals_global_path[start_i:], x_vals_global_path[: finish_i - s_vals_global_path.size]), axis=0)
        y_search_vector = np.concatenate((y_vals_global_path[start_i:], y_vals_global_path[: finish_i - s_vals_global_path.size]), axis=0)
    else:
        s_search_vector = s_vals_global_path[start_i: finish_i]
        x_search_vector = x_vals_global_path[start_i: finish_i]
        y_search_vector = y_vals_global_path[start_i: finish_i]

    #remove the last value to avoid ambiguity since first and last value may be the same
    distances = np.zeros(s_search_vector.size)
    for ii in range(0, s_search_vector.size):
        distances[ii] = math.dist([x_search_vector[ii], y_search_vector[ii]], x_y_state[0:2])
    # maybe just get the index from the min operator

    local_index = np.argmin(distances)
    # check if the found minimum is on the boarder (indicating that the real minimum is outside of the search vector)

    #this offers some protection against failing the local search but it doesn't fix all of the possible problems
    #for example if pth loops back (like a bean shape)
    # then you can still get an error (If you have lane boundary information then you colud put a check on the actual value of the min)
    if local_index == 0 or local_index == s_search_vector.size-1:
        # print('search vector was not long enough, doing search on full path')
        distances_2 = np.zeros(s_vals_global_path.size)
        for ii in range(0, s_vals_global_path.size):
            distances_2[ii] = math.dist([x_vals_global_path[ii], y_vals_global_path[ii]], x_y_state[0:2])
        index = np.argmin(distances_2)
    else:
        index = np.where(s_vals_global_path == s_search_vector[local_index])
        # this seemeingly reduntant steps are to extract an int from the "where" operand
        index = index[0]
        index = index[0]


    s = float(s_vals_global_path[index])
    return s, index


def straight(xlims, ylims, n_checkpoints):
    Checkpoints_x = np.linspace(xlims[0], xlims[1], n_checkpoints)
    Checkpoints_y = np.linspace(ylims[0], ylims[1], n_checkpoints)
    return Checkpoints_x, Checkpoints_y

def curve(centre, R,theta_extremes, n_checkpoints):
    theta_init = np.pi * theta_extremes[0]
    theta_end = np.pi * theta_extremes[1]
    theta_vec = np.linspace(theta_init, theta_end, n_checkpoints)
    Checkpoints_x = centre[0] + R * np.cos(theta_vec)
    Checkpoints_y = centre[1] + R * np.sin(theta_vec)
    return Checkpoints_x, Checkpoints_y




def load_GP_parameters_from_file(abs_path_parameters_folder):
    print('loading GP_parameters from: ')
    print(abs_path_parameters_folder)

    # X data
    x_data_vec_vx = np.load(abs_path_parameters_folder + "/forces_params_X_data_vx.npy")
    x_data_vec_vy = np.load(abs_path_parameters_folder + "/forces_params_X_data_vy.npy")
    x_data_vec_w = np.load(abs_path_parameters_folder + "/forces_params_X_data_w.npy")
    # covar inducing points for Orthogonally decoupled SVGP
    try:
        x_data_vec_vx_cov = np.load(abs_path_parameters_folder + "/forces_params_X_data_vx_cov.npy")
        x_data_vec_vy_cov = np.load(abs_path_parameters_folder + "/forces_params_X_data_vy_cov.npy")
        x_data_vec_w_cov = np.load(abs_path_parameters_folder + "/forces_params_X_data_w_cov.npy")
    except:
        #return empty vectors if there are no covariance inducing points (thus you are loading a standard SVGP)
        x_data_vec_vx_cov = []
        x_data_vec_vy_cov = []
        x_data_vec_w_cov = []


    # vx
    file_forces_params_Delta_vx_model_params = abs_path_parameters_folder + "/forces_params_Delta_vx_model_params.npy"
    file_forces_params_Delta_vx_right_vec = abs_path_parameters_folder + "/file_forces_params_Delta_vx_right_vec.npy"
    GP_params_Delta_vx = np.load(file_forces_params_Delta_vx_model_params)
    outputscale_Delta_vx = GP_params_Delta_vx[0]
    lengthscales_Delta_vx = GP_params_Delta_vx[1:]
    right_vec_block_Delta_vx = np.load(file_forces_params_Delta_vx_right_vec)
    central_mat_Delta_vx_vector = np.load(abs_path_parameters_folder + "/file_forces_params_central_matrix_vx_vec.npy")

    # vy
    file_forces_params_Delta_vy_model_params = abs_path_parameters_folder + "/forces_params_Delta_vy_model_params.npy"
    file_forces_params_Delta_vy_right_vec = abs_path_parameters_folder + "/file_forces_params_Delta_vy_right_vec.npy"
    GP_params_Delta_vy = np.load(file_forces_params_Delta_vy_model_params)
    outputscale_Delta_vy = GP_params_Delta_vy[0]
    lengthscales_Delta_vy = GP_params_Delta_vy[1:]
    right_vec_block_Delta_vy = np.load(file_forces_params_Delta_vy_right_vec)
    central_mat_Delta_vy_vector = np.load(abs_path_parameters_folder + "/file_forces_params_central_matrix_vy_vec.npy")

    # w
    file_forces_params_Delta_w_model_params = abs_path_parameters_folder + "/forces_params_Delta_w_model_params.npy"
    file_forces_params_Delta_w_right_vec = abs_path_parameters_folder + "/file_forces_params_Delta_w_right_vec.npy"
    GP_params_Delta_w = np.load(file_forces_params_Delta_w_model_params)
    outputscale_Delta_w = GP_params_Delta_w[0]
    lengthscales_Delta_w = GP_params_Delta_w[1:]
    right_vec_block_Delta_w = np.load(file_forces_params_Delta_w_right_vec)
    central_mat_Delta_w_vector = np.load(abs_path_parameters_folder + "/file_forces_params_central_matrix_w_vec.npy")


    return x_data_vec_vx, x_data_vec_vy, x_data_vec_w, outputscale_Delta_vx, lengthscales_Delta_vx, right_vec_block_Delta_vx,\
                       outputscale_Delta_vy, lengthscales_Delta_vy, right_vec_block_Delta_vy, \
                       outputscale_Delta_w, lengthscales_Delta_w, right_vec_block_Delta_w,\
                       central_mat_Delta_vx_vector, central_mat_Delta_vy_vector, central_mat_Delta_w_vector,\
                        x_data_vec_vx_cov, x_data_vec_vy_cov, x_data_vec_w_cov