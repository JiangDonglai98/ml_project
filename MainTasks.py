# %%
import numpy as np
import pandas as pd
import os
from Algorithms import AIRLSGradient, SubGradient
from DataLoader import DataLoader
import matplotlib.pyplot as plt

root = r'G:\Master\master_course\Machine_Learning\course_project\Final_project_package'
task_loader = DataLoader()
task_loader.read_in_images(root, 'Data\\Yale-B02', 'yale', 'pgm')
task_loader.read_in_files(root, r'Data', 'escalator', 'csv', (160, 130, 200), (2, 1, 0))
task_loader.gen_artificial_matrices()


# %%
# task1
def compare_grad_norm(norm_dict, name, save_path):
    """
    :param      norm_dict: dict of calculated error norm value list with their mehtod name
    :param      name: plot name
    :param      save_path: your output picture path
    """
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(norm_dict.keys()))))
    plt.figure(figsize=(8, 6), dpi=200)
    picture_name = f'||theta_k − theta_star|| compare for {name}'
    plt.title(picture_name)
    for name_, list_ in norm_dict.items():
        c = next(color)
        plt.plot(range(len(list_)), list_, label=name_, marker='o', markersize=1, color=c)
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('error norm')
    plt.savefig(os.path.join(save_path, name + '.png'))
    plt.show()


def draw_error_norm(error_list: list, save_path: str, name: str):
    plt.figure(figsize=(8, 6), dpi=200)
    picture_name = '||U_k * V_k_T − L_star||'
    plt.title(picture_name)
    X_list = list(range(len(error_list)))
    plt.plot(X_list, error_list, color='red')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('error norm')
    plt.savefig(os.path.join(save_path, name + '.png'))
    plt.show()


init_U = task_loader.gen_gaussian_matrix(50, 5, seed=55)
init_V = task_loader.gen_gaussian_matrix(50, 5, seed=66)
task1_3_X_star = task_loader.dataset_dict['artificial'][3]['X_star']
task1_3_L_star = task_loader.dataset_dict['artificial'][3]['L_star']
opts_task1 = {'sigma': 0.0001, 'iter_num': 80}
task1_AIRLS = AIRLSGradient(task1_3_X_star, init_U, init_V, opts_task1)
task1_AIRLS.gradient()


def cal_error_list(U_k_list: np.ndarray, V_k_list: np.ndarray, L_star: np.ndarray):
    error_list = [np.linalg.norm(U_k_list[i] @ (V_k_list[i]).T - L_star) for i in range(len(U_k_list))]
    return error_list


task_1_a_error_list = cal_error_list(task1_AIRLS.U_list, task1_AIRLS.V_list, task1_3_L_star)
draw_error_norm(task_1_a_error_list, os.path.join(root, 'Current_results'), f'/task1_a')

# %%
# task2

# %%
# task3
