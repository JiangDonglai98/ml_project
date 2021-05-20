# utf-8
# author: Donglai Jiang

# %%
import numpy as np
import os
from Algorithms import OriginAIRLSGradient, AIRLSGradient, SubGradient, KAIRLSGradient, ElementAIRLSGradient, L1AIRLSGradient, EqualAIRLSGradient
from DataLoader import DataLoader
import matplotlib.pyplot as plt
import time
import pandas as pd

plt.style.use('ggplot')
root = r'G:\Master\master_course\Machine_Learning\course_project\Final_project_package'
task_loader = DataLoader()
task_loader.read_in_images(root, 'Data' + os.sep + 'Yale-B02', 'yale', 'pgm')
task_loader.read_in_files(root, r'Data', 'escalator', 'csv', (160, 130, 200), (2, 1, 0))
task_loader.gen_artificial_matrices()

total_task_parm_compare_dict = {'task': [], 'algorithm': [], 'function': [], 'time': [], 'r': [],
                                'iter_num': [], 'sigma': [], 'miu': []}


# %%
def logging(task_name: str):
    def timer(function):
        """
        a time decorator
        :param function: function you wan to count on time
        :return: result of target function
        """

        def wrapper(*args, **kwargs):
            time_start = time.time()
            res = function(*args, **kwargs)
            names = function.__code__.co_varnames
            algo_name = ''
            try:
                par_dict = args[names.index('opts')]
            except:
                par_dict = None
            try:
                r = args[names.index('r')]
            except:
                r = None
            for i in args:
                if i is AIRLSGradient:
                    algo_name = AIRLSGradient.name()
                if i is SubGradient:
                    algo_name = SubGradient.name()
            cost_time = time.time() - time_start
            print('-' * 50)
            print("任务:{}, 算法:{},【{}】运行时间：【{}】秒".format(task_name, algo_name, function.__name__, cost_time))
            path = os.path.join(root, 'Current_results')
            filepath = os.path.join(path, 'project_log.log')
            with open(filepath, mode='a') as f:
                f.write('-' * 50 + '\n')
                f.write("任务:{}, 算法:{},【{}】运行时间：【{}】秒\n".format(task_name, algo_name, function.__name__, cost_time))
                total_task_parm_compare_dict['task'].append(task_name)
                total_task_parm_compare_dict['algorithm'].append(algo_name)
                total_task_parm_compare_dict['function'].append(function.__name__)
                total_task_parm_compare_dict['time'].append(cost_time)
                total_task_parm_compare_dict['r'].append(r)
                print("参数:")
                print(f"r: {r}")
                f.write("参数:\n")
                f.write(f"r: {r}\t")
                for name, value in par_dict.items():
                    print(f"{name}: {value}\t")
                    f.write(f"{name}: {value}\t")
                    if name == 'iter_num':
                        total_task_parm_compare_dict['iter_num'].append(value)
                if par_dict.get('sigma', -1) != -1:
                    total_task_parm_compare_dict['sigma'].append(value)
                else:
                    total_task_parm_compare_dict['sigma'].append(None)
                if par_dict.get('miu', -1) != -1:
                    total_task_parm_compare_dict['miu'].append(value)
                else:
                    total_task_parm_compare_dict['miu'].append(None)
                f.write('\n')
            return res

        return wrapper

    return timer


# %%
# task1
def compare_error_norm(norm_dict, name, save_path):
    """
    :param      norm_dict: dict of calculated error norm value list with their mehtod name
    :param      name: plot name
    :param      save_path: your output picture path
    """
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(norm_dict.keys()))))
    plt.figure(figsize=(8, 6), dpi=200)
    picture_name = f'||U_k * V_k_T − L_star|| compare for {name}'
    plt.title(picture_name)
    for name_, list_ in norm_dict.items():
        c = next(color)
        plt.plot(range(len(list_)), list_, label=name_, marker='o', markersize=1, color=c)
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('error norm')
    plt.savefig(os.path.join(save_path, name + '.png'))
    plt.show()


def compare_p_error_norm(error_list: list, save_path: str, name: str):
    plt.figure(figsize=(8, 6), dpi=200)
    picture_name = f'||U_k * V_k_T − L_star|| for different p of {name}'
    plt.title(picture_name)
    error_list.sort(key=lambda x: x[0])
    x_list = [float(p[0]) for p in error_list]
    y_list = [p[1] for p in error_list]
    plt.plot(x_list, y_list, marker='o', markersize=3, color='blue')
    plt.xlabel('p')
    plt.ylabel('error norm')
    plt.savefig(os.path.join(save_path, name + '.png'))
    plt.show()


def draw_error_norm(error_list: list, save_path: str, name: str):
    plt.figure(figsize=(8, 6), dpi=200)
    picture_name = '||U_k * V_k_T − L_star||'
    plt.title(picture_name)
    X_list = list(range(len(error_list)))
    plt.plot(X_list, error_list, color='red', label=f'{name}')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('error norm')
    plt.savefig(os.path.join(save_path, name + '.png'))
    plt.show()


def cal_error_list(U_k_list: np.ndarray, V_k_list: np.ndarray, L_star: np.ndarray):
    error_list = [np.linalg.norm(U_k_list[i] @ (V_k_list[i]).T - L_star) for i in range(len(U_k_list))]
    return error_list


@logging('task1')
def total_error_list_dict(p_dict: dict, init_U: np.ndarray, init_V: np.ndarray, grad_class, opts: dict) -> dict:
    """
    :param p_dict           all p and its U_star, V_star, S_star, L_star, X_star
    :param init_U:          same init U for all
    :param init_V:          same init V for all
    :param grad_class:      your implemented grad_class, must have method gradient and attribute U_list, V_list
    :param opts:            same other parameters for each gradient class
    :return:                dict of each p and its error list
    """
    result = {}
    for p, matrix_dict in p_dict.items():
        temp_grad = grad_class(matrix_dict['X_star'], init_U, init_V, opts)
        temp_grad.gradient()
        result[str(p / 10)] = cal_error_list(temp_grad.U_list, temp_grad.V_list, matrix_dict['L_star'])
    return result

# %%
init_U = task_loader.gen_gaussian_matrix(50, 5, seed=116020097)
init_V = task_loader.gen_gaussian_matrix(50, 5, seed=22041082)
task1_matrix_dict = task_loader.dataset_dict['artificial']
opts_task1_AIRLS = {'sigma': 1E-10, 'iter_num': 1000}
opts_task1_sub = {'iter_num': 1000, 'miu': 0.1}
# %%
task1_total_error_list_dict_OriginAIRLS = total_error_list_dict(task1_matrix_dict, init_U, init_V, OriginAIRLSGradient,
                                                          opts_task1_AIRLS)
task1_total_error_list_dict_AIRLS = total_error_list_dict(task1_matrix_dict, init_U, init_V, AIRLSGradient,
                                                          opts_task1_AIRLS)
task1_total_error_list_dict_subgradient = total_error_list_dict(task1_matrix_dict, init_U, init_V, SubGradient,
                                                                opts_task1_sub)
task1_total_error_list_dict_KAIRLS = total_error_list_dict(task1_matrix_dict, init_U, init_V, KAIRLSGradient,
                                                          opts_task1_AIRLS)
task1_total_error_list_dict_ElementAIRLS = total_error_list_dict(task1_matrix_dict, init_U, init_V, ElementAIRLSGradient,
                                                          opts_task1_AIRLS)
task1_total_error_list_dict_L1AIRLS = total_error_list_dict(task1_matrix_dict, init_U, init_V, L1AIRLSGradient,
                                                          opts_task1_AIRLS)
task1_total_error_list_dict_EqualAIRLS = total_error_list_dict(task1_matrix_dict, init_U, init_V, EqualAIRLSGradient,
                                                          opts_task1_AIRLS)

# parameter p = 0:3 algorithm error list plot
# OriginAIRLS plot
draw_error_norm(task1_total_error_list_dict_OriginAIRLS['0.3'], os.path.join(root, 'Current_results'), 'task1_a_OriginAIRLS')

# AIRLS plot
draw_error_norm(task1_total_error_list_dict_AIRLS['0.3'], os.path.join(root, 'Current_results'), 'task1_a_AIRLS')

# subgradient plot
draw_error_norm(task1_total_error_list_dict_subgradient['0.3'], os.path.join(root, 'Current_results'),
                'task1_a_subgradient')

# KAIRLS plot
draw_error_norm(task1_total_error_list_dict_KAIRLS['0.3'], os.path.join(root, 'Current_results'), 'task1_a_KAIRLS')

# ElementAIRLS plot
draw_error_norm(task1_total_error_list_dict_ElementAIRLS['0.3'], os.path.join(root, 'Current_results'), 'task1_a_ElementAIRLS')

# L1AIRLS plot
draw_error_norm(task1_total_error_list_dict_L1AIRLS['0.3'], os.path.join(root, 'Current_results'), 'task1_a_L1AIRLS')

# EqualAIRLS plot
draw_error_norm(task1_total_error_list_dict_EqualAIRLS['0.3'], os.path.join(root, 'Current_results'), 'task1_a_EqualAIRLS')

# %%
# p in {0.1; 0.2; 0.3; ...... ; 0.8} end error plot
# OriginAIRLS plot
OriginAIRLS_end_error_list = [(p, error[-1]) for p, error in task1_total_error_list_dict_OriginAIRLS.items()]
compare_p_error_norm(OriginAIRLS_end_error_list, os.path.join(root, 'Current_results'), 'task1_b_OriginAIRLS')

# AIRLS plot
AIRLS_end_error_list = [(p, error[-1]) for p, error in task1_total_error_list_dict_AIRLS.items()]
compare_p_error_norm(AIRLS_end_error_list, os.path.join(root, 'Current_results'), 'task1_b_AIRLS')

# subgradient plot
subgradient_end_error_list = [(p, error[-1]) for p, error in task1_total_error_list_dict_subgradient.items()]
compare_p_error_norm(subgradient_end_error_list, os.path.join(root, 'Current_results'), 'task1_b_subgradient')

# KAIRLS plot
KAIRLS_end_error_list = [(p, error[-1]) for p, error in task1_total_error_list_dict_KAIRLS.items()]
compare_p_error_norm(KAIRLS_end_error_list, os.path.join(root, 'Current_results'), 'task1_b_KAIRLS')

# ElementAIRLS plot
ElementAIRLS_end_error_list = [(p, error[-1]) for p, error in task1_total_error_list_dict_ElementAIRLS.items()]
compare_p_error_norm(ElementAIRLS_end_error_list, os.path.join(root, 'Current_results'), 'task1_b_ElementAIRLS')

# EqualAIRLS plot
EqualAIRLS_end_error_list = [(p, error[-1]) for p, error in task1_total_error_list_dict_EqualAIRLS.items()]
compare_p_error_norm(EqualAIRLS_end_error_list, os.path.join(root, 'Current_results'), 'task1_b_EqualAIRLS')

# %%
# task2
def gen_compare_face_plot(restore_face_list: list, true_face_list: list, grad: str, r: int):
    count = 1
    folder = os.path.join('Current_results', f'face_r{r}_{grad}')
    folder_path = os.path.join(root, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for fake_face, true_face in zip(restore_face_list, true_face_list):
        subtracted = true_face - fake_face
        temp_combine = task_loader.combine_array([true_face, fake_face, subtracted], axis=1)
        task_loader.show_img(temp_combine)
        task_loader.array_to_img(temp_combine, folder_path, f'task2_face_{count}_{grad}')
        count += 1


@logging('task2')
def task2_different_r_strategy(flatten_face_matrix: np.ndarray, grad_class, opts: dict, true_faces_list: list,
                               r: int, d1: int = 192, d2: int = 168, n: int = 64):
    if d1 != 1 and d2 != 1:
        init_U = task_loader.gen_gaussian_matrix(d1 * d2, r, seed=55)
        init_V = task_loader.gen_gaussian_matrix(n, r, seed=66)
    else:
        init_U = task_loader.gen_gaussian_matrix(d1, r, seed=55)
        init_V = task_loader.gen_gaussian_matrix(d2, r, seed=66)
    temp_grad = grad_class(flatten_face_matrix, init_U, init_V, opts)
    temp_grad.gradient()
    temp_faces = temp_grad.U_list[-1] @ temp_grad.V_list[-1].T
    if d1 != 1 and d2 != 1:
        restore_face_list = task_loader.restore_img(temp_faces, d1, d2)
        gen_compare_face_plot(restore_face_list, true_faces_list, temp_grad.__str__(), r)
    else:
        gen_compare_face_plot([temp_faces, ], true_faces_list, temp_grad.__str__(), r)


flatten_face_matrix = task_loader.flatten_img_list(task_loader.dataset_dict['yale'])
opts_task2_sub = {'iter_num': 200, 'miu': 0.01}
opts_task2_AIRLS = {'sigma': 0.0001, 'iter_num': 50}
init_U = task_loader.gen_gaussian_matrix(192 * 168, 20, seed=55)
init_V = task_loader.gen_gaussian_matrix(64, 20, seed=66)

# %%
# AIRLS plot
task2_different_r_strategy(flatten_face_matrix, AIRLSGradient, opts_task2_AIRLS, task_loader.dataset_dict['yale'], 30)
task2_different_r_strategy(flatten_face_matrix, AIRLSGradient, opts_task2_AIRLS, task_loader.dataset_dict['yale'], 20)
task2_different_r_strategy(flatten_face_matrix, AIRLSGradient, opts_task2_AIRLS, task_loader.dataset_dict['yale'], 10)
task2_different_r_strategy(flatten_face_matrix, AIRLSGradient, opts_task2_AIRLS, task_loader.dataset_dict['yale'], 5)
task2_different_r_strategy(flatten_face_matrix, AIRLSGradient, opts_task2_AIRLS, task_loader.dataset_dict['yale'], 3)
task2_different_r_strategy(flatten_face_matrix, AIRLSGradient, opts_task2_AIRLS, task_loader.dataset_dict['yale'], 2)
task2_different_r_strategy(flatten_face_matrix, EqualAIRLSGradient, opts_task2_AIRLS, task_loader.dataset_dict['yale'], 2)

# subgradient plot
task2_different_r_strategy(flatten_face_matrix, SubGradient, opts_task2_sub, task_loader.dataset_dict['yale'], 30)
task2_different_r_strategy(flatten_face_matrix, SubGradient, opts_task2_sub, task_loader.dataset_dict['yale'], 20)
task2_different_r_strategy(flatten_face_matrix, SubGradient, opts_task2_sub, task_loader.dataset_dict['yale'], 10)
task2_different_r_strategy(flatten_face_matrix, SubGradient, opts_task2_sub, task_loader.dataset_dict['yale'], 5)
task2_different_r_strategy(flatten_face_matrix, SubGradient, opts_task2_sub, task_loader.dataset_dict['yale'], 3)
task2_different_r_strategy(flatten_face_matrix, SubGradient, opts_task2_sub, task_loader.dataset_dict['yale'], 2)


# %%
# task3
def gen_compare_video(restore_video_tensor: np.ndarray, true_video_tensor: np.ndarray, grad: str, r: int, d1: int,
                      d2: int):
    name = os.path.join('Current_results', f'video_r{r}_{grad}')
    total_video = []
    for fake_video, true_video in zip(restore_video_tensor, true_video_tensor):
        subtracted = true_video - fake_video
        total_video.append(task_loader.combine_array([true_video, fake_video, subtracted], axis=1))

    task_loader.array_to_video(np.array(total_video), 10, d2 * 3, d1, root, name)


@logging('task3')
def task3_different_r_strategy(flatten_video_matrix: np.ndarray, grad_class, opts: dict, true_video_tensor: np.ndarray,
                               r: int, d1: int = 130, d2: int = 160, n: int = 200):
    init_U = task_loader.gen_gaussian_matrix(d1 * d2, r, seed=55)
    init_V = task_loader.gen_gaussian_matrix(n, r, seed=66)
    temp_grad = grad_class(flatten_video_matrix, init_U, init_V, opts)
    temp_grad.gradient()
    temp_videos = temp_grad.U_list[-1] @ temp_grad.V_list[-1].T
    restore_video_tensor = np.array(task_loader.restore_img(temp_videos, d1, d2))
    gen_compare_video(restore_video_tensor, true_video_tensor, temp_grad.__str__(), r, d1, d2)


flatten_video_matrix = task_loader.flatten_img_list(task_loader.dataset_dict['escalator'][0])
opts_task3_sub = {'iter_num': 300, 'miu': 0.01}
opts_task3_AIRLS = {'sigma': 0.0001, 'iter_num': 100}
# %%
# AIRLS video
task3_different_r_strategy(flatten_video_matrix, AIRLSGradient, opts_task3_AIRLS,
                           task_loader.dataset_dict['escalator'][0],
                           2)
task3_different_r_strategy(flatten_video_matrix, AIRLSGradient, opts_task3_AIRLS,
                           task_loader.dataset_dict['escalator'][0],
                           3)
task3_different_r_strategy(flatten_video_matrix, AIRLSGradient, opts_task3_AIRLS,
                           task_loader.dataset_dict['escalator'][0],
                           5)
task3_different_r_strategy(flatten_video_matrix, AIRLSGradient, opts_task3_AIRLS,
                           task_loader.dataset_dict['escalator'][0],
                           10)
task3_different_r_strategy(flatten_video_matrix, AIRLSGradient, opts_task3_AIRLS,
                           task_loader.dataset_dict['escalator'][0],
                           20)

# subgradient video
task3_different_r_strategy(flatten_video_matrix, SubGradient, opts_task3_sub, task_loader.dataset_dict['escalator'][0],
                           2)
task3_different_r_strategy(flatten_video_matrix, SubGradient, opts_task3_sub, task_loader.dataset_dict['escalator'][0],
                           3)
task3_different_r_strategy(flatten_video_matrix, SubGradient, opts_task3_sub, task_loader.dataset_dict['escalator'][0],
                           5)
task3_different_r_strategy(flatten_video_matrix, SubGradient, opts_task3_sub, task_loader.dataset_dict['escalator'][0],
                           10)
task3_different_r_strategy(flatten_video_matrix, SubGradient, opts_task3_sub, task_loader.dataset_dict['escalator'][0],
                           20)

# %%
parms_df = pd.DataFrame(total_task_parm_compare_dict)
parms_df.to_csv(os.path.join(root, os.path.join('Current_results', 'project_function_parameters.csv')))
