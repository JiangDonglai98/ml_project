# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2


# %%

class DataLoader:
    """
    author: Donglai Jiang
    functions:
    This is a class written to load and transform the given data to our project target data format.
    It includes the functionality of:
    1.Get all the file paths from a given root path, directory name, words of file name and the file format.
    2.Read the figure format files to the numpy.ndarray format.
    3.Show the read in images.
    4.read common files to the numpy.ndarray format.
    5.Transform the numpy.ndarray format files to the figure (default .jpg format).
    6.Transform the numpy.ndarray format files to the video (default .avi format) to run.
    """

    def __init__(self):
        self.dataset_dict = {}

    @staticmethod
    def get_file_names(root_path: str, dir_name: str, file_name: str, file_type: str) -> list:
        if os.path.isdir(root_path):
            file_list = glob.glob(os.path.join(root_path, dir_name) + '/*' + file_name + '*' + '.' + file_type)
            return file_list
        else:
            raise Exception("wrong paths are inputted!!! Please reconstruct a new root path.")

    @staticmethod
    def read_fig(path: str, shape: tuple = None, transpose: tuple = None) -> np.ndarray:
        im = plt.imread(path)
        name = path.split('\\')[-1]
        print('-' * 5 + name + '-' * 5)
        print("shape:", im.shape)
        print("type:", type(im))
        if shape:
            im.resize(shape)
        if transpose:
            im = np.transpose(im, transpose)
        print('Resize to shape:', im.shape)
        print("-" * 10)
        return im

    @staticmethod
    def show_img(img: np.ndarray, with_axes: bool = False):
        plt.imshow(img, cmap='gray')
        if not with_axes:
            plt.axis('off')
        plt.show()

    @staticmethod
    def read_file(path: str, shape: tuple = None, transpose: tuple = None, delimiter=",") -> np.ndarray:
        name = path.split('\\')[-1]
        with open(path) as f:
            file = np.loadtxt(f, delimiter=delimiter)
        print('-' * 5 + name + '-' * 5)
        print("shape:", file.shape)
        print("type:", type(file))
        if shape:
            file.resize(shape)
        if transpose:
            file = np.transpose(file, transpose)
        print('Resize to shape:', file.shape)
        print("-" * 10)
        return file

    @staticmethod
    def array_to_img(img: np.ndarray, out_path: str, name: str, out_form: str = '.jpg'):
        shape = img.shape
        print(f'=> target image frame size: {shape[0]}, {shape[1]}')
        cv2.imwrite(os.path.join(out_path, name + out_form), img)

    @staticmethod
    def array_to_video(video_file: np.ndarray, fps: int, width: int, height: int, out_path: str,
                       name: str, codec: str = 'DIVX', out_form: str = '.avi'):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(os.path.join(out_path, name + out_form), fourcc, fps, (width, height))  # , False
        print(f'=> target video frame size: {width}, {height}')
        for img in video_file:
            # DataLoader.show_img(img)
            cv_img = img.astype('uint8')
            gray_3c = cv2.merge([cv_img, cv_img, cv_img])
            # print(cv_img)
            # print(gray_3c.shape)
            out.write(gray_3c)
        out.release()

    @staticmethod
    def gen_gaussian_matrix(row_num: int, col_num: int, mean: float = 0.0, std: float = 1.0,
                            sparse: bool = False, ratio: float = 0.0, seed: int = 5) -> np.ndarray:
        np.random.seed(seed)
        if not sparse:
            matrix = np.random.normal(loc=mean, scale=std, size=(row_num, col_num))
        else:
            total_num = int(row_num * col_num * (1 - ratio))
            matrix = np.random.normal(loc=mean, scale=std, size=row_num * col_num)
            indices = np.random.choice(np.arange(row_num * col_num), replace=False, size=total_num)
            matrix[indices] = 0
            matrix = matrix.reshape(row_num, col_num)
        return matrix


    def read_in_images(self, root_path: str, dir_name: str, file_name: str, file_type: str, shape=None, transpose=None):
        file_path_list = self.get_file_names(root_path, dir_name, file_name, file_type)
        self.dataset_dict[file_name] = [self.read_fig(path, shape=shape, transpose=transpose) for path in
                                        file_path_list]

    def read_in_files(self, root_path: str, dir_name: str, file_name: str, file_type: str, shape=None, transpose=None):
        file_path_list = self.get_file_names(root_path, dir_name, file_name, file_type)
        self.dataset_dict[file_name] = [self.read_file(path, shape=shape, transpose=transpose) for path in
                                        file_path_list]


if __name__ == '__main__':
    root = r'G:\Master\master_course\Machine_Learning\course_project\Final_project_package'
    test_loader = DataLoader()
    test_loader.read_in_images(root, 'Data\\Yale-B02', 'yale', 'pgm')
    test_loader.read_in_files(root, r'Data', 'escalator', 'csv', (160, 130, 200), (2, 1, 0))
    test_loader.show_img(test_loader.dataset_dict['yale'][0])
    test_loader.show_img(test_loader.dataset_dict['escalator'][0][0])
    test_loader.array_to_video(test_loader.dataset_dict['escalator'][0], 10, 160, 130, root, 'test')
    test_loader.array_to_img(test_loader.dataset_dict['escalator'][0][0], root, r'Current_results\test_figure')
    test_img = cv2.imread(os.path.join(root, r'Data\Yale-B02\yaleB02_P00A+000E+20.pgm'))
    U_star = test_loader.gen_gaussian_matrix(50, 5, seed=5)
    V_star = test_loader.gen_gaussian_matrix(50, 5, seed=6)
    S_star = test_loader.gen_gaussian_matrix(50, 50, sparse=True, ratio=0.3, seed=7)
    print('U_star:\n', U_star)
    print('V_star:\n', V_star)
    print('S_star:\n', S_star)

