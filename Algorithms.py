import numpy as np
from abc import abstractmethod, ABCMeta, ABC


class Gradient(metaclass=ABCMeta):
    @abstractmethod
    def gradient(self):
        pass


class SubGradient(Gradient, ABC):
    def __init__(self, X: np.ndarray, init_U: np.ndarray, init_V: np.ndarray, opts: dict):
        self.X = X
        self.init_U = init_U
        self.init_V = init_V
        self.opts = opts
        self.U_list = []
        self.V_list = []

    @staticmethod
    def sub_grad_U(X: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        :param X:
        :param U:
        :param V:
        :return:
        """
        return - V.T @ np.sign(X - U @ V.T)

    @staticmethod
    def sub_grad_V(X: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        pass

    def gradient(self):
        pass

    def __str__(self):
        return 'subgradient'


class AIRLSGradient(Gradient, ABC):
    def __init__(self, X: np.ndarray, init_U: np.ndarray, init_V: np.ndarray, opts: dict):
        self.X = X
        self.init_U = init_U
        self.init_V = init_V
        self.opts = opts
        self.U_list = []
        self.V_list = []

    @staticmethod
    def IRLS_VT(Y: np.ndarray, X: np.ndarray, W: np.ndarray, sigma: float) -> tuple:
        beta = np.float32(np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y)
        # diags = np.sqrt(np.linalg.norm(X, axis=1) + np.linalg.norm(beta.T, axis=1))
        # diags[diags < sigma] = sigma
        # diags = 1 / diags
        # W_t = np.diag(diags[:X.shape[0]])

        temp_W = np.float32(np.abs(Y - X @ beta))
        # temp_W[temp_W < sigma] = sigma
        # W_t = 1 / temp_W

        mean = np.mean(temp_W, axis=1)
        W_t = np.float32(np.diag(1 / np.maximum(sigma, mean)))
        return beta, W_t

    @staticmethod
    def IRLS_U(Y: np.ndarray, beta: np.ndarray, W: np.ndarray, sigma: float) -> tuple:
        X = np.float32(Y @ W @ beta.T @ np.linalg.inv(beta @ W @ beta.T))
        # diags = np.sqrt(np.linalg.norm(X, axis=1) + np.linalg.norm(beta.T, axis=1))
        # diags[diags < sigma] = sigma
        # diags = 1 / diags
        # W_t = np.diag(diags[:beta.shape[1]])

        temp_W = np.float32(np.abs(Y - X @ beta))
        # temp_W[temp_W < sigma] = sigma
        # W_t = 1 / temp_W

        mean = np.mean(temp_W, axis=0)
        W_t = np.float32(np.diag(1 / np.maximum(sigma, mean)))
        return X, W_t

    def gradient(self):
        W_VT_init = np.float32(np.eye(self.init_U.shape[0]))  # diag d
        W_U_init = np.float32(np.eye(self.init_V.shape[0]))   # diag n
        VT_k, W_t_V = AIRLSGradient.IRLS_VT(self.X, self.init_U, W_VT_init, self.opts['sigma'])
        U_k, W_t_U = AIRLSGradient.IRLS_U(self.X, self.init_V.T, W_U_init, self.opts['sigma'])
        # print("I am here")
        self.U_list.append(U_k)
        self.V_list.append(VT_k.T)
        for i in range(self.opts['iter_num']):
            # print(i)
            VT_k, W_t_V = AIRLSGradient.IRLS_VT(self.X, U_k, W_t_V, self.opts['sigma'])
            U_k, W_t_U = AIRLSGradient.IRLS_U(self.X, VT_k, W_t_U, self.opts['sigma'])
            self.U_list.append(U_k)
            self.V_list.append(VT_k.T)

    def __str__(self):
        return 'AIRLS'