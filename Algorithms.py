import numpy as np


class SubGradient:
    @classmethod
    def name(cls):
        return 'subgradient'

    def __init__(self, X: np.ndarray, init_U: np.ndarray, init_V: np.ndarray, opts: dict):
        self.X = X
        self.init_U = init_U
        self.init_V = init_V
        self.opts = opts
        self.U_list = []
        self.V_list = []

    # @staticmethod
    # def sub_grad_U(X: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    #     result = X.T - V @ U.T
    #     result = np.sign(result)
    #     return - V.T @ result
    #
    # @staticmethod
    # def sub_grad_V(X: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    #     result = X - U @ V.T
    #     result = np.sign(result)
    #     return - U.T @ result

    @staticmethod
    # @jit(nopython=True)
    def sub_grad_U(X, y, theta):
        result = y.T - X @ theta
        result = np.sign(result)
        return np.float32(- X.T @ result)

    @staticmethod
    # @jit(nopython=True)
    def sub_grad_V(X, y, theta):
        result = y - X @ theta
        result = np.sign(result)
        return np.float32(- X.T @ result)

    def gradient(self):
        t = 0
        U_k = self.init_U
        V_k = self.init_V
        # miu_U = self.opts['miu']
        # miu_V = self.opts['miu']
        miu = self.opts['miu']
        while t < self.opts['iter_num']:
            self.U_list.append(U_k)
            self.V_list.append(V_k)
            # U_k_T = U_k.T - miu_U * SubGradient.sub_grad_U(self.X, U_k, V_k)
            # V_k_T = V_k.T - miu_V * SubGradient.sub_grad_V(self.X, U_k, V_k)
            # U_k_T = U_k.T - miu * SubGradient.sub_grad_U(self.X, U_k, V_k)
            # V_k_T = V_k.T - miu * SubGradient.sub_grad_V(self.X, U_k, V_k)
            U_k_T = U_k.T - miu / (t + 1) * SubGradient.sub_grad_U(V_k, self.X, U_k.T)
            V_k_T = V_k.T - miu / (t + 1) * SubGradient.sub_grad_V(U_k, self.X, V_k.T)
            U_k = U_k_T.T
            V_k = V_k_T.T
            # miu_U = (np.linalg.norm(U_k - self.U_list[-1])) / np.linalg.norm(subgrad_U(V_k, X, U_k.T))
            # miu_V = (np.linalg.norm(V_k - self.V_list[-1])) / np.linalg.norm(subgrad_V(U_k, X, V_k.T))
            # miu_U = c /
            # miu_V = c /
            t += 1
        self.U_list.append(U_k)
        self.V_list.append(V_k)

    def __str__(self):
        return 'subgradient'


class AIRLSGradient:
    @classmethod
    def name(cls):
        return 'AIRLS'

    def __init__(self, X: np.ndarray, init_U: np.ndarray, init_V: np.ndarray, opts: dict):
        self.X = X
        self.init_U = init_U
        self.init_V = init_V
        self.opts = opts
        self.U_list = []
        self.V_list = []

    @staticmethod
    # @jit(nopython=True)
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
    # @jit(nopython=True)
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
        W_U_init = np.float32(np.eye(self.init_V.shape[0]))  # diag n
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
