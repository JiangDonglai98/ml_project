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


class OriginAIRLSGradient(AIRLSGradient):
    @staticmethod
    def IRLS(Y: np.ndarray, X: np.ndarray, sigma: float, iter_num: int = 2, tol: float = 10E-4) -> np.ndarray:
        n, p = X.shape
        W = np.eye(n)
        beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
        delta = np.array([sigma] * n).reshape(1, n)
        for iter in range(iter_num):
            beta_last = beta
            W = (1.0 / np.maximum(delta, abs(Y - X.dot(beta)).T))[0]
            W = np.diag(W)
            beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
            error = sum(abs(beta - beta_last))
            if error < tol:
                return beta
        return beta

    def gradient(self):
        d, n = self.X.shape
        r = self.init_U.shape[1]
        U = self.init_U.copy()
        V = self.init_V.copy()
        for i in range(self.opts['iter_num']):

            for j in range(n):
                y = self.X[:, j].reshape(d, 1)
                theta = OriginAIRLSGradient.IRLS(y, U, self.opts['sigma'])
                V[j] = theta.T

            for k in range(d):
                y = self.X[k].T.reshape(n, 1)
                theta = OriginAIRLSGradient.IRLS(y, V, self.opts['sigma'])
                U[k] = theta.T.reshape(1, r)
            self.U_list.append(U.copy())
            self.V_list.append(V.copy())

    def __str__(self):
        return 'OriginAIRLS'


class EqualAIRLSGradient(AIRLSGradient):
    @staticmethod
    def matrix_to_tensor(first_dim: int, M: np.ndarray) -> np.ndarray:
        expanded = np.zeros((first_dim,) + (M.shape[1], M.shape[1]), dtype=M.dtype)
        diag = np.arange(M.shape[1])
        expanded[:, diag, diag] = M
        return expanded

    @staticmethod
    def IRLS_VT(Y: np.ndarray, X: np.ndarray, W: np.ndarray, sigma: float) -> tuple:
        tensor_W = EqualAIRLSGradient.matrix_to_tensor(Y.shape[1], W)
        beta = np.float32(np.einsum('ijk,ki->ij', np.linalg.inv(np.einsum('ij,kjl->kil', X.T, tensor_W) @ X) @ X.T @ tensor_W, Y)).T

        temp_W = np.float32(np.abs(Y - X @ beta))

        W_t = np.float32(1 / np.maximum(sigma, temp_W)).T
        return beta, W_t

    @staticmethod
    def IRLS_U(Y: np.ndarray, beta: np.ndarray, W: np.ndarray, sigma: float) -> tuple:
        tensor_W = EqualAIRLSGradient.matrix_to_tensor(Y.shape[0], W)
        X = np.float32(np.einsum('ij,ijk->ik', Y, tensor_W @ beta.T @ np.linalg.inv(np.einsum('ij,kjl->kil', beta, tensor_W) @ beta.T)))

        temp_W = np.float32(np.abs(Y - X @ beta))

        W_t = np.float32(1 / np.maximum(sigma, temp_W))
        return X, W_t

    def gradient(self):
        W_VT_init = np.float32(np.ones((self.X.shape[1], self.X.shape[0])))  # diag d
        W_U_init = np.float32(np.ones((self.X.shape[0], self.X.shape[1])))  # diag n
        VT_k, W_t_V = EqualAIRLSGradient.IRLS_VT(self.X, self.init_U, W_VT_init, self.opts['sigma'])
        U_k, W_t_U = EqualAIRLSGradient.IRLS_U(self.X, self.init_V.T, W_U_init, self.opts['sigma'])
        # print("I am here")
        self.U_list.append(U_k)
        self.V_list.append(VT_k.T)
        for i in range(self.opts['iter_num']):
            # print(i)
            VT_k, W_t_V = EqualAIRLSGradient.IRLS_VT(self.X, U_k, W_t_V, self.opts['sigma'])
            U_k, W_t_U = EqualAIRLSGradient.IRLS_U(self.X, VT_k, W_t_U, self.opts['sigma'])
            self.U_list.append(U_k)
            self.V_list.append(VT_k.T)

    def __str__(self):
        return 'EqualAIRLS'

class L1AIRLSGradient(AIRLSGradient):
    """
    use the l1 replace mean
    """

    @staticmethod
    def IRLS_VT(Y: np.ndarray, X: np.ndarray, W: np.ndarray, sigma: float) -> tuple:
        beta = np.float32(np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y)

        temp_W = np.float32(np.abs(Y - X @ beta))

        mean = np.max(temp_W, axis=1)
        W_t = np.float32(np.diag(1 / np.maximum(sigma, mean)))
        return beta, W_t

    @staticmethod
    def IRLS_U(Y: np.ndarray, beta: np.ndarray, W: np.ndarray, sigma: float) -> tuple:
        X = np.float32(Y @ W @ beta.T @ np.linalg.inv(beta @ W @ beta.T))
        temp_W = np.float32(np.abs(Y - X @ beta))

        mean = np.max(temp_W, axis=0)
        W_t = np.float32(np.diag(1 / np.maximum(sigma, mean)))
        return X, W_t

    def gradient(self):
        W_VT_init = np.float32(np.eye(self.init_U.shape[0]))  # diag d
        W_U_init = np.float32(np.eye(self.init_V.shape[0]))  # diag n
        VT_k, W_t_V = L1AIRLSGradient.IRLS_VT(self.X, self.init_U, W_VT_init, self.opts['sigma'])
        U_k, W_t_U = L1AIRLSGradient.IRLS_U(self.X, self.init_V.T, W_U_init, self.opts['sigma'])
        # print("I am here")
        self.U_list.append(U_k)
        self.V_list.append(VT_k.T)
        for i in range(self.opts['iter_num']):
            # print(i)
            VT_k, W_t_V = L1AIRLSGradient.IRLS_VT(self.X, U_k, W_t_V, self.opts['sigma'])
            U_k, W_t_U = L1AIRLSGradient.IRLS_U(self.X, VT_k, W_t_U, self.opts['sigma'])
            self.U_list.append(U_k)
            self.V_list.append(VT_k.T)


class KAIRLSGradient(AIRLSGradient):
    """
    use diminishing k of Weight W
    """

    def gradient(self):
        t = 1
        W_VT_init = np.float32(np.eye(self.init_U.shape[0]))  # diag d
        W_U_init = np.float32(np.eye(self.init_V.shape[0]))  # diag n
        VT_k, W_t_V = KAIRLSGradient.IRLS_VT(self.X, self.init_U, W_VT_init, self.opts['sigma'])
        U_k, W_t_U = KAIRLSGradient.IRLS_U(self.X, self.init_V.T, W_U_init, self.opts['sigma'])
        self.U_list.append(U_k)
        self.V_list.append(VT_k.T)
        for i in range(self.opts['iter_num']):
            VT_k, W_t_V = KAIRLSGradient.IRLS_VT(self.X, U_k, W_t_V, self.opts['sigma'])
            U_k, W_t_U = KAIRLSGradient.IRLS_U(self.X, VT_k, W_t_U, self.opts['sigma'])
            W_t_V = W_t_V / t
            W_t_U = W_t_U / t
            self.U_list.append(U_k)
            self.V_list.append(VT_k.T)
            t += 1

    def __str__(self):
        return 'KAIRLS'


class ElementAIRLSGradient(AIRLSGradient):
    @staticmethod
    def swallow(X: np.ndarray, length: int) -> np.ndarray:
        result = np.zeros(length)
        X_len = len(X)
        if X_len <= length:
            result[:X_len] = X
        else:
            result = X[:length]
        return result

    @staticmethod
    def IRLS_VT(Y: np.ndarray, X: np.ndarray, W: np.ndarray, sigma: float) -> tuple:
        beta = np.float32(np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y)
        length = len(beta.T)
        dxd = ElementAIRLSGradient.swallow(np.linalg.norm(X, axis=1), length)
        nxn = ElementAIRLSGradient.swallow(np.linalg.norm(beta.T, axis=1), length)
        diags = np.sqrt(dxd + nxn)
        diags[diags < sigma] = sigma
        diags = 1 / diags
        W_t = np.float32(np.diag(diags))
        return beta, W_t

    @staticmethod
    def IRLS_U(Y: np.ndarray, beta: np.ndarray, W: np.ndarray, sigma: float) -> tuple:
        X = np.float32(Y @ W @ beta.T @ np.linalg.inv(beta @ W @ beta.T))
        length = len(X)
        dxd = ElementAIRLSGradient.swallow(np.linalg.norm(X, axis=1), length)
        nxn = ElementAIRLSGradient.swallow(np.linalg.norm(beta.T, axis=1), length)
        diags = np.sqrt(dxd + nxn)
        diags[diags < sigma] = sigma
        diags = 1 / diags
        W_t = np.float32(np.diag(diags))
        return X, W_t

    def gradient(self):
        W_VT_init = np.float32(np.eye(self.init_U.shape[0]))  # diag d
        W_U_init = np.float32(np.eye(self.init_V.shape[0]))  # diag n
        VT_k, W_t_V = ElementAIRLSGradient.IRLS_VT(self.X, self.init_U, W_VT_init, self.opts['sigma'])
        U_k, W_t_U = ElementAIRLSGradient.IRLS_U(self.X, self.init_V.T, W_U_init, self.opts['sigma'])
        # print("I am here")
        self.U_list.append(U_k)
        self.V_list.append(VT_k.T)
        for i in range(self.opts['iter_num']):
            # print(i)
            VT_k, W_t_V = ElementAIRLSGradient.IRLS_VT(self.X, U_k, W_t_V, self.opts['sigma'])
            U_k, W_t_U = ElementAIRLSGradient.IRLS_U(self.X, VT_k, W_t_U, self.opts['sigma'])
            self.U_list.append(U_k)
            self.V_list.append(VT_k.T)
