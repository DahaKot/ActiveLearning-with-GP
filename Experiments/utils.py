import numpy as np
import GPy

def get_inv_K(m, X_train):
    K = np.arange(X_train.shape[0]**2)
    K = K.reshape((X_train.shape[0], -1))
    K = np.zeros_like(K, dtype = 'float')

    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[0]):
            K[i][j] = m.kern.K_of_r(np.linalg.norm(X_train[i] - X_train[j]))
            K[i][j] += np.random.uniform(0, 1e-8, 1)
    
    return np.linalg.inv(K)

def get_K(m, X_train):
    K = np.arange(X_train.shape[0]**2)
    K = K.reshape((X_train.shape[0], -1))
    K = np.zeros_like(K, dtype = 'float')

    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[0]):
            K[i][j] = m.kern.K_of_r(np.linalg.norm(X_train[i] - X_train[j]))
            K[i][j] += np.random.uniform(0, 1e-8, 1)
    
    return K

# attention! works for only one new_point
def update_K(m, X_train, K, new_point, a):
    b = np.array([m.kern.K_of_r(0)])

    K = np.concatenate((K, a), axis = 1)
    a = np.concatenate((a, b), axis = 0)
    K = np.concatenate((K, np.transpose(a)), axis = 0)
    
    return K

# attention! works for only one new_point
def update_inv_K(m, X_train, inv_K, new_point, a):
    b = np.array([m.kern.K_of_r(0)])
    
    atK_1 = np.matmul(np.transpose(a), inv_K)
    K_1a = np.matmul(inv_K, a)
        
    c = 1 / (b - np.matmul(atK_1, a))
    
    inv_K = inv_K + c * np.matmul(K_1a, atK_1)
    
    inv_K = np.concatenate((inv_K, c*K_1a), axis = 1)
    atK_1 = np.concatenate((atK_1, np.array([[-1]])), axis = 1)
    inv_K = np.concatenate((inv_K, -c*atK_1), axis = 0)
    
    return inv_K