import numpy as np
import GPy
import utils

lik = GPy.likelihoods.Bernoulli()

def calculate_scores_rand(U, m, X_train, y_train, K, inv_K):
    return (np.random.rand(len(U))).reshape(-1, 1)
    
def calculate_scores_vari(U, m, X_train, y_train, K, inv_K):
    return m._raw_predict(U.reshape(-1, 1))[1]
    
def calculate_scores_RKHS(U, m, X_train, y_train, K, inv_K):
    b = np.full(U.shape, m.kern.K_of_r(0))
    A = m.kern.K(U, X_train).T
    
    t = np.array([1 if x[0] >= 0.5 else -1 for x in m.predict(U)[0]])
    
    K_1A = np.dot(inv_K, A)
    f_u = np.dot(np.transpose(y_train), K_1A)
    t_f_u = np.multiply(t, f_u)

    aKa = np.expand_dims(np.diagonal(np.dot(A.T, K_1A)), 1)

    return np.divide((np.ones(t_f_u.shape) - t_f_u) ** 2, (b - aKa).T).T
    
def calculate_scores_Hvar(U, m, X_train, y_train, K, inv_K):
    A = m.kern.K(U, X_train)

    t = np.array([1 if x[0] >= 0.5 else -1 for x in m.predict(U)[0]])

    K_1A = np.dot(inv_K, A.T)
    f_u = np.dot(np.transpose(y_train), K_1A)
    t_f_u = np.multiply(t, f_u)

    return ((np.ones(t_f_u.shape) - t_f_u) ** 2).T
    
def calculate_scores_sqsm(U, m, X_train, y_train, K, inv_K):
    scores = []
    
    for i in range(U.shape[0]):
        t = 1 if m.predict(U[i].reshape(-1, 1))[0] >= 0.5 else -1
        
        kernel = GPy.kern.RBF(1, variance=m.kern.variance[0], lengthscale=m.kern.lengthscale[0])

        m_v = GPy.core.GP(X = np.append(X_train, U[i]).reshape(-1, 1),
                Y = np.append(y_train, t).reshape(-1, 1), 
                kernel = kernel, 
                inference_method = GPy.inference.latent_function_inference.Laplace(),
                likelihood = lik)
        
        prediction_v = m_v.predict(U.reshape(-1, 1))[0]
        prediction = m.predict(U.reshape(-1, 1))[0]
         
        score = np.linalg.norm(prediction_v - prediction)

        scores.append(score)
        
    return np.array(scores).reshape(-1, 1)

def calculate_scores_l2fm(U, m, X_train, y_train, K, inv_K):
    scores = []
        
    b = m.kern.K_of_r(0)
    
    for i in range(U.shape[0]):
        score = 0
        t = 1 if m.predict(U[i].reshape(-1, 1))[0] >= 0.5 else -1
        
        for j in range(U.shape[0]):
            a = np.zeros((X_train.shape[0],1))
    
            for k in range(inv_K.shape[0]):
                a[k] = m.kern.K_of_r(np.linalg.norm(X_train[k] - U[j])) + np.random.uniform(0, 1e-8, 1)
            
            a_t_inv_K = np.dot(np.transpose(a), inv_K)
            aKa = np.dot(a_t_inv_K, a)
            
            diff = aKa - m.kern.K_of_r(np.linalg.norm(U[j] - U[i]))
            diff *= np.dot(a_t_inv_K, y_train) - t
            diff /= b - aKa
            
            score += diff ** 2
            
        score = np.sqrt(score)

        scores.append(score)
        
    return np.array(scores).reshape(-1, 1)