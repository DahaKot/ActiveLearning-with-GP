import numpy as np
import GPy
import Utils

lik = GPy.likelihoods.Bernoulli()
ndim = 2

def calculate_scores_rand(U, m, X_train, y_train, inv_K):
    return (np.random.rand(U.reshape(-1, ndim).shape[0])).reshape(-1, 1)
    
def calculate_scores_vari(U, m, X_train, y_train, inv_K):
    return m._raw_predict(U.reshape(-1, ndim))[1]
    
def calculate_scores_RKHS(U, m, X_train, y_train, inv_K):
    b = np.full(U.shape[0], m.kern.K_of_r(0))
    A = m.kern.K(U, X_train).T
    
    t = np.array([1 if x[0] >= 0.5 else -1 for x in m.predict(U)[0]])
    
    K_1A = np.dot(inv_K, A)
    f_u = np.dot(np.transpose(y_train), K_1A)
    t_f_u = np.multiply(t, f_u)

    aKa = np.expand_dims(np.diagonal(np.dot(A.T, K_1A)), 1)
    
    return np.divide((np.ones(t_f_u.shape) - t_f_u) ** 2, (b.reshape(-1, 1) - aKa).T).T
    
def calculate_scores_Hvar(U, m, X_train, y_train, inv_K):
    A = m.kern.K(U, X_train)

    t = np.array([1 if x[0] >= 0.5 else -1 for x in m.predict(U)[0]])

    K_1A = np.dot(inv_K, A.T)
    f_u = np.dot(np.transpose(y_train), K_1A)
    t_f_u = np.multiply(t, f_u)

    return ((np.ones(t_f_u.shape) - t_f_u) ** 2).T
    
def calculate_scores_sqsm(U, m, X_train, y_train, inv_K):
    scores = []
    
    for i in range(U.shape[0]):
        t = 1 if m.predict(U[i].reshape(-1, ndim))[0] >= 0.5 else -1
        
        kernel = GPy.kern.RBF(ndim, variance=m.kern.variance[0], lengthscale=m.kern.lengthscale[0])

        m_v = GPy.core.GP(X = np.concatenate((X_train, U[i].reshape(-1, ndim)), axis = 0),
                Y = np.append(y_train, t).reshape(-1, 1), 
                kernel = kernel, 
                inference_method = GPy.inference.latent_function_inference.Laplace(),
                likelihood = lik)
        
        prediction_v = m_v.predict(U.reshape(-1, ndim))[0]
        prediction = m.predict(U.reshape(-1, ndim))[0]
         
        score = np.linalg.norm(prediction_v - prediction)

        scores.append(score)
        
    return np.array(scores).reshape(-1, 1)

def calculate_scores_l2fm(U, m, X_train, y_train, inv_K):
    b = np.full(U.shape[0], m.kern.K_of_r(0))
    A = m.kern.K(U, X_train).T

    AK_1 = np.dot(A.T, inv_K)
    aKa = np.expand_dims(np.diagonal(np.dot(AK_1, A)), 1)

    aKy = np.dot(AK_1, y_train)    
    t = np.array([1 if x[0] >= 0.5 else -1 for x in m.predict(U)[0]]).reshape(-1, 1)
    t = np.repeat(t, U.shape[0], axis = 1).T

    multiplier = np.divide(aKy - t, b - aKa)

    Kuv = m.kern.K(np.array(U), U)

    diff = np.multiply(multiplier, aKa - Kuv)

    return np.linalg.norm(diff, axis = 0)