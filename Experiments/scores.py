import numpy as np
import GPy
import utils

lik = GPy.likelihoods.Bernoulli()

def calculate_scores_rand(U, m, X_train, y_train, K, inv_K):
    return np.random.rand(len(U))
    
def calculate_scores_vari(U, m, X_train, y_train, K, inv_K):
    max_score = m._raw_predict(U[0].reshape(-1, 1))[1]
    max_ind = 0
    scores = []
    
    for i in range(len(U)):
        if max_score < m._raw_predict(U[i].reshape(-1, 1))[1]:
            max_score = m._raw_predict(U[i].reshape(-1, 1))[1]
            max_ind = i
            
        scores.append(m._raw_predict(U[i].reshape(-1, 1))[1][0])
    
    return scores
    
def calculate_scores_RKHS(U, m, X_train, y_train, K, inv_K):
    max_score = -1
    max_ind = 0
    scores = []
    
    b = float(m.kern.K_of_r(0))
    a = np.zeros((X_train.shape[0], 1))
    y = y_train

    for j in range(len(U)):
        for i in range(X_train.shape[0]):
            a[i] = m.kern.K_of_r(np.linalg.norm(X_train[i] - U[j]))
            
        t = 1 if m.predict(U[j].reshape(-1, 1))[0][0] >= 0.5 else -1
        
        f_u = np.dot(np.transpose(y), np.dot(inv_K, a))

        score = (1 - t * f_u[0]) ** 2 / (b - np.dot(np.transpose(a), np.dot(inv_K, a)))
        score = score[0]
        
        if max_score < score:
            max_score = score
            max_ind = j
        scores.append(score)

    return scores
    
def calculate_scores_Hvar(U, m, X_train, y_train, K, inv_K):
    max_score = -1
    max_ind = 0
    scores = []
    
    b = float(m.kern.K_of_r(0))
    a = np.zeros((X_train.shape[0], 1))
    y = y_train

    for j in range(len(U)):
        for i in range(X_train.shape[0]):
            a[i] = m.kern.K_of_r(np.linalg.norm(X_train[i] - U[j]))
            
        t = 1 if m.predict(U[j].reshape(-1, 1))[0][0] >= 0.5 else -1
        
        f_u = np.dot(np.transpose(y), np.dot(inv_K, a))
        
        #lets try to eliminate demoninator
        score = (1 - t * f_u[0]) ** 2
        score = score[0]
        
        if max_score < score:
            max_score = score
            max_ind = j
        scores.append(score)

    return scores
    
def calculate_scores_sqsm(U, m, X_train, y_train, K, inv_K):
    max_score = -1
    max_ind = 0
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
            
        if max_score < score:
            max_score = score
            max_ind = i
        scores.append(score)
        
    return scores

def calculate_scores_l2fm(U, m, X_train, y_train, K, inv_K):
    max_score = -1
    scores = []
        
    b = m.kern.K_of_r(0)
    
    for i in range(U.shape[0]):
        score = 0
        t = 1 if m.predict(U[i].reshape(-1, 1))[0] >= 0.5 else -1
        
        for j in range(U.shape[0]):
            a = np.zeros((X_train.shape[0], 1))
    
            for k in range(inv_K.shape[0]):
                 a[k] = m.kern.K_of_r(np.linalg.norm(X_train[k] - U[j])) + np.random.uniform(0, 1e-8, 1)
            
            a_t_inv_K = np.dot(np.transpose(a), inv_K)
            aKa = np.dot(a_t_inv_K, a)
            
            diff = aKa - m.kern.K_of_r(np.linalg.norm(U[j] - U[i]))
            diff *= np.dot(a_t_inv_K, y_train) - t
            diff /= b - aKa
            
            score += diff ** 2
            
        score = np.sqrt(score)

        if max_score < score:
            max_score = score
        scores.append(score)
        
    return scores
