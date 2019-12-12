#configure plotting
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import matplotlib;matplotlib.rcParams['figure.figsize'] = (4,2.5)
import matplotlib;matplotlib.rcParams['text.usetex'] = True
import matplotlib;matplotlib.rcParams['font.size'] = 8
import matplotlib;matplotlib.rcParams['font.family'] = 'serif'

import GPy
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
np.random.seed(1)

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

class Data(nn.Module):
    def __init__(self, ndim, total_n, start_n, end_n, test_size):
        super(Data, self).__init__()
        
        self.var = 100
        self.lengthscale = 0.11

        self.lik = GPy.likelihoods.Bernoulli()
        
        self.ndim = ndim
        
        self.total_n = total_n
        self.start_n = start_n
        self.end_n = end_n
        self.test_size = test_size
    
    def generate_data(self):
        if self.ndim == 1:
            self.k = GPy.kern.RBF(self.ndim, variance=self.var, lengthscale=self.lengthscale)
            
            self.X = np.random.rand(self.total_n, 1)

            #draw the latent function value
            self.f = np.random.multivariate_normal(np.zeros(self.total_n), self.k.K(self.X))
            self.Y = self.lik.samples(self.f).reshape(-1,1)
            
            self.p = self.lik.gp_link.transf(self.f) # squash the latent function
            plt.plot(self.X, self.p, 'r.')
            plt.title('latent probabilities');plt.xlabel('$x$');plt.ylabel('$\sigma(f(x))$')
            
        elif self.ndim == 2:
            self.X, self.Y = make_blobs(np.array([self.total_n//2, self.total_n//2]), self.ndim, 
                                        cluster_std = 3, center_box = (-20, 20), shuffle = True, random_state = 42)
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(self.X.T[0], self.X.T[1], self.Y, zdir='z', s=20, c=None, depthshade=True)

        self.U, self.X_train, self.y_U, self.y_train = train_test_split(self.X, self.Y, test_size = self.start_n)
        self.U, self.X_test, self.y_U, self.y_test = train_test_split(self.X, self.Y, test_size = self.test_size)

        return self.U, self.X_train, self.y_U, self.y_train, self.X_test, self.y_test
    
    def draw_score(self, scores, U, score_name, iteration, point, label):
        scores = np.array(scores)
        order = np.argsort(U, axis = 0)

        U = np.array(U)[order].reshape(-1, 1)

        scores = (np.array(scores)[order]).reshape(-1, 1)

        plt.clf()
        plt.plot(U, scores)
        plt.plot(point, max(scores), 'go')

        open(score_name + str(iteration) + 'score' + '.png', 'w+')
        plt.savefig(score_name + str(iteration) + 'score' + '.png')

        plt.clf()
        plt.plot(self.X, self.p, 'r.')
        plt.plot(point, label, 'go')

        open(score_name + str(iteration) + 'prob' + '.png', 'w+')
        plt.savefig(score_name + str(iteration) + 'prob' + '.png')

