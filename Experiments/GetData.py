#configure plotting
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import matplotlib;matplotlib.rcParams['figure.figsize'] = (10,6.25)
import matplotlib;matplotlib.rcParams['text.usetex'] = True
import matplotlib;matplotlib.rcParams['font.size'] = 8
import matplotlib;matplotlib.rcParams['font.family'] = 'serif'

import GPy
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(1)

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
import pandas as pd

import ScoreFunctions

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
        return 1 / 0
        
    def draw_score(self, scores, U, score_name, iteration, point, label):
        return 1 / 0
        
class Generated1dimData(Data):
    def __init__(self, total_n, start_n, end_n, test_size):
        super(Generated1dimData, self).__init__(1, total_n, start_n, end_n, test_size)
    
    def generate_data(self):
        self.k = GPy.kern.RBF(self.ndim, variance=self.var, lengthscale=self.lengthscale)

        self.X = np.random.rand(self.total_n, 1)

        #draw the latent function value
        self.f = np.random.multivariate_normal(np.zeros(self.total_n), self.k.K(self.X))
        self.Y = self.lik.samples(self.f).reshape(-1,1)

        self.p = self.lik.gp_link.transf(self.f) # squash the latent function
        plt.plot(self.X, self.p, 'r.')
        plt.title('latent probabilities');plt.xlabel('$x$');plt.ylabel('$\sigma(f(x))$')

        self.U, self.X_train, self.y_U, self.y_train = train_test_split(self.X, self.Y, test_size = self.start_n)
        self.U, self.X_test, self.y_U, self.y_test = train_test_split(self.U, self.y_U, test_size = self.test_size)

        return self.U, self.X_train, self.y_U, self.y_train, self.X_test, self.y_test
    
    def draw_score(self, scores, U, score_name, iteration, point, label):
        scores = np.array(scores)
        order = np.argsort(U, axis = 0)

        U = np.array(U)[order].reshape(-1, self.ndim)
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
        
class Generated2dimData(Data):
    def __init__(self, total_n, start_n, end_n, test_size):
        super(Generated2dimData, self).__init__(2, total_n, start_n, end_n, test_size)
        
        self.n_points_to_show = 20
        self.n_points_per_axes = 10
    
    def generate_data(self):
        self.X, self.Y = make_blobs(np.array([self.total_n//2, self.total_n//2]), self.ndim, 
                                        cluster_std = 4, center_box = (-10, 10), shuffle = False, random_state = 2)
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.X.T[0], self.X.T[1], self.Y, zdir='z', s=20, c=None, depthshade=True)
        plt.savefig("InitialData.png")

        self.U, self.X_train, self.y_U, self.y_train = train_test_split(self.X, self.Y, test_size = self.start_n)
        self.U, self.X_test, self.y_U, self.y_test = train_test_split(self.U, self.y_U, test_size = self.test_size)

        return self.U, self.X_train, self.y_U, self.y_train, self.X_test, self.y_test
    
    def draw_score(self, m, U, score_name, iteration, X_train, score, y_train, inv_K):
        # creating grid for further plots
        points = np.array(np.meshgrid(np.linspace(-20, 20, self.n_points_per_axes), np.linspace(-20, 20, self.n_points_per_axes)))
        x, y = points[1], points[0]
        xy = points.T.reshape(-1,2)
        
        # 3-d plot of score-function
        z = score(xy, m, X_train, y_train, inv_K)
        np.savetxt("./scores/" + str(score)[27:31] + "/" + str(iteration) + ".txt", z)
        z = np.log(abs(z))
        probability = m.predict(xy)[0]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(self.X.T[0], self.X.T[1], self.Y*max(z), zdir='z', s=20, c=None, depthshade=True)
        ax.plot_surface(x, y, z.reshape(self.n_points_per_axes, self.n_points_per_axes), cmap='inferno', alpha = 0.4)

        plt.savefig(score_name + str(iteration) + 'score' + '.png')
        
        # 2-d plot of recently chosen points
#         plt.clf()
#         plt.plot(U.T[0], U.T[1], 'bo')

#         X_init = X_train[:self.n_points_to_show]
#         X_recent = X_train[-self.n_points_to_show:]
        
#         plt.plot(X_init.T[0], X_init.T[1], 'g+')
#         plt.plot(X_recent.T[0], X_recent.T[1], 'ro')     

        # probability plot
#         plt.clf()
        
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
        
#         ax.scatter(X_train.T[0], X_train.T[1], y_train*max(probability))
#         ax.plot_surface(x, y, probability.reshape(self.n_points_per_axes, self.n_points_per_axes), cmap='inferno', alpha = 0.4)

#         plt.savefig(score_name + str(iteration) + 'prob' + '.png')

        # countour plot of score-function
        plt.clf()
        z = z.reshape(points.shape[1], -1)

        plt.plot(X_train[np.nonzero(y_train)[0]].T[1], X_train[np.nonzero(y_train)[0]].T[0], 'bo', alpha = 0.1)
        plt.plot(X_train[np.where(y_train == 0)[0]].T[1], X_train[np.where(y_train == 0)[0]].T[0], 'ro', alpha = 0.1)
        plt.plot(X_train[-1].T[1], X_train[-1].T[0], 'k*')
        
        cmap = plt.cm.get_cmap("winter", 15)
        
        cs = plt.contour(np.linspace(-20, 20, self.n_points_per_axes), np.linspace(-20, 20, self.n_points_per_axes), z)
#         fig.colorbar(cs)
        plt.savefig(score_name + str(iteration) + 'contour' + '.png')
        
        plt.close("all")
        
class HTRU_2(Data):

    def __init__(self, start_n, end_n, test_size):
        super(HTRU_2, self).__init__(0, 0, start_n, end_n, test_size)
    
    def generate_data(self):
        dataset = pd.read_csv("HTRU_2.csv", header = None, 
                              names = ["0", "1", "2", "3", "4", "5", "6", "7", "class"])
        
        self.Y = dataset["class"].values
        self.X = dataset[["0", "1", "2", "3", "4", "5", "6", "7"]].values

        self.U, self.X_train, self.y_U, self.y_train = train_test_split(
                                                        self.X, 
                                                        self.Y, 
                                                        test_size = self.start_n,
                                                        train_size = self.end_n * 2)
        
        self.U, self.X_test, self.y_U, self.y_test = train_test_split(
                                                        self.U, 
                                                        self.y_U, 
                                                        test_size = self.test_size)

        return self.U, self.X_train, self.y_U, self.y_train, self.X_test, self.y_test
    
class Haberman(Data):

    def __init__(self, start_n, end_n, test_size):
        super(Haberman, self).__init__(0, 0, start_n, end_n, test_size)
    
    def generate_data(self):
        dataset = pd.read_csv("haberman.data", header = None, delimiter = ",",
                              names = ["0", "1", "2", "class"])
        
        self.Y = dataset["class"].values
        self.X = dataset[["0", "1", "2"]].values

        self.U, self.X_train, self.y_U, self.y_train = train_test_split(
                                                        self.X, 
                                                        self.Y, 
                                                        test_size = self.start_n)
        
        self.U, self.X_test, self.y_U, self.y_test = train_test_split(
                                                        self.U, 
                                                        self.y_U, 
                                                        test_size = self.test_size)

        return self.U, self.X_train, self.y_U - 1, self.y_train - 1, self.X_test, self.y_test - 1    
    
class Skin(Data):

    def __init__(self, start_n, end_n, test_size):
        super(Skin, self).__init__(0, 0, start_n, end_n, test_size)
    
    def generate_data(self):
        dataset = pd.read_csv("Skin_NonSkin.txt", header = None, delimiter = "\t",
                              names = ["0", "1", "2", "class"])
        
        self.Y = dataset["class"].values
        self.X = dataset[["0", "1", "2"]].values

        self.U, self.X_train, self.y_U, self.y_train = train_test_split(
                                                        self.X, 
                                                        self.Y, 
                                                        test_size = self.start_n,
                                                        train_size = self.end_n * 2)
        
        self.U, self.X_test, self.y_U, self.y_test = train_test_split(
                                                        self.U, 
                                                        self.y_U, 
                                                        test_size = self.test_size)

        return self.U, self.X_train, self.y_U - 1, self.y_train - 1, self.X_test, self.y_test - 1    