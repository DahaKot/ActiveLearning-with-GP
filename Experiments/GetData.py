import GPy
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification

import pandas as pd

class Data(nn.Module):
    def __init__(self, total_n_, start_n_, end_n_, test_n_, ndim_):
        super(Data, self).__init__()
        
        self.ndim = ndim_
        
        self.total_n = total_n_
        self.start_n = start_n_
        self.end_n = end_n_
        self.test_n = test_n_
        
class OneDim(Data):
    def __init__(self, total_n_, start_n_, end_n_, test_n_):
        super(OneDim, self).__init__(total_n_, start_n_, end_n_, test_n_, 1)
        
        self.var = 200
        self.lengthscale = 0.1
        
        self.lik = GPy.likelihoods.Bernoulli()
    
    def generate_data(self, draw_data, exp_name):
        self.k = GPy.kern.RBF(self.ndim, variance = self.var, lengthscale = self.lengthscale)

        self.X = np.random.rand(self.total_n, 1)

        # latent function
        self.f = np.random.multivariate_normal(np.zeros(self.total_n), self.k.K(self.X))
        
        self.y = self.lik.samples(self.f).reshape(-1,1)

        # squash the latent function
        self.p = self.lik.gp_link.transf(self.f)
        
        #draw the latent function value
        if (draw_data):
            plt.plot(self.X, self.p, "r.")
            plt.plot(self.X, self.y, "bo")
            plt.title("latent probabilities")
            plt.xlabel("x")
            plt.ylabel("$\sigma(f(x))$")
            
            plt.savefig("Results/" + exp_name + "/initial_data.png")

        self.U, self.X_train, self.y_U, self.y_train = train_test_split(self.X, self.y, test_size = self.start_n)
        self.U, self.X_test, self.y_U, self.y_test   = train_test_split(self.U, self.y_U, test_size = self.test_n)
        
        plt.plot(self.U, self.y_U, "ro")
        plt.plot(self.X_train, self.y_train, "bo")

        return self.U, self.X_train, self.y_U, self.y_train, self.X_test, self.y_test
    
    def draw_score(self, scores, score_name, exp_name, iteration, m, score, inv_K):
        scores = np.array(scores)
        order = np.argsort(self.U, axis = 0)

        U = np.array(self.U)[order].reshape(-1, self.ndim)
        scores = (scores[order]).reshape(-1, 1)

        # draw score function
        plt.clf()
        plt.plot(U, scores)
        plt.plot(self.X_train[-1], max(scores), "go")
        
        plt.xlabel("U")
        plt.ylabel("score-function")
        
        open("./Results/" + exp_name + "/score_plots/" + score_name + "/" + str(iteration) + "score.png", "w+")
        plt.savefig("./Results/" + exp_name + "/score_plots/" + score_name + "/" + str(iteration) + "score.png")

        # draw probabilities
        plt.clf()
        plt.plot(self.X, self.p, "r.")
        plt.plot(self.X_train[-1], self.y_train[-1], "go")
        
        plt.xlabel("X")
        plt.ylabel("p(x $\in$ first class)")

        open("./Results/" + exp_name + "/score_plots/" + score_name + "/" + str(iteration) + "prob.png", "w+")
        plt.savefig("./Results/" + exp_name + "/score_plots/" + score_name + "/" + str(iteration) + "prob.png")
        
class TwoDim(Data):
    def __init__(self, total_n, start_n, end_n, test_n):
        super(TwoDim, self).__init__(total_n, start_n, end_n, test_n, 2)
        
        self.n_points_to_show = 10
        self.n_points_per_axis = 100
        
        # flags for different types of plots
        
        self.surface = True
        self.recent  = True
        self.prob    = True
        self.contour = True
    
    def generate_data(self, draw_data, exp_name):
        # make blobs can be replaced by make_classification
        self.X, self.Y = make_blobs(
            np.array([self.total_n//2, self.total_n//2]), 
            self.ndim, 
            cluster_std  = 4, 
            center_box   = (-10, 10), 
            shuffle      = False, 
            random_state = 1)
            
        if (draw_data): 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = "3d")

            ax.scatter(self.X.T[0], self.X.T[1], self.Y, zdir = "z", s = 20, c = None, depthshade = True)
            
            plt.xlabel("x")
            plt.ylabel("y")
            
            plt.savefig("Results/" + exp_name + "/initial_data.png")

        self.U, self.X_train, self.y_U, self.y_train = train_test_split(self.X, self.Y, test_size = self.start_n)
        self.U, self.X_test, self.y_U, self.y_test   = train_test_split(self.U, self.y_U, test_size = self.test_n)

        return self.U, self.X_train, self.y_U, self.y_train, self.X_test, self.y_test
    
    def draw_score(self, scores, score_name, exp_name, iteration, m, score, inv_K):
        # 1. create grid for further plots
        axis = np.linspace(-20, 20, self.n_points_per_axis)
        
        points = np.array(np.meshgrid(axis, axis))
        x, y = points[1], points[0]
        xy = points.T.reshape(-1, 2)
        
        # 2. calculate score on the grid
        z = score(xy, m, self.X_train, self.y_train, inv_K)
        np.savetxt("Results/" + exp_name + "/scores/" + score_name + "/" + str(iteration) + ".txt", z)
        
        z = np.log(abs(z))
        
        # 3. plot 3-d score surface
        if self.surface:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = "3d")

            ax.scatter(self.X_train.T[0], self.X_train.T[1], self.y_train * max(z), zdir = "z", s = 20, depthshade = True)
            ax.plot_surface(x, y, z.reshape(self.n_points_per_axis, self.n_points_per_axis), cmap = 'inferno', alpha = 0.4)

            plt.xlabel("x")
            plt.ylabel("y")

            plt.savefig("./Results/" + exp_name + "/score_plots/" + score_name + "/" + str(iteration) + "surface.png")
        
        # 3. 2-d plot of recently chosen points
        if self.recent:
            plt.clf()
            plt.plot(self.U.T[0], self.U.T[1], 'bo')

            X_init = self.X_train[:self.n_points_to_show]
            X_recent = self.X_train[-self.n_points_to_show:]

            plt.plot(X_init.T[0], X_init.T[1], "g+")
            plt.plot(X_recent.T[0], X_recent.T[1], "ro")
            
            plt.xlabel("x")
            plt.ylabel("y")

            plt.savefig("./Results/" + exp_name + "/score_plots/" + score_name + "/" + str(iteration) + "recent_points.png")

        # 4. probability plot
        if self.prob:
            plt.clf()

            prob = m.predict(xy)[0]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(self.X_train.T[0], self.X_train.T[1], self.y_train)
            ax.plot_surface(x, y, prob.reshape(self.n_points_per_axis, self.n_points_per_axis), cmap = 'inferno', alpha = 0.4)

            plt.xlabel("x")
            plt.ylabel("y")
            
            plt.savefig("./Results/" + exp_name + "/score_plots/" + score_name + "/" + str(iteration) + "prob.png")

        # 5. countour plot of score-function
        if self.contour:
            plt.clf()
            z = z.reshape(points.shape[1], -1)

            # divide 1 and 0 classes
            X1 = self.X_train[np.nonzero(self.y_train)[0]]
            X0 = self.X_train[np.where(self.y_train == 0)[0]]

            # plot X_train and recently chosen points
            plt.plot(X1.T[1], X1.T[0], 'bo', alpha = 0.1)
            plt.plot(X0.T[1], X0.T[0], 'ro', alpha = 0.1)

            plt.plot(X_recent.T[0], X_recent.T[1], "k*")

            cmap = plt.cm.get_cmap("winter", 15)

            # plot contour of score function
            cs = plt.contour(axis, axis, z)
            fig.colorbar(cs)
            
            plt.xlabel("x")
            plt.ylabel("y")
            
            plt.savefig("./Results/" + exp_name + "/score_plots/" + score_name + "/" + str(iteration) + "contour.png")
        
        plt.close("all")
        
class HTRU_2(Data):

    def __init__(self, total_n, start_n, end_n, test_n):
        super(HTRU_2, self).__init__(0, start_n, end_n, test_n, 8)
    
    def generate_data(self, draw_data, exp_name):
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
                                                        test_size = self.test_n)

        return self.U, self.X_train, self.y_U, self.y_train, self.X_test, self.y_test
    
class Haberman(Data):

    def __init__(self, total_n, start_n, end_n, test_n):
        super(Haberman, self).__init__(0, start_n, end_n, test_n, 3)
    
    def generate_data(self, draw_data, exp_name):
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
                                                        test_size = self.test_n)
        self.y_U, self.y_train, self.y_test = self.y_U - 1, self.y_train - 1, self.y_test - 1

        return self.U, self.X_train, self.y_U, self.y_train, self.X_test, self.y_test    
    
class Skin(Data):

    def __init__(self, total_n, start_n, end_n, test_n):
        super(Skin, self).__init__(total_n, start_n, end_n, test_n, 3)
    
    def generate_data(self, draw_data, exp_name):
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
                                                        test_size = self.test_n)
        self.y_U, self.y_train, self.y_test = self.y_U - 1, self.y_train - 1, self.y_test - 1

        return self.U, self.X_train, self.y_U, self.y_train, self.X_test, self.y_test 