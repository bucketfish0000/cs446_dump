import numpy as np
import hw1_utils as utils
import matplotlib.pyplot as plt

def empirical_risk_gradient(X,Y,w,n): 

    grad = np.dot(X.T, np.dot(X, w) - Y) / n
    return grad

def linear_gd(X,Y,lrate=0.1,num_iter=1000):
    X_b=np.c_[np.ones((X.shape[0], 1)), X]
    n, d = X_b.shape
    w = np.zeros(d)

    for i in range(num_iter):
        w -= lrate * empirical_risk_gradient(X_b,Y,w,n)
    return w

########

def linear_normal(X,Y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
    return w

def plot_linear():
    X,Y=utils.load_reg_data()
    w_1 = linear_gd(X,Y)
    w_2 =linear_normal(X,Y)
    plt.plot(X.T[0],Y,'o') 
    curve_x = [[0.01,3.99],[1,1]]
    curve_y_2 = [w_2 * i for i in curve_x]
    plt.plot(curve_x[0],curve_y_2[0])
    plt.savefig("5_3.png")
    return

if __name__ == "__main__":
    plot_linear()