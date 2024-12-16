import numpy as np
import functions

#exception class used to tell when the algorithm has convered
class Converged(Exception):
    def __init__(self, message = "The algorithm has converged."):
        self.message = message
        super().__init__(self.message)

#generate N training points x in R^3 between -gamma and gamma and y = function(x)
def random_data(N = 500, gamma = 1, function = lambda x_1, x_2, x_3: x_1*x_2 + x_3):
    training_x = []
    training_y = np.empty((0,))
    for i in range(N):
        x_1 = np.random.uniform(-gamma, gamma)
        x_2 = np.random.uniform(-gamma, gamma)
        x_3 = np.random.uniform(-gamma, gamma)
        training_x.append([x_1, x_2, x_3])
        training_y = np.append(training_y, function(x_1, x_2, x_3))
    return np.array(training_x), training_y

#find next iteration of w given the current iteration w_k, training data, x, y, lambda.
# Also returns new lambda
# Raises Converged of algo converged
def next_iterate(last_iterate, x, y, l, num_iter, maxIterations, stop = .01):
    w_k = last_iterate
    dr = functions.dr_w_matrix(x, w_k)
    f_w = functions.f_w_vec(x, w_k)
    r = f_w - y
    w_k_1 = w_k - np.linalg.inv(dr.T @ dr + l*np.identity(16)) @ dr.T @ r # next iterate
    r_1 = functions.f_w_vec(x, w_k_1) - y
    if np.inner(r_1.T, r_1) < stop or num_iter >= maxIterations: # check if algorithm has ended
        raise Converged
    elif np.linalg.norm(r_1)**2 < np.linalg.norm(r)**2: # check if next iterate is better
        return w_k_1, .8*l
    else:
        return w_k, 2*l


# training loss with respect to weights w, given training data (x,y) and l (lambda) for the 
# current iteration
def training_loss(w, x, y, l):
    return np.inner((functions.f_w_vec(x,w) - y), (functions.f_w_vec(x,w) - y)) + l*np.linalg.norm(w)**2

#function used to run the algorithm, returns weights and loss
def Leven_Marq(training_x, training_y, l=.00001, initial=np.random.normal(0, 1, 16)):
    loss = []
    num_iter = 0
    w = initial
    best_w = w
    best_loss = training_loss(w, training_x, training_y, l)
    while True:
        try:
            w, l = next_iterate(w, training_x, training_y, l, num_iter, maxIterations=200)
            loss.append(training_loss(w, training_x, training_y, l))
            if loss[-1] < best_loss:
                best_w = w
                best_loss = loss[-1]
        except Converged:
            break
        except KeyboardInterrupt:
            break
        num_iter += 1

    return (best_w, loss)

def squared_error(x, y, w):
    return np.inner((functions.f_w_vec(x,w) - y), (functions.f_w_vec(x,w) - y))

def generate_noisy_data(num, gamma, epsilon, function):
    x = []
    y = []
    for i in range(num):
        x_1 = np.random.uniform(-gamma, gamma)
        x_2 = np.random.uniform(-gamma, gamma)
        x_3 = np.random.uniform(-gamma, gamma)
        x.append([x_1, x_2, x_3])
        y.append(function(x_1, x_2, x_3) + np.random.uniform(-epsilon, epsilon))
    return np.array(x), np.array(y)