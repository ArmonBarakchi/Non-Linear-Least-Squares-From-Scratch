
import numpy as np


#derivate of tanh(x)
def deriv_tanh(x):
    e = np.exp(1)
    return (4 * e ** (2 * x))/((e ** (2 * x) + 1) ** 2)

def f_w(x,w):
    x1, x2, x3 = x
    w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16 = w
    return (w1*np.tanh(w2*x1 + w3*x2 + w4*x3 + w5) + w6*np.tanh(w7*x1 + w8*x2 + w9*x3 + w10) +
            w11*np.tanh(w12*x1 + w13*x2 + w14*x3 + w15) + w16)
def f_w_vec(x,w):
    y = np.empty((0,))
    for x_n in x:
        y = np.append(y, f_w(x_n,w))
    return y

#gradient of f_w(x) with respect to w
def gradient_fw (x, w):
    x_1, x_2, x_3 = x
    w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16 = w
    return np.array(
        [np.tanh(w_2 * x_1 + w_3 * x_2 + w_4 * x_3 + w_5), w_1 * deriv_tanh(w_2 * x_1 + w_3 * x_2 + w_4 * x_3 + w_5) * x_1,
         w_1 * deriv_tanh(w_2 * x_1 + w_3 * x_2 + w_4 * x_3 + w_5) * x_2,
         w_1 * deriv_tanh(w_2 * x_1 + w_3 * x_2 + w_4 * x_3 + w_5) * x_3,
         w_1 * deriv_tanh(w_2 * x_1 + w_3 * x_2 + w_4 * x_3 + w_5), np.tanh(w_7 * x_1 + w_8 * x_2 + w_9 * x_3 + w_10),
         w_6 * deriv_tanh(w_7 * x_1 + w_8 * x_2 + w_9 * x_3 + w_10) * x_1,
         w_6 * deriv_tanh(w_7 * x_1 + w_8 * x_2 + w_9 * x_3 + w_10) * x_2,
         w_6 * deriv_tanh(w_7 * x_1 + w_8 * x_2 + w_9 * x_3 + w_10) * x_3,
         w_6 * deriv_tanh(w_7 * x_1 + w_8 * x_2 + w_9 * x_3 + w_10), np.tanh(w_12 * x_1 + w_13 * x_2 + w_14 * x_3 + w_15),
         w_11 * deriv_tanh(w_12 * x_1 + w_13 * x_2 + w_14 * x_3 + w_15) * x_1,
         w_11 * deriv_tanh(w_12 * x_1 + w_13 * x_2 + w_14 * x_3 + w_15) * x_2,
         w_11 * deriv_tanh(w_12 * x_1 + w_13 * x_2 + w_14 * x_3 + w_15) * x_3,
         w_11 * deriv_tanh(w_12 * x_1 + w_13 * x_2 + w_14 * x_3 + w_15), 1]
    )
# derivative matrix of r_n(w) for all x in training data
def dr_w_matrix(x, w):
    mat = np.empty((0, 16))
    for x_n in x:
        mat = np.vstack((mat, gradient_fw(x_n,w)))
    return mat
