import numpy as np
import functions
import helpers
import matplotlib.pyplot as plt

#generate test data based on gammas
def calc_rand_test(num, gammas, function):
    x_test = []
    y_test = []
    for gamma in gammas:
        x, y = helpers.random_data(N = num, gamma = gamma, function = function)
        x_test.append(x)
        y_test.append(y)
    return np.array(x_test), np.array(y_test)

#generate training data
x,y = helpers.random_data(N = 500, gamma = 1, function = lambda x_1, x_2, x_3: x_1*x_2 + x_3)

#get weights and losses for different initializations
weights_2, loss2 = helpers.Leven_Marq(training_x = x, training_y = y, l = 0.01)
weights_2_05, loss2_05 = helpers.Leven_Marq(training_x = x, training_y =y, l = 0.01, initial = np.random.normal(0, .5, 16))
weights_2_2, loss2_2 = helpers.Leven_Marq(training_x = x, training_y =y, l = 0.01, initial = np.random.normal(0, 2, 16))


weights_4, loss4 = helpers.Leven_Marq(training_x = x, training_y = y, l =0.0001)
weights_4_05, loss4_05 = helpers.Leven_Marq(training_x = x, training_y =y, l =0.0001, initial = np.random.normal(0, .5, 16))
weights_4_2, loss4_2 = helpers.Leven_Marq(training_x = x, training_y =y, l =0.0001, initial = np.random.normal(0, 2, 16))


weights_6, loss6 = helpers.Leven_Marq(training_x = x, training_y = y, l = 0.000001)
weights_6_05, loss6_05 = helpers.Leven_Marq(training_x = x, training_y =y, l = 0.000001, initial = np.random.normal(0, .5, 16))
weights_6_2, loss6_2 = helpers.Leven_Marq(training_x = x, training_y =y, l = 0.000001, initial = np.random.normal(0, 2, 16))

#generate random test data based on different values of gamma
test_x, test_y = calc_rand_test(100, [.25, .5, 1, 1.25, 1.5], lambda x_1, x_2, x_3: x_1*x_2 + x_3)
weights_05 = [weights_2_05, weights_4_05, weights_6_05] # weights for N(0,0.5)
weights_01 = [weights_2, weights_4, weights_6]# weights for N(0,1)
weights_02 = [weights_2_2, weights_4_2, weights_6_2]# weights for N(0,2)
error_05 = []
error_01 = []
error_02 = []
for i in range(3):
    e05 = []
    e01 = []
    e02 = []
    for j in range(5):
        e05.append(helpers.squared_error(test_x[j], test_y[j], weights_05[i]))
        e01.append(helpers.squared_error(test_x[j], test_y[j], weights_01[i]))
        e02.append(helpers.squared_error(test_x[j], test_y[j], weights_02[i]))
    error_05.append(e05)
    error_01.append(e01)
    error_02.append(e02)

#create table for squared errors
columns = [r"$\Gamma_T = .25$", r"$\Gamma_T = .5$", r"$\Gamma_T = 1$", r"$\Gamma_T = 1.25$", r"$\Gamma_T = 1.5$"]
rows = [r"$\lambda = 10^{-2}$", r"$\lambda = 10^{-4}$", r"$\lambda = 10^{-6}$"]
error_05 = np.array(error_05)
error_01 = np.array(error_01)
error_02 = np.array(error_02)
error_05 = np.round(error_05, 5)
error_01 = np.round(error_01, 5)
error_02 = np.round(error_02, 5)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].axis('tight')
ax[0].axis('off')
ax[0].table(cellText=error_05, rowLabels=rows, colLabels=columns, loc='center')
ax[0].set_title("Weights initialized according to $N(0,.5)$", y=.7)

ax[1].axis('tight')
ax[1].axis('off')
ax[1].table(cellText=error_01, rowLabels=rows, colLabels=columns, loc='center')
ax[1].set_title("Weights initialized according to $N(0,1)$", y=.7)

ax[2].axis('tight')
ax[2].axis('off')
ax[2].table(cellText=error_02, rowLabels=rows, colLabels=columns, loc='center')
ax[2].set_title("Weights initialized according to $N(0,2)$", y=.7)

plt.tight_layout()
fig.suptitle("Sum of squared error for each initialization of the algorithm:", fontsize=16)
plt.show()