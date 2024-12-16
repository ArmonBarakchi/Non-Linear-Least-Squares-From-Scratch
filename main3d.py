import helpers
from matplotlib import pyplot as plt
import numpy as np


def calc_rand_test(num, gammas, function):
    x_test = []
    y_test = []
    for gamma in gammas:
        x, y = helpers.random_data(N = num, gamma = gamma, function = function)
        x_test.append(x)
        y_test.append(y)
    return np.array(x_test), np.array(y_test)

x_noise01, y_noise01 = helpers.generate_noisy_data(num = 500, gamma = 1, epsilon = 0.1, function = lambda x_1, x_2, x_3: x_1*x_2 + x_3)
x_noise02, y_noise02 = helpers.generate_noisy_data(num = 500, gamma = 1, epsilon = 0.2, function = lambda x_1, x_2, x_3: x_1*x_2 + x_3)
x_noise03, y_noise03 = helpers.generate_noisy_data(num = 500, gamma = 1, epsilon = 0.3, function = lambda x_1, x_2, x_3: x_1*x_2 + x_3)


#epsilon = 0.1 for l = 10^-2, 10^-4, 10^-6, and for N(0,1) and N(0,.5)
#naming format is weights_noise(epsilon level)_(l)_N(0,1/0.5)
weights_noise01_001_1, loss_noise01_001_1 = helpers.Leven_Marq(training_x = x_noise01, training_y = y_noise01, l = 0.01, initial = np.random.normal(0,1,16))
weights_noise01_001_05, loss_noise01_001_05 = helpers.Leven_Marq(training_x = x_noise01, training_y = y_noise01, l =0.01, initial = np.random.normal(0, 0.5, 16))

weights_noise01_00001_1, loss_noise01_00001_1 = helpers.Leven_Marq(training_x = x_noise01, training_y = y_noise01, l = 0.0001, initial = np.random.normal(0,1,16))
weights_noise01_00001_05, loss_noise01_00001_05 = helpers.Leven_Marq(training_x = x_noise01, training_y = y_noise01, l =0.0001, initial = np.random.normal(0, 0.5, 16))

weights_noise01_0000001_1, loss_noise01_0000001_1 = helpers.Leven_Marq(training_x = x_noise01, training_y = y_noise01, l = 0.000001, initial = np.random.normal(0,1,16))
weights_noise01_0000001_05, loss_noise01_0000001_05 = helpers.Leven_Marq(training_x = x_noise01, training_y = y_noise01, l =0.000001, initial = np.random.normal(0, 0.5, 16))

#epsilon = 0.2 for l = 10^-2, 10^-4, 10^-6, and for N(0,1) and N(0,.5)

weights_noise02_001_1, loss_noise02_001_1 = helpers.Leven_Marq(training_x = x_noise02, training_y = y_noise01, l = 0.01, initial = np.random.normal(0,1,16))
weights_noise02_001_05, loss_noise02_001_05 = helpers.Leven_Marq(training_x = x_noise02, training_y = y_noise01, l =0.01, initial = np.random.normal(0, 0.5, 16))

weights_noise02_00001_1, loss_noise02_00001_1 = helpers.Leven_Marq(training_x = x_noise02, training_y = y_noise01, l = 0.0001, initial = np.random.normal(0,1,16))
weights_noise02_00001_05, loss_noise02_00001_05 = helpers.Leven_Marq(training_x = x_noise02, training_y = y_noise01, l =0.0001, initial = np.random.normal(0, 0.5, 16))

weights_noise02_0000001_1, loss_noise02_0000001_1 = helpers.Leven_Marq(training_x = x_noise02, training_y = y_noise01, l = 0.000001, initial = np.random.normal(0,1,16))
weights_noise02_0000001_05, loss_noise02_0000001_05 = helpers.Leven_Marq(training_x = x_noise02, training_y = y_noise01, l =0.000001, initial = np.random.normal(0, 0.5, 16))

#epsilon = 0.3 for l = 10^-2, 10^-4, 10^-6, and for N(0,1) and N(0,.5)

weights_noise03_001_1, loss_noise03_001_1 = helpers.Leven_Marq(training_x = x_noise03, training_y = y_noise01, l = 0.01, initial = np.random.normal(0,1,16))
weights_noise03_001_05, loss_noise03_001_05 = helpers.Leven_Marq(training_x = x_noise03, training_y = y_noise01, l =0.01, initial = np.random.normal(0, 0.5, 16))

weights_noise03_00001_1, loss_noise03_00001_1 = helpers.Leven_Marq(training_x = x_noise03, training_y = y_noise01, l = 0.0001, initial = np.random.normal(0,1,16))
weights_noise03_00001_05, loss_noise03_00001_05 = helpers.Leven_Marq(training_x = x_noise03, training_y = y_noise01, l =0.0001, initial = np.random.normal(0, 0.5, 16))

weights_noise03_0000001_1, loss_noise03_0000001_1 = helpers.Leven_Marq(training_x = x_noise03, training_y = y_noise01, l = 0.000001, initial = np.random.normal(0,1,16))
weights_noise03_0000001_05, loss_noise03_0000001_05 = helpers.Leven_Marq(training_x = x_noise03, training_y = y_noise01, l =0.000001, initial = np.random.normal(0, 0.5, 16))

#epsilon = 0.1
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(loss_noise01_001_1, label = "N(0,1)")
plt.plot(loss_noise01_001_05, label = "N(0,.5)")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title(r"$\lambda = 10^{-2}$")
plt.legend()
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.subplot(2,2,2)
plt.plot(loss_noise01_00001_1, label = "N(0,1)")
plt.plot(loss_noise01_00001_05, label = "N(0,.5)")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title(r"$\lambda = 10^{-4}$")
plt.legend()
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.subplot(2,2,3)
plt.plot(loss_noise01_0000001_1, label = "N(0,1)")
plt.plot(loss_noise01_0000001_05, label = "N(0,.5)")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title(r"$\lambda = 10^{-6}$")
plt.legend()
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.suptitle(r"Training loss for various values of $\lambda$ and $\epsilon = .1$", fontsize=16);

#epsilon = 0.2
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(loss_noise02_001_1, label = "N(0,1)")
plt.plot(loss_noise02_001_05, label = "N(0,.5)")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title(r"$\lambda = 10^{-2}$")
plt.legend()
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.subplot(2,2,2)
plt.plot(loss_noise02_00001_1, label = "N(0,1)")
plt.plot(loss_noise02_00001_05, label = "N(0,.5)")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title(r"$\lambda = 10^{-3}$")
plt.legend()
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.subplot(2,2,3)
plt.plot(loss_noise02_0000001_1, label = "N(0,1)")
plt.plot(loss_noise02_0000001_05, label = "N(0,.5)")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title(r"$\lambda = 10^{-4}$")
plt.legend()
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.suptitle(r"Training loss for various values of $\lambda$ and $\epsilon = .2$", fontsize=16);

#epsilon = 0.3

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(loss_noise03_001_1, label = "N(0,1)")
plt.plot(loss_noise03_001_05, label = "N(0,.5)")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title(r"$\lambda = 10^{-2}$")
plt.legend()
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.subplot(2,2,2)
plt.plot(loss_noise03_00001_1, label = "N(0,1)")
plt.plot(loss_noise03_00001_05, label = "N(0,.5)")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title(r"$\lambda = 10^{-3}$")
plt.legend()
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.subplot(2,2,3)
plt.plot(loss_noise03_0000001_1, label = "N(0,1)")
plt.plot(loss_noise03_0000001_05, label = "N(0,.5)")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title(r"$\lambda = 10^{-4}$")
plt.legend()
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.suptitle(r"Training loss for various values of $\lambda$ and $\epsilon = .3$", fontsize=16);
plt.show()

print("Different epsilons:")
print("Average training loss for epsilon = 0.1: ", round((loss_noise01_001_1[-1] + loss_noise01_001_05[-1] + loss_noise01_00001_1[-1] + loss_noise01_00001_05[-1] + loss_noise01_0000001_1[-1] + loss_noise01_0000001_05[-1])/6,2), ". Reached in an average of ", round((len(loss_noise01_001_1) + len(loss_noise01_001_05) + len(loss_noise01_00001_1) + len(loss_noise01_00001_05) + len(loss_noise01_0000001_1) + len(loss_noise01_0000001_05))/6,2), " iterations.")
print("Average training loss for epsilon = 0.2: ", round((loss_noise02_001_1[-1] + loss_noise02_001_05[-1] + loss_noise02_00001_1[-1] + loss_noise02_00001_05[-1] + loss_noise02_0000001_1[-1] + loss_noise02_0000001_05[-1])/6,2), ". Reached in an average of ", round((len(loss_noise02_001_1) + len(loss_noise02_001_05) + len(loss_noise02_00001_1) + len(loss_noise02_00001_05) + len(loss_noise02_0000001_1) + len(loss_noise02_0000001_05))/6,2), " iterations.")
print("Average training loss for epsilon = 0.3: ", round((loss_noise03_001_1[-1] + loss_noise03_001_05[-1] + loss_noise03_00001_1[-1] + loss_noise03_00001_05[-1] + loss_noise03_0000001_1[-1] + loss_noise03_0000001_05[-1])/6,2), ". Reached in an average of ", round((len(loss_noise03_001_1) + len(loss_noise03_001_05) + len(loss_noise03_00001_1) + len(loss_noise03_00001_05) + len(loss_noise03_0000001_1) + len(loss_noise03_0000001_05))/6,2), " iterations.")

rtest_x, rtest_y = calc_rand_test(100, [.5, .75, 1, 1.25, 1.5], lambda x_1, x_2, x_3: x_1*x_2 +x_3)
# epsilon = 0.1
rweights_noise01_05 = [weights_noise01_001_05, weights_noise01_00001_05, weights_noise01_0000001_05]
rweights_noise01_01 = [weights_noise01_001_1, weights_noise01_00001_1, weights_noise01_0000001_1]
r105_error = []
r101_error = []

# epsilon = .2
rweights_noise02_05 = [weights_noise02_001_05, weights_noise02_00001_05, weights_noise02_0000001_05]
rweights_noise02_01 = [weights_noise02_001_1, weights_noise02_00001_1, weights_noise02_0000001_1]
r205_error = []
r201_error = []

# epsilon = .3
rweights_noise03_05 = [weights_noise03_001_05, weights_noise03_00001_05, weights_noise03_0000001_05]
rweights_noise03_01 = [weights_noise03_001_1, weights_noise03_00001_1, weights_noise03_0000001_1]
r305_error = []
r301_error = []

for i in range(3):
    r105 = []
    r101 = []
    r205 = []
    r201 = []
    r305 = []
    r301 = []
    for j in range(5):
        r105.append(helpers.squared_error(rtest_x[j], rtest_y[j], rweights_noise01_05[i]))
        r101.append(helpers.squared_error(rtest_x[j], rtest_y[j], rweights_noise01_01[i]))
        r205.append(helpers.squared_error(rtest_x[j], rtest_y[j], rweights_noise02_05[i]))
        r201.append(helpers.squared_error(rtest_x[j], rtest_y[j], rweights_noise02_01[i]))
        r305.append(helpers.squared_error(rtest_x[j], rtest_y[j], rweights_noise03_05[i]))
        r301.append(helpers.squared_error(rtest_x[j], rtest_y[j], rweights_noise03_01[i]))
    r105_error.append(r105)
    r101_error.append(r101)
    r205_error.append(r205)
    r201_error.append(r201)
    r305_error.append(r305)
    r301_error.append(r301)

columns = [r"$\Gamma_T = .5$", r"$\Gamma_T = .75$", r"$\Gamma_T = 1$", r"$\Gamma_T = 1.25$", r"$\Gamma_T = 1.5$"]
rows = [r"$\lambda = 10^{-2}$", r"$\lambda = 10^{-4}$", r"$\lambda = 10^{-6}$"]
r105_error = np.array(r105_error)
r101_error = np.array(r101_error)
r205_error = np.array(r205_error)
r201_error = np.array(r201_error)
r305_error = np.array(r305_error)
r301_error = np.array(r301_error)
r105_error = np.round(r105_error, 5)
r101_error = np.round(r101_error, 5)
r205_error = np.round(r205_error, 5)
r201_error = np.round(r201_error, 5)
r305_error = np.round(r305_error, 5)
r301_error = np.round(r301_error, 5)
fig1, ax1 = plt.subplots(1, 2, figsize=(15, 5))

ax1[0].axis('tight')
ax1[0].axis('off')
ax1[0].table(cellText=r105_error, rowLabels=rows, colLabels=columns, loc='center')
ax1[0].set_title("Weights initialized according to $N(0,.5)$", y= .7)

ax1[1].axis('tight')
ax1[1].axis('off')
ax1[1].table(cellText=r101_error, rowLabels=rows, colLabels=columns, loc='center')
ax1[1].set_title(r"Weights initialized according to $N(0,1)$", y= .7)

plt.tight_layout()
fig1.suptitle(r"Sum of squared error for each initialization of the algorithm and $\epsilon$ = 0.1", fontsize=16)

fig2, ax2 = plt.subplots(1, 2, figsize=(15, 5))

ax2[0].axis('tight')
ax2[0].axis('off')
ax2[0].table(cellText=r205_error, rowLabels=rows, colLabels=columns, loc='center')
ax2[0].set_title("Weights initialized according to $N(0,.5)$", y= .7)

ax2[1].axis('tight')
ax2[1].axis('off')
ax2[1].table(cellText=r201_error, rowLabels=rows, colLabels=columns, loc='center')
ax2[1].set_title("Weights initialized according to $N(0,1)$", y= .7)

plt.tight_layout()
fig2.suptitle(r"Sum of squared error for each initialization of the algorithm and $\epsilon$ = 0.2:", fontsize=16)

fig3, ax3 = plt.subplots(1, 2, figsize=(15, 5))

ax3[0].axis('tight')
ax3[0].axis('off')
ax3[0].table(cellText=r305_error, rowLabels=rows, colLabels=columns, loc='center')
ax3[0].set_title("Weights initialized according to $N(0,.5)$", y= .7)

ax3[1].axis('tight')
ax3[1].axis('off')
ax3[1].table(cellText=r301_error, rowLabels=rows, colLabels=columns, loc='center')
ax3[1].set_title("Weights initialized according to $N(0,1)$", y= .7)

plt.tight_layout()
fig3.suptitle(r"Sum of squared error for each initialization of the algorithm and $\epsilon$ = 0.3:", fontsize=16)
plt.show()