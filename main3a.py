import numpy as np
import helpers
import matplotlib.pyplot as plt

#create random data
x,y = helpers.random_data(N = 500, gamma = 1, function = lambda x_1, x_2, x_3: x_1*x_2 + x_3)

#generate weights/losses for each initialization
weights_2, loss2 = helpers.Leven_Marq(training_x = x, training_y = y, l = 0.01)
weights_2_05, loss2_05 = helpers.Leven_Marq(training_x = x, training_y =y, l = 0.01, initial = np.random.normal(0, .5, 16))
weights_2_2, loss2_2 = helpers.Leven_Marq(training_x = x, training_y =y, l = 0.01, initial = np.random.normal(0, 2, 16))


weights_4, loss4 = helpers.Leven_Marq(training_x = x, training_y = y, l =0.0001)
weights_4_05, loss4_05 = helpers.Leven_Marq(training_x = x, training_y =y, l =0.0001, initial = np.random.normal(0, .5, 16))
weights_4_2, loss4_2 = helpers.Leven_Marq(training_x = x, training_y =y, l =0.0001, initial = np.random.normal(0, 2, 16))


weights_6, loss6 = helpers.Leven_Marq(training_x = x, training_y = y, l = 0.000001)
weights_6_05, loss6_05 = helpers.Leven_Marq(training_x = x, training_y =y, l = 0.000001, initial = np.random.normal(0, .5, 16))
weights_6_2, loss6_2 = helpers.Leven_Marq(training_x = x, training_y =y, l = 0.000001, initial = np.random.normal(0, 2, 16))

#plot the results
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(loss2, label = "N(0,1)")
plt.plot(loss2_05, label = "N(0,.5)")
plt.plot(loss2_2, label = "N(0,2)")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title(r"$\lambda = 10^{-2}$")
plt.legend()
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.subplot(2,2,2)
plt.plot(loss4, label = "N(0,1)")
plt.plot(loss4_05, label = "N(0,.5)")
plt.plot(loss4_2, label = "N(0,2)")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title(r"$\lambda = 10^{-4}$")
plt.legend()
plt.subplots_adjust(wspace=0.4, hspace=0.6)

plt.subplot(2,2,3)
plt.plot(loss6, label = "N(0,1)")
plt.plot(loss6_05, label = "N(0,.5)")
plt.plot(loss6_2, label = "N(0,2)")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title(r"$\lambda = 10^{-6}$")
plt.legend()
plt.subplots_adjust(wspace=0.4, hspace=0.6)
plt.show()

#print some data about the results
print("Different lambdas:")
print("Average training loss for lambda = 10^-2: ", round((loss2[-1] + loss2_05[-1] + loss2_2[-1])/3,3), ". Reached in an average of ", round((len(loss2) + len(loss2_05) + len(loss2_2))/3,3), " iterations.")
print("Average training loss for lambda = 10^-4: ", round((loss4[-1] + loss4_05[-1] + loss4_2[-1])/3,3), ". Reached in an average of ", round((len(loss4) + len(loss4_05) + len(loss4_2))/3,3), " iterations.")
print("Average training loss for lambda = 10^-6: ", round((loss6[-1] + loss6_05[-1] + loss6_2[-1])/3,3), ". Reached in an average of ", round((len(loss6) + len(loss6_05) + len(loss6_2))/3,3), " iterations.")
