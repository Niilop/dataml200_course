import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

import warnings
import matplotlib
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

# generate 5 data points and corresponding output y (0, 1)
np.random.seed(13)
x_h = np.random.normal(1.1,0.3,5)
x_e = np.random.normal(1.9,0.4,5)

y_h = np.zeros(x_h.shape)
y_h[:] = 0.0
y_e = np.zeros(x_e.shape)
y_e[:] = +1.0

# set training data in one vector
x_tr = np.concatenate((x_h,x_e))
y_tr = np.concatenate((y_h,y_e))

num_of_epochs = 100
learning_rates = [0.1, 0.25, 0.5, 0.75, 1]
all_MSE = []
for rate in learning_rates:
    # set initial weights as 0
    w0_t = 0
    w1_t = 0
    learning_rate = rate
    for e in range(num_of_epochs):
        for x_ind, x in enumerate(x_tr):

            yhat = expit(w1_t*x+w0_t)
            w1_t = w1_t - learning_rate*(-2*(y_tr[x_ind] - yhat)*(yhat)*(1-yhat)*x)
            w0_t = w0_t - learning_rate*(-2*(y_tr[x_ind] - yhat)*(yhat)*(1-yhat)*1)

            y_pred = expit(w1_t*x_tr+w0_t)
            MSE = np.sum((y_tr-y_pred)**2)/(len(y_tr))

        if np.mod(e, 20) == 0 or e == 1:  # Plot after every 20th epoch
            y_pred = expit(w1_t * x_tr + w0_t)
            MSE = np.sum((y_tr - y_pred) ** 2) / (len(y_tr))

            plt.title(f'Epoch={e} w0={w0_t:.2f} w1={w1_t:.2f} MSE={MSE:.2f} LR={learning_rate:.2f}')
            plt.plot(x_h, y_h, 'co', label="hobbit")
            plt.plot(x_e, y_e, 'mo', label="elf")
            x = np.linspace(0.0, +5.0, 50)
            plt.plot(x, expit(w1_t * x + w0_t), 'b-', label='y=logsig(w1x+w0)')
            plt.plot([0.5, 5.0], [0.5, 0.5], 'k--', label='y=0 (class boundary)')
            plt.xlabel('height [m]')
            plt.legend()
            plt.show()
    plt.clf()
    all_MSE.append(MSE)
    np.set_printoptions(precision=2)

    print(f'LR={learning_rate:.2f} MSE={MSE:.2f} True values y={y_tr} and predicted values y_pred={y_pred}')

