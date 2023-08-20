import matplotlib.pyplot as plt
import numpy as np

print("Reading data... Done!")
X_train = np.loadtxt("X_train.dat")
X_test = np.loadtxt("X_test.dat")

Y_train = np.loadtxt("Y_train.dat")
Y_test = np.loadtxt("Y_test.dat")

n = len(Y_train)
meanY = sum(Y_train)/n

n = len(Y_test)
MAE = sum(Y_test - meanY)/n
print("Baseline accuracy (MAE):", MAE)
