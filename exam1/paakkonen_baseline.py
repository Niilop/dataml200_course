import matplotlib.pyplot as plt
import numpy as np

print("Reading data... Done!")
X_train = np.loadtxt("X_train.dat")
X_test = np.loadtxt("X_test.dat")

Y_train = np.loadtxt("Y_train.dat")
Y_test = np.loadtxt("Y_test.dat")

print("Computing baseline regression.")
n = len(Y_train)
meanY = sum(Y_train)/n

n = len(Y_test)
MAE = sum(meanY - Y_test)/n

print("Baseline accuracy (MAE):", MAE)
