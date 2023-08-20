import numpy as np

print("Reading data... Done!")
X_train = np.loadtxt("X_train.dat")
X_test = np.loadtxt("X_test.dat")

Y_train = np.loadtxt("Y_train.dat")
Y_test = np.loadtxt("Y_test.dat")


n = len(Y_train)
meanY = sum(Y_train)/n

n = len(Y_test)
MAE = sum(meanY - Y_test)/n

def one_nn (X_test, X_train, Y_train):
    predicts = []
    # calculates each distance for each X test value
    dist = np.sqrt(np.sum((X_test[:, np.newaxis, :] - X_train)**2, axis=-1))
    for d in dist:
        # picks the index of the lowest distance
        min = np.argmin(d)
        predicts.append((Y_train[min]))
    return predicts
Y_pred = one_nn (X_test, X_train, Y_train)

print("Shape of training data", X_train.shape)
print("Computing 1-nn regression")

print("   Baseline accuracy (MAE):", MAE)

n = len(Y_test)
MAE = sum(Y_pred - Y_test)/n

print("   1NN regr. accuracy (MAE):", MAE)