import numpy as np
npzfile = np.load("data.npz")
print(npzfile.files)
#for i in npzfile.files:
#    print(npzfile[i], npzfile[i].shape)
X_train = npzfile['arr_0']
Y_train = npzfile['arr_1']
print(X_train.shape)
print(Y_train.shape)