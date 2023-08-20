import tensorflow as tf
from skimage.io import imread_collection
import numpy as np
from sklearn.model_selection import train_test_split

file1 = 'GTSRB_subset_2/class1/*.jpg'
file2 = 'GTSRB_subset_2/class2/*.jpg'

images1 = np.array(imread_collection(file1))
images2 = np.array(imread_collection(file2))

# Create labels for the image data using the length of the image arrays, label1 = 0, label2 = 1
labels1 = np.zeros(len(images1))
labels2 = np.ones(len(images2))

# X containing all the image data with shape (660, 64, 64, 3)
X = np.vstack((images1, images2))

# y containing all the corresponding labels of X
y = np.concatenate([labels1,labels2])

# X flatten the X to (660, 12288)
X_flat = X.reshape((660, -1))

# Normalize X
X = X/255


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(10, (3,3),input_shape=(64, 64, 3), activation='relu',strides=(2,2)),
    #tf.keras.layers.Conv2D(10, (3,3),input_shape=(64, 64, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    #tf.keras.layers.Conv2D(10, (3,3), activation='relu'),
    # without strides, it reaches nearly 1 accuracy
    tf.keras.layers.Conv2D(10, (3,3), activation='relu',strides=(2,2)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation='sigmoid')
])
model.summary()

model.compile(optimizer='SGD',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics="accuracy")


y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=2)

# Train the model for 20 epochs
model.fit(X_train, y_train_onehot,batch_size=32 ,epochs=5, shuffle=False)

test_loss, test_acc = model.evaluate(X_test, y_test_onehot)
print('Test accuracy:', test_acc)
