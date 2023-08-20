import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

noise_factor = 0.2

train_images_noisy = train_images + noise_factor * tf.random.normal(shape=train_images.shape)
test_images_noisy = test_images + noise_factor * tf.random.normal(shape=test_images.shape)

train_images_noisy = tf.clip_by_value(train_images_noisy,clip_value_min=0., clip_value_max=1.)
test_images_noisy = tf.clip_by_value(test_images_noisy,clip_value_min=0., clip_value_max=1.)


CNN_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

train_labels_onehot = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels_onehot = tf.keras.utils.to_categorical(test_labels, num_classes=10)

CNN_model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy,metrics='accuracy')

CNN_model.fit(train_images, train_labels_onehot,
              epochs=5,
              shuffle=False)

print('\nClean trained CNN test accuracy for clean images:')
test_loss, test_acc = CNN_model.evaluate(test_images, test_labels_onehot)
print()

print('Clean trained CNN test accuracy for noisy images:')
test_loss, test_acc = CNN_model.evaluate(test_images_noisy, test_labels_onehot)
print()

encoder = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
    #tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)

    #tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    #tf.keras.layers.Dense(64,activation='relu')
])

decoder = tf.keras.models.Sequential([
    #tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
    tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')

    #tf.keras.layers.Dense(784,activation='sigmoid'),
    #tf.keras.layers.Reshape((28,28, 1))
])

autoencoder = tf.keras.models.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(),metrics="accuracy")

train_labels_onehot = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels_onehot = tf.keras.utils.to_categorical(test_labels, num_classes=10)

autoencoder.fit(train_images_noisy, train_images,
                epochs=5,
                shuffle=True,
                validation_data=(test_images_noisy, test_images))


encoded_imgs = encoder(test_images_noisy).numpy()
decoded_imgs = decoder(encoded_imgs).numpy()

os.environ['KMP_DUPLICATE_LIB_OK']='True'
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(test_images_noisy[i])
  plt.title("noisy images")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  ax = plt.subplot(2, n, i + n + 1)
  plt.imshow(tf.squeeze(decoded_imgs[i]))
  plt.title('Decoded')
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.show()

print('\nClean trained CNN test accuracy for denoised images:')
test_loss, test_acc = CNN_model.evaluate(decoded_imgs, test_labels_onehot)
print()

CNN_model.fit(train_images_noisy, train_labels_onehot,epochs=5, shuffle=False)
print('\nNoisy trained CNN test accuracy for noisy images:')
test_loss, test_acc = CNN_model.evaluate(test_images_noisy, test_labels_onehot)