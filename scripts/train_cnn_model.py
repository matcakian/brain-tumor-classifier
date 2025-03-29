import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
	# create, train, and save the model
	train_npz = np.load("./data/train.npz")
	test_npz = np.load("./data/test.npz")

	x_train, x_val, y_train, y_val = train_test_split(train_npz["images"], train_npz["labels"],
		test_size=0.2, random_state=58)

	x_test = test_npz["images"]
	y_test = test_npz["labels"]

	cnn_model = models.Sequential([
		layers.Conv2D(32, (3, 3), activation="relu", input_shape=(256, 256, 3)),
		layers.MaxPooling2D((2, 2)),

		layers.Conv2D(64, (3, 3), activation="relu"),
		layers.MaxPooling2D((2, 2)),

		layers.Conv2D(128, (3, 3), activation="relu"),
		layers.MaxPooling2D((2, 2)),

		layers.Flatten(),
		layers.Dense(128, activation="relu"),
		layers.Dense(4, activation=None)
	])

	cnn_model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(from_logits=True), 
		metrics=["accuracy"])

	cnn_model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val), batch_size=32)

	cnn_model.save("./models/cnn_model.h5")

