import tensorflow as tf
import numpy as np
import os
import sys
from prepare_dataset import load_image, preprocess_image


if __name__ == "__main__":
	# Process invalid input
	if len(sys.argv) != 3:
		print("Usage: python3 scripts/client.py path_to_model.h5 path_to_image.jpg")
		sys.exit(1)

	model_path, image_path = sys.argv[1], sys.argv[2]

	if not os.path.exists(model_path):
		print(f"Model not found at: {model_path}")
		sys.exit(1)

	if not os.path.exists(image_path):
		print(f"Image not found at {image_path}")
		sys.exit(1)

	labels = ["glioma", "meningioma", "notumor", "pituitary"]

	model = tf.keras.models.load_model(model_path)
	image = preprocess_image(load_image(image_path))
	image = np.expand_dims(image, axis=0)

	logits = model.predict(image)
	probabilities = tf.nn.softmax(logits[0]).numpy()
	predicted_class = np.argmax(probabilities)
	confidence = probabilities[predicted_class]
	label = labels[predicted_class]

	for label_id, probability in sorted(enumerate(probabilities), key=lambda tmp: tmp[1], 
		reverse=True):
		print(f"{labels[label_id]}: {probability * 100:.2f}%")
		print(logits[0][label_id])

