import os
import cv2
import numpy as np


def preprocess_image(image, image_size=(256, 256)):
	image = cv2.resize(image, image_size)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.astype("float32") / 255.0

	return image


def load_image(image_path):
	image = cv2.imread(image_path)
	
	return image


def load_dataset_from_dir(save_path, base_dir, labels, image_size=(256, 256)):
	image_list = []
	label_list = []

	for label_id, label in enumerate(labels):
		label_dir = os.path.join(base_dir, label)
		for filename in os.listdir(label_dir):
			filepath = os.path.join(label_dir, filename)
			if os.path.isfile(filepath) and filepath.lower().endswith((".jpg", ".png", ".jpeg")):
				image = preprocess_image(load_image(filepath), image_size)
				image_list.append(image)
				label_list.append(label_id)

	images = np.array(image_list, dtype="float32")
	labels = np.array(label_list, dtype="int32")

	np.savez(save_path, images=images, labels=labels)
	print(f"Saved {len(images)} images to '{save_path}'")


if __name__ == "__main__":
	# The directories of .jpg images grouped by the label
	train_dir = "./archive/Training"
	test_dir = "./archive/Testing"

	labels = ["glioma", "meningioma", "notumor", "pituitary"]

	load_dataset_from_dir("./data/train.npz", train_dir, labels)
	load_dataset_from_dir("./data/test.npz", test_dir, labels)
	
