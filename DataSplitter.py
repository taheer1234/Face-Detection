import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

#Get absolute paths to different directories.
data_dir = "data"
abspath_data_dir = os.path.join(os.path.abspath(data_dir))
abspath_images_dir = os.path.join(abspath_data_dir, "images")
abspath_labels_dir = os.path.join(abspath_data_dir, "labels")

#If the images path is not empty run the following.
if os.listdir(abspath_images_dir):
    #Get the addresses of all the image files.
    image_paths = [os.path.join(abspath_images_dir, file_path) for file_path in os.listdir(abspath_images_dir)]

    #Split the data into Train, Validation, and Test sets.
    train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
    train_paths, val_paths = train_test_split(train_paths, test_size=0.25, random_state=42)

    # Print the number of images in each set
    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Test images: {len(test_paths)}")

    #Transfering all the generated images and label files from the "data\images" or "data\labels" folders to the corresponding ones in the train, test, or val folders.
    for folder in ["train", "val", "test"]:
        for file_name in os.listdir(abspath_images_dir):
            current_file_path = os.path.join(abspath_images_dir, file_name)
            
            #Check if the image file belongs to the train folder.
            if folder == "train":
                if current_file_path in train_paths:
                    new_file_path = os.path.join(abspath_data_dir, folder, "images", file_name)
                    
                    #Find the corresponding label file (if it exists) and transfer it to the "train\labels" folder.
                    label_file_name = file_name.split(".")[0]+".json"
                    current_label_file_path = os.path.join(abspath_labels_dir, label_file_name)
                    if os.path.exists(current_label_file_path):
                        new_label_file_path = os.path.join(abspath_data_dir, folder, "labels", label_file_name)
                        os.replace(current_label_file_path, new_label_file_path)

                    #Transfer the image to the "train\images" folder.
                    os.replace(current_file_path, new_file_path)

            #Check if the image file belongs to the test folder.
            if folder == "test":
                if current_file_path in test_paths:
                    new_file_path = os.path.join(abspath_data_dir, folder, "images", file_name)

                    #Find the corresponding label file (if it exists) and transfer it to the "test\labels" folder.
                    label_file_name = file_name.split(".")[0]+".json"
                    current_label_file_path = os.path.join(abspath_labels_dir, label_file_name)
                    if os.path.exists(current_label_file_path):
                        new_label_file_path = os.path.join(abspath_data_dir, folder, "labels", label_file_name)
                        os.replace(current_label_file_path, new_label_file_path)

                    #Transfer the image to the "test\images" folder.
                    os.replace(current_file_path, new_file_path)

            #Check if the image file belongs to the val folder.
            if folder == "val":
                if current_file_path in val_paths:
                    new_file_path = os.path.join(abspath_data_dir, folder, "images", file_name)

                    #Find the corresponding label file (if it exists) and transfer it to the "val\labels" folder.
                    label_file_name = file_name.split(".")[0]+".json"
                    current_label_file_path = os.path.join(abspath_labels_dir, label_file_name)
                    if os.path.exists(current_label_file_path):
                        new_label_file_path = os.path.join(abspath_data_dir, folder, "labels", label_file_name)
                        os.replace(current_label_file_path, new_label_file_path)

                    #Transfer the image to the "val\images" folder.
                    os.replace(current_file_path, new_file_path)
    print("Successfully transfered all files to the train, test, and validation directories.")
else:
    print("No images found in data\\images directory.")