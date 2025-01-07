import albumentations as alb
import json
import os
import cv2
import numpy as np

#Get absolute paths to different directories.
data_dir = "data"
abspath_data_dir = os.path.join(os.path.abspath(data_dir))
augmented_data_dir = "augmented_data"
abspath_augmented_data_dir = os.path.join(os.path.abspath(augmented_data_dir))

#Initialize the data augmentator.
augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.VerticalFlip(p=0.5),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2)],
                         bbox_params=alb.BboxParams(format="albumentations", label_fields=["class_labels"]))

#Apply the augmentator to every image in the train, test, and validation folders. 
for folder in ["train", "test", "val"]:
    for image_file in os.listdir(os.path.join(abspath_data_dir, folder, "images")):
        #Grab an image and its label (if it exists).
        image = cv2.imread(os.path.join(abspath_data_dir, folder, "images", image_file))

        coords = [0,0,0.0001,0.001]
        label_path = os.path.join(abspath_data_dir, folder, "labels", f"{image_file.split(".")[0]}.json")
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                #Grab the coordinates of the bounding box in the image and normalize them.
                label = json.load(f)
                coords = label["shapes"][0]["points"]
                coords = [item for index in coords for item in index]
                coords = list(np.divide(coords, [640, 480, 640, 480]))
        
        try:
            for x in range(60):
                #Augment the image.
                augmented = augmentor(image=image, bboxes=[coords], class_labels=["face"])
                #Save the augmented image.
                cv2.imwrite(os.path.join(abspath_augmented_data_dir, folder, "images", f"{image_file.split(".")[0]}.{x}.jpg"), augmented["image"])
                
                #Create the label information for the corresponding image. 
                annotation = {}
                annotation["image"] = image_file

                if os.path.exists(label_path):
                    if len(augmented["bboxes"]) == 0:
                        annotation["bbox"] = [0, 0, 0, 0]
                        annotation["class"] = 0
                    else:
                        annotation["bbox"] = augmented["bboxes"][0]
                        annotation["class"] = 1
                else:
                    annotation["bbox"] = [0, 0, 0, 0]
                    annotation["class"] = 0
                
                #Save the label information for the corresponding image. 
                with open(os.path.join(abspath_augmented_data_dir, folder, "labels", f"{image_file.split(".")[0]}.{x}.json"), "w") as f:
                    json.dump(annotation, f)

        except Exception as error:
            print(error)

print("Completed augmented data creation.")