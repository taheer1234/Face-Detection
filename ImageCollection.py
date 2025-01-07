import os
import tensorflow as tf
import time
import cv2
import uuid

data_dir = "data"
abspath_images_dir = os.path.join(os.path.abspath(data_dir), "images")
image_number = 30

cap = cv2.VideoCapture(0)
for imgnum in range(image_number):
    print("COllecting Image: {}".format(imgnum))
    ret, frame = cap.read()
    image_name = os.path.join(abspath_images_dir, f"{str(uuid.uuid1())}.jpg")
    cv2.imwrite(image_name, frame)
    cv2.imshow("frame", frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
print("Successfully colected images.")