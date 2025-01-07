import os
import tensorflow as tf
import time
import cv2
import uuid
import numpy as np
import json
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

#Get absolute paths to different directories.
data_dir = "data"
abspath_data_dir = os.path.join(os.path.abspath(data_dir))
augmented_data_dir = "augmented_data"
abspath_augmented_data_dir = os.path.join(os.path.abspath(augmented_data_dir))

#Define the absolute paths for the augementd data's train, test, and val image and label directories.
train_images_dir = os.path.join(abspath_augmented_data_dir, "train", "images")
test_images_dir = os.path.join(abspath_augmented_data_dir, "test", "images")
val_images_dir = os.path.join(abspath_augmented_data_dir, "val", "images")

train_labels_dir = os.path.join(abspath_augmented_data_dir, "train", "labels")
test_labels_dir = os.path.join(abspath_augmented_data_dir, "test", "labels")
val_labels_dir = os.path.join(abspath_augmented_data_dir, "val", "labels")

#Convert file to the containing image.
def load_image(file_path):
    byte_image = tf.io.read_file(file_path)
    image = tf.io.decode_jpeg(byte_image)
    return image

#Generate three datasets from the image files in the train, test, and val folders of the augmented data folder. Load there image, resize to a shape of (120, 120), and normalize the image's values to between 0 and 1.
train_images_dataset = tf.data.Dataset.list_files(os.path.join(train_images_dir, "*.jpg"), shuffle=False)
train_images_dataset = train_images_dataset.map(load_image)
train_images_dataset = train_images_dataset.map(lambda x: tf.image.resize(x, (120, 120)))
train_images_dataset = train_images_dataset.map(lambda x: x/255)

test_images_dataset = tf.data.Dataset.list_files(os.path.join(test_images_dir, "*.jpg"), shuffle=False)
test_images_dataset = test_images_dataset.map(load_image)
test_images_dataset = test_images_dataset.map(lambda x: tf.image.resize(x, (120, 120)))
test_images_dataset = test_images_dataset.map(lambda x: x/255)

val_images_dataset = tf.data.Dataset.list_files(os.path.join(val_images_dir, "*.jpg"), shuffle=False)
val_images_dataset = val_images_dataset.map(load_image)
val_images_dataset = val_images_dataset.map(lambda x: tf.image.resize(x, (120, 120)))
val_images_dataset = val_images_dataset.map(lambda x: x/255)

#Combine the datasets from the train, test, and val folders into one dataset.
images = train_images_dataset.concatenate(test_images_dataset).concatenate(val_images_dataset)

#Generates the address of a file from the images folder everytime the iterator is called.
# batch = images.as_numpy_iterator().next()

#Extract information from the label file.
def load_labels(label_path):
    with open(label_path.numpy(), "r", encoding = "utf-8") as f:
        label = json.load(f)
    return [label["class"], label["bbox"]]

#Generate three datasets from the label files in the train, test, and val folders of the augmented data folder. Load the information of the labels.
train_labels = tf.data.Dataset.list_files(os.path.join(train_labels_dir, "*.json"), shuffle=False)
train_labels = train_labels.map(lambda x : tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
test_labels = tf.data.Dataset.list_files(os.path.join(test_labels_dir, "*.json"), shuffle=False)
test_labels = test_labels.map(lambda x : tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
val_labels = tf.data.Dataset.list_files(os.path.join(val_labels_dir, "*.json"), shuffle=False)
val_labels = val_labels.map(lambda x : tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

#Merge images with their labels to form a single dataset.
train = tf.data.Dataset.zip((train_images_dataset, train_labels))
train = train.shuffle(3000)
train = train.batch(8)
train = train.prefetch(4)

test = tf.data.Dataset.zip((test_images_dataset, test_labels))
test = test.shuffle(1000)
test = test.batch(8)
test = test.prefetch(4)

val = tf.data.Dataset.zip((val_images_dataset, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

def build_model():
    input_layer = Input(shape=(120, 120, 3))
    vgg = VGG16(include_top=False)(input_layer)

    #Classification part of the model.
    f1 = GlobalMaxPooling2D()(vgg)
    class_layer_1 = Dense(2048, activation="relu")(f1)
    output_class = Dense(1, activation="sigmoid")(class_layer_1)

    #Regression part of the model.
    f2 = GlobalMaxPooling2D()(vgg)
    regress_layer_1 = Dense(2048, activation="relu")(f2)
    output_regress = Dense(4, activation="sigmoid")(regress_layer_1)

    face_tracker = Model(inputs=input_layer, outputs=[output_class, output_regress])
    return face_tracker

#Initialize model.
face_tracker = build_model()

#Create optimizers and loss functions for the classifier and regression model.
batches_per_epoch = len(train)
learning_rate_decay = (1.0/0.75 - 1) / batches_per_epoch #Slows down learning to not overfit

#Both models will use the Adam optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=learning_rate_decay)

#The localization loss is the loss function for the regression model. It helps us determine the loss between the actual coordinates of the bounding box vs the predicted coordinates of the bounding box.
def localizationloss(ytrue, ypredict):

    ytrue = tf.cast(ytrue, tf.float32)

    delta_coord = tf.reduce_sum(tf.square(ytrue[:,:2] - ypredict[:,:2]))

    h_true = ytrue[:,3] - ytrue[:,1]
    w_true = ytrue[:,2] - ytrue[:,0]

    h_pred = ypredict[:,3] - ypredict[:,1]
    w_pred = ypredict[:,2] - ypredict[:,0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

    return delta_coord + delta_size

#Initializing the loss functions.
classloss = tf.keras.losses.BinaryCrossentropy()
regressionloss = localizationloss

#Building a model training class (The equivalent of a model.compile() and/or a model.fit()).
class facetracking(Model):
    def __init__(self, face_tracker, **kwargs):
        super().__init__(**kwargs)
        self.model = face_tracker

    def compile(self, optimizer, classloss, regressionloss, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.classloss = classloss
        self.regloss = regressionloss
    
    #This functoin goes through what happens at each step during training. i.e: Forward Pass, Loss Calculation, Backpropogation (to get the change in loss function w.r.t all the parameters), Update Parameters. 
    def train_step(self, batch, **kwargs):
        #Grab the true values of the batch (to perform a forward pass, and to calculate loss later on).
        X, y = batch
        y_class = tf.reshape(y[0], (-1, 1))
        #Set up a tracker to track the operations performed on variables within its scope so that the gradients of a loss function w.r.t each of those variables can be calculated.
        #It tracks the operations a variable goes through to apply chain rule to it to later to get the gradients of a loss function w.r.t that variable.
        with tf.GradientTape() as tape:
            
            #Forward Pass.
            pred_class, pred_coords = self.model(X, training=True)

            #Loss Calculation.
            batch_class_loss = self.classloss(y_class, pred_class)
            batch_reg_loss = self.regloss(y[1], pred_coords)

            total_loss = batch_reg_loss + 0.5 * batch_class_loss

        #Backpropogation.
        grad = tape.gradient(total_loss, self.model.trainable_variables)

        #Update Parameters.
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        #Metrics to track are returned (so to print them later to visualise how training is going). 
        return {"total_loss": total_loss, "class_loss": batch_class_loss, "regression_loss": batch_reg_loss}

    #Validation step.
    def test_step(self, batch, **kwargs):
        X, y = batch

        y_class = tf.reshape(y[0], (-1, 1))
        #Model training set to False because we are only validating how training is going in this step and not actually training.
        pred_class, pred_coords = self.model(X, training=False)
        
        batch_class_loss = self.classloss(y_class, pred_class)
        batch_reg_loss = self.regloss(y[1], pred_coords)

        total_loss = batch_reg_loss + 0.5 * batch_class_loss

        #Metrics to track are returned (so to print them later to visualise how training is going). 
        return {"total_loss": total_loss, "class_loss": batch_class_loss, "regression_loss": batch_reg_loss}
    
    #Get the prediction of a single batch. 
    def call(self, X, **kwargs):
        print(f"Received input: {X}")
        print(f"Model: {self.model}")
        if isinstance(X, tf.Tensor):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
        return self.model(X)
    
    def get_config(self):
        config = super().get_config().copy
        config.update({
            'face_tracker': self.model.to_json()  # Save the model architecture in JSON format
        })
        return config

    @classmethod
    def from_config(cls, config):
        face_tracker_config = config.pop('face_tracker')
        if isinstance(face_tracker_config, dict):
            # Convert dict to JSON string if needed
            face_tracker_config = json.dumps(face_tracker_config)
        face_tracker = tf.keras.models.model_from_json(face_tracker_config)  # Rebuild the model from JSON
        return cls(face_tracker=face_tracker, **config)

# Only execute this block when the script is run directly
if __name__ == "__main__":
    model = facetracking(face_tracker)
    model.compile(optimizer, classloss, regressionloss)
    hist = model.fit(train, epochs=20, validation_data=val)
    model.build(input_shape=(None, 120, 120, 3))

    #Save the model.
    model_dir = "models"
    abspath_model_dir = os.path.abspath(model_dir)
    model.save(os.path.join(abspath_model_dir, "FaceDetection.keras"))