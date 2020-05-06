from keras.models import load_model
import cv2
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.models import Sequential
import tensorflow as tf 

#Load Dog Breed labels
print('Loading dog breed ......')
with open('DogBreed.pkl', 'rb') as f:
    dog_names = pickle.load(f)


def load_Xception():
    Xception_model = Sequential()
    Xception_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    Xception_model.add(Dense(133, activation='softmax'))
    Xception_model.compile(loss='categorical_crossentropy',
                           optimizer='Adam', metrics=['accuracy'])
    Xception_model.load_weights('model/weights.best.Xception.hdf5')
    return Xception_model


def ResNet50_predict_labels(img_path):
    from keras.applications.resnet50 import preprocess_input
    # returns prediction vector for image located at img_path
    print('Loading ResNet50  ......')
    ResNet50_model = ResNet50(weights='imagenet')
 
    img = preprocess_input(path_to_tensor(img_path))

    pred = ResNet50_model.predict(img)
    return np.argmax(pred)


def face_detector(img_path):

    print('Loading Haarcascade  ......')
    face_cascade = cv2.CascadeClassifier(
    'model/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def extract_Xception(tensor):
    from keras.applications.xception import Xception, preprocess_input
    return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def Xception_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    # obtain predicted vector
    print("Loading Xception......")
    Xception_model = load_Xception()

    predicted_vector = Xception_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def breed_prediction(img_path, classifier_type='Xception'):

    model = classifier_type
    if model == 'Xception':
        predictor = Xception_predict_breed(img_path)
        print("Prediction : ", predictor)
    if not model:
        raise ValueError("Please specify the model...")
    
    if face_detector(img_path):
        # text = "Hello, human!\nYou look like a "
        text = predictor.split('.')[-1]

    # if not human_found:
    elif dog_detector(img_path):
        # text = "Hello, dog!\nYour predicted breed is "
        text = predictor.split('.')[-1]
    else:
        raise ModuleNotFoundError
    print("Returning prediction")
    return text