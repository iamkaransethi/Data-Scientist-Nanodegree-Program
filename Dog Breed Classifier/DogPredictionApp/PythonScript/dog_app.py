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


def load_Xception():
    Xception_model = Sequential()
    Xception_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    Xception_model.add(Dense(133, activation='softmax'))
    Xception_model.compile(loss='categorical_crossentropy',
                           optimizer='Adam', metrics=['accuracy'])
    Xception_model.load_weights('model/weights.best.Xception.hdf5')
    return Xception_model


Xception_model = load_Xception()
def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def face_detector(img_path):
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
    predicted_vector = Xception_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def breed_prediction(img_path, classifier_type='Xception'):

    model = classifier_type

    if model == 'Xception':
        predictor = Xception_predict_breed(img_path)

    if not model:
        raise ValueError("Please specify the model...")

    if face_detector(img_path):
        text = "Hello, human!\nYou look like a "
        text += predictor.split('.')[-1]
    elif dog_detector(img_path):
        text = "Hello, dog!\nYour predicted breed is "
        text += predictor.split('.')[-1]
    else:
        raise ModuleNotFoundError

    print(text)

    scale_percent = 50  # percent of original size
    img = cv2.imread(img_path)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    im_resized = cv2.resize(img, (width, height),
                            interpolation=cv2.INTER_LINEAR)

    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
    a.set_title(text)
    plt.show()


if __name__ == '__main__':
    
    #Load model
    ResNet50_model = ResNet50(weights='imagenet')
    Xception_model = load_Xception()
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_alt.xml')
    #Load Dog Breed labels
    with open('DogBreed.pkl', 'rb') as f:
        dog_names = pickle.load(f)

    #Taking input argument
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="Path to image to be scanned")
    args = vars(ap.parse_args())
    img_path = args["image"]

    breed_prediction(img_path, classifier_type='Xception')
