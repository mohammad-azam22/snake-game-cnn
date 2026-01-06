import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical

def center_and_resize(img):
    pixels = img.load()

    # Find the bounding box of the white object
    width, height = img.size
    min_x, min_y = width, height
    max_x, max_y = 0, 0

    for x in range(width):
        for y in range(height):
            if pixels[x, y] > 0:  # Assuming white is 255 and black is 0
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

    # Calculate the dimensions and center of the white object
    obj_width = max_x - min_x + 1
    obj_height = max_y - min_y + 1

    # Create a new 256x256 black image
    max_dim = obj_width if obj_width > obj_height else obj_height
    if max_dim < 0: 
        max_dim = 0
    new_img = Image.new("L", (max_dim + 10, max_dim + 10), 0)
    new_pixels = new_img.load()

    # Calculate the position to paste the white object in the center
    paste_x = 10
    paste_y = 5

    # Paste the white object into the new image
    for x in range(obj_width):
        for y in range(obj_height):
            if pixels[min_x + x, min_y + y] > 0:
                new_pixels[paste_x + x, paste_y + y] = pixels[min_x + x, min_y + y]

    return new_img

def preprocess_y(y):
    mapping = {"L": 0, "U": 1, "R": 2, "D": 3}
    y_temp = []

    for label in y:
        y_temp.append(mapping.get(label))   

    y_temp = to_categorical(np.array(y_temp))    # one-hot encoding
    
    return y_temp

def train_preprocessing_pipeline(X, y):
    X_temp = []

    for im in X:
        im = im.convert('L')    # grayscaling
        x = np.array(im)    # converting to numpy array
        x = x.reshape(64, 64, 1)    # reshaping
        X_temp.append(x)

    X = np.array(X_temp) / 255    # normalizing
    y = preprocess_y(y)

    return X, y

def inference_preprocessing_pipeline(X):
    X_temp = []

    for im in X:
        im = im.convert('L')    # grayscaling
        im = center_and_resize(im)    # centering
        im = im.resize((64, 64), Image.Resampling.LANCZOS)    # resampling image
        x = np.array(im)    # converting to numpy array
        x = x.reshape(64, 64, 1)    # reshaping
        X_temp.append(x)

    X = np.array(X_temp) / 255    # normalizing

    return X
