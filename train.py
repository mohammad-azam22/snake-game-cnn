import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator

from preprocessing import train_preprocessing_pipeline
from model import cnn_model

def main(): 
    base_dir = "./data/train"
    classes = ["L", "U", "R", "D"]

    X = []
    y = []

    for cls in classes:
        imgs = os.listdir(f"{base_dir}/{cls}")
        for img in imgs:
            im = load_img(f"{base_dir}/{cls}/{img}")    # loading images
            X.append(im)
            y.append(cls)
    
    X, y = train_preprocessing_pipeline(X, y)

    train_datagen = ImageDataGenerator(
        rotation_range=10,        # slight tilt
        width_shift_range=0.1,    # off-center drawings
        height_shift_range=0.1,
        zoom_range=0.1,           # zoom in/out
        shear_range=0.05,         # VERY small
        fill_mode="nearest"
    )

    indices = np.arange(0, len(X))
    np.random.shuffle(indices)
    
    # train 80% | validation 20%
    upper_train_index = int(len(indices) * 80 / 100)

    X_train = X[indices[:upper_train_index]]
    X_val = X[indices[upper_train_index:]]

    y_train = y[indices[:upper_train_index]]
    y_val = y[indices[upper_train_index:]]

    train_generator = train_datagen.flow(
        X_train,
        y_train,
        batch_size=32
    )

    model = cnn_model(num_classes=len(classes))
    history = model.fit(train_generator, validation_data=(X_val, y_val), epochs=10, batch_size=200)

    model.save("./models/letter_cnn_v1.keras")

if __name__ == "__main__":
    main()
