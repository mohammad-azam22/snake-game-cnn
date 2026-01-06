# Larger CNN for the MNIST Dataset
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.layers import Dropout # type: ignore
from tensorflow.keras.layers import Flatten # type: ignore
from tensorflow.keras.layers import Conv2D # type: ignore
from tensorflow.keras.layers import MaxPooling2D # type: ignore

# define the larger model
def cnn_model(num_classes):
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(64, 64, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model