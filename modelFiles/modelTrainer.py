import keras
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
classifier = Sequential()

classifier.add(Conv2D(64, 3, 3, input_shape = (112,112, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(64, 3, 3, input_shape = (56, 56, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(1000, activation = 'relu'))
classifier.add(Dense(output_dim = 30, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/home/aslt/signData/top30Classes', target_size = (112, 112), batch_size = 10, class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/home/aslt/signData/top30ClassesTesting', target_size = (112, 112), batch_size = 10, class_mode = 'categorical')


classifier.fit_generator(training_set, samples_per_epoch = 50, nb_epoch = 1, validation_data = test_set, nb_val_samples = 300)

classifier.save('ASLModel.h5')
