import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras import regularizers

classifier = Sequential()

classifier.add(Conv2D(32, 3, 3, input_shape = (64,64, 3), activation = 'relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(1000, activation = 'relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
classifier.add(Dense(output_dim = 30, activation = 'softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.25)

train_generator = train_datagen.flow_from_directory(
    '/home/aslt/signData/top30Classes',
    target_size=(64, 64),
    batch_size= 15,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    '/home/aslt/signData/top30Classes',
    target_size=(64, 64),
    batch_size= 15,
    class_mode='categorical',
    subset='validation')
filepath="/home/aslt/modelFiles"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
classifier.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples,
    epochs = 10,
    callbacks = [checkpoint])



classifier.save('aslModel.h5')
