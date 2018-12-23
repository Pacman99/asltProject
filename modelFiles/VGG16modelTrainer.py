from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.models import Model
classifier = Sequential()
base_model = VGG16(weights='imagenet', include_top=False)
new_classifier = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
classifier.add(new_classifier)
classifier.add(Conv2D(64, 3, 3, input_shape = (224,224, 3), activation = 'relu'))
classifier.add(Conv2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(128, 3, 3, activation = 'relu'))
classifier.add(Conv2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(256, 3, 3, activation = 'relu'))
classifier.add(Conv2D(256, 3, 3, activation = 'relu'))
classifier.add(Conv2D(256, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(512, 3, 3, activation = 'relu'))
classifier.add(Conv2D(512, 3, 3, activation = 'relu'))
classifier.add(Conv2D(512, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(512, 3, 3, activation = 'relu'))
classifier.add(Conv2D(512, 3, 3, activation = 'relu'))
classifier.add(Conv2D(512, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(4096, activation = 'relu'))
classifier.add(Dense(4096, activation = 'relu'))
classifier.add(Dense(1000, activation = 'relu'))
classifier.add(Dense(output_dim = 30, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.25)

train_generator = train_datagen.flow_from_directory(
    '/home/aslt/signData/top30Classes',
    target_size=(224, 224),
    batch_size= 15,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    '/home/aslt/signData/top30Classes',
    target_size=(224, 224),
    batch_size= 15,
    class_mode='categorical',
    subset='validation')

filePath="/home/aslt/modelFiles/callbacksAlexNetWeights.hdf5"
checkpoint = ModelCheckpoint(filePath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
classifier.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples,
    epochs = 10,
    callbacks = [checkpoint])



classifier.save('aslModel.h5')
