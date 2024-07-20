from keras import Sequential, layers
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from PIL import Image

# check the dataset for any corrupted or non-image files that may be there - then delete them
def validate_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify that it is an image
            except (IOError, SyntaxError) as e:
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)

validate_images(os.path.dirname(__file__)+"/dataset/train/Cat")
validate_images(os.path.dirname(__file__)+"/dataset/train/Dog")
validate_images(os.path.dirname(__file__)+"/dataset/validation/Cat")
validate_images(os.path.dirname(__file__)+"/dataset/validation/Dog")

# process the images and apply data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.dirname(__file__)+"/dataset/train",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

validation_generator = train_datagen.flow_from_directory(
    os.path.dirname(__file__)+"/dataset/validation",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)


# build the model
model = Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150,150,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# compile it
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# train it
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=50
)

# save it
model.save(os.path.dirname(__file__)+"/cat_dog_classifier.h5")

# plot graph
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()