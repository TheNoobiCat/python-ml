from keras import models
import os
from PIL import Image
import numpy as np

def load_and_preprocess_image(image_path, target_size=(150, 150)):
    # load the image
    img = Image.open(image_path)
    
    # resize the image
    img = img.resize(target_size)
    
    # convert the image to a numpy array
    img_array = np.array(img)
    
    # Normalize it
    img_array = img_array / 255.0
    
    # Expand to match the model input shape (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


model = models.load_model(os.path.dirname(__file__)+"/cat_dog_classifier.h5")
def predict_image(image_path):
    img_array = load_and_preprocess_image(image_path)
    
    # make the prediction
    prediction = model.predict(img_array)
    
    # see what the model predicted
    if prediction[0] > 0.5:
        print("It's a dog!")
    else:
        print("It's a cat!")

# Set it as dog_to_test.jpg or cat_to_test.jpg! I left two images for you to try out :)
predict_image(os.path.dirname(__file__)+"/dog_to_test.jpg")