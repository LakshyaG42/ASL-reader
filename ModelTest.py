import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('asl_model.h5')

# Dictionary mapping index to ASL characters
asl_characters = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
                  10: "a", 11: "b", 12: "c", 13: "d", 14: "e", 15: "f", 16: "g", 17: "h", 18: "i",
                  19: "j", 20: "k", 21: "l", 22: "m", 23: "n", 24: "o", 25: "p", 26: "q", 27: "r",
                  28: "s", 29: "t", 30: "u", 31: "v", 32: "w", 33: "x", 34: "y", 35: "z"}

# Function to predict ASL character from input image
def predict_asl_character(image_path):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize image 
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Preprocess image
    img = img.astype('float32') / 255.0

    # Prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    return asl_characters[predicted_class]


image_path = 'example_image.jpg'
predicted_character = predict_asl_character(image_path)
print("Predicted ASL Character:", predicted_character)
