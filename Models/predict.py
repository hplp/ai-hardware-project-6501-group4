import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing import image

# Define a function to load and preprocess a single image
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(32, 32))  # Resize to 32x32
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to the range [0, 1]
    return img_array

# Path to the single image you want to predict
image_path = "mnist_images/test/0/10.png"  # Replace with your image path

# Load the best model
best_model = tf.keras.models.load_model('best_model.h5')

# Load and preprocess the image
img_array = load_and_preprocess_image(image_path)

# Make prediction
prediction = best_model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class

# Display the image and the predicted label
img = image.load_img(image_path)
plt.imshow(img)
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()

print(f"Predicted class: {predicted_class}")
