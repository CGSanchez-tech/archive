import os
import cv2
import time 
import numpy as np
#issues: matplotlib needed to be installed again
#pip install matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#new 
from keras import utils # tools for creating one-hot encoding
from keras.models import Sequential                       # Type of model we wish to use
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten       # Types of layers we wish to use
#old: from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from IPython.display import display, Image
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
#from multiprocessing import Pool
#new
from keras.models import load_model

start_cell4 = time.time()

loaded_model = load_model("facial_expression_model.h5")


custom_test_image_name = input("Enter a JPEG image: ")
#make a format/input check maybe?

custom_test_image = cv2.imread(custom_test_image_name)
custom_test_image = cv2.cvtColor(custom_test_image, cv2.COLOR_BGR2GRAY)
custom_test_image = cv2.resize(custom_test_image, (48, 48))
custom_test_image = custom_test_image.astype('float32') / 255.0

# Reshape the image to match the model input shape
custom_test_image = np.expand_dims(custom_test_image, axis=0)
custom_test_image = np.expand_dims(custom_test_image, axis=-1)

########new above

# Make predictions on the custom test image
prediction = loaded_model.predict(custom_test_image)
prediction_prob = prediction[0]

emotion_label = np.argmax(prediction[0])

# Map the predicted label to emotion class
emotion_classes = {0: 'happy', 1: 'sad', 2: 'angry'}
predicted_emotion = emotion_classes[emotion_label]

# Print the custom test image and its predicted label
print(f"Predicted Emotion: {predicted_emotion}")
print(f"Confidence [happy, sad, angry]: {prediction_prob}")

#Display the custom test image using matplotlib
plt.imshow(custom_test_image[0, :, :, 0])
plt.title(f"Predicted Emotion: {predicted_emotion}")
plt.axis('off')  # Hide axes
plt.show()

from PIL import Image
# Display the original custom test image using PIL
img_pil = Image.open(custom_test_image_name)
plt.imshow(np.array(img_pil))
plt.title(f"Predicted Emotion: {predicted_emotion}")
plt.axis('off')  # Hide axes
plt.show()

end_cell4 = time.time()
elapsed_time4 = end_cell4 - start_cell4
print(f"Time taken for testing images : {elapsed_time4} seconds")
print("")

