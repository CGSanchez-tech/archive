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

print("yippie")

#cell1
start_cell1 = time.time()

# Define the path to the dataset folders
happy_folder = "./Happy"
sad_folder = "./Sad"
angry_folder = "./Angry"


# Function to load and preprocess images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))  # Resize to a fixed size for the model
            images.append(img)
    return images


#####below new: DID NOT WORK - BIG BUG: https://github.com/python/cpython/issues/98360   
# Trying to open up the files faster to reduce run time
# Tried to use multiprocessing to parallelize the image loading process
# Not the best idea though, since this isn't even the part of the 
# code that takes the most time.
#
#def load_image(filename):
#    img = cv2.imread(filename)
#    if img is not None:
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        img = cv2.resize(img, (48, 48))
#    return img
#
#def load_images_from_folder_parallel(folder):
#    images = []
#    filenames = [os.path.join(folder, filename) for filename in os.listdir(folder)]
#    with Pool() as pool:
#        images = pool.map(load_image, filenames)
#    return images
#
# Replace the load_images_from_folder function with load_images_from_folder_parallel
#happy_images = load_images_from_folder_parallel(happy_folder)
#sad_images = load_images_from_folder_parallel(sad_folder)
#angry_images = load_images_from_folder_parallel(angry_folder)
#
#
######above new

# Load images and labels for each emotion
happy_images = load_images_from_folder(happy_folder)
sad_images = load_images_from_folder(sad_folder)
angry_images = load_images_from_folder(angry_folder)


# Create labels for each emotion category
# These labels label the images in their respective folders, 
# the array that prints out is 250 for each category 
# which is the amount of items in the folder
happy_labels = [0] * len(happy_images)
sad_labels = [1] * len(sad_images)
angry_labels = [2] * len(angry_images)

print("")
print("Labels Made")
print("")
end_cell1 = time.time()
elapsed_time1 = end_cell1 - start_cell1
print(f"Time taken for loading images: {elapsed_time1} seconds")
print("")
#cell2
start_cell2 = time.time()
# Concatenate images and labels
X = np.array(happy_images + sad_images + angry_images)
y = np.array(happy_labels + sad_labels + angry_labels)

# Normalize pixel values to range [0, 1]
X = X.astype('float32') / 255.0

# One-hot encode the labels
y = utils.to_categorical(y, 3)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##cell3
# Build the CNN model
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

print("Model Summary time")
print(model.summary())

print("")
end_cell2 = time.time()
elapsed_time2 = end_cell2 - start_cell2
print(f"Time taken for CNN model: {elapsed_time2} seconds")


print("")
start_cell3 = time.time()
# Compile the model with class weights
model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy']) 

##cell 4 

# Calculate class weights
total_samples = len(y_train)
class_weights = {0: total_samples / np.sum(y_train[:, 0]), 
                 1: total_samples / np.sum(y_train[:, 1]), 
                 2: total_samples / np.sum(y_train[:, 2])}
# Train the model with class weights
history = model.fit(X_train.reshape(-1, 48, 48, 1), y_train, batch_size=32, epochs=100, validation_split=0.1, class_weight=class_weights,verbose=0)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test.reshape(-1, 48, 48, 1), y_test)
losstr, accuracytr = model.evaluate(X_train.reshape(-1, 48, 48, 1), y_train)


# Save the trained model
model.save("facial_expression_model.h5")
print("Trained model saved")

print(f"Test accuracy: {accuracy*100:.2f}%")
print(f"Train accuracy: {accuracytr*100:.2f}%")

end_cell3 = time.time()
elapsed_time3 = end_cell3 - start_cell3
print(f"Time taken for testing images : {elapsed_time3} seconds")
print("")

###Testing on images

start_cell4 = time.time()
from keras.models import load_model
# Load the saved model
loaded_model = load_model("facial_expression_model.h5")

print("Loading and preprocessing images...")

# Function to load and preprocess images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))  # Resize to a fixed size for the model
            images.append(img)
    return images

#print("Testing an image")
# Load a custom test image
# Make this into a command in the terminal
#custom_test_image_path = "./Angry/02.jpg"

#custom_test_image = cv2.imread(custom_test_image_path)
#custom_test_image = cv2.cvtColor(custom_test_image, cv2.COLOR_BGR2GRAY)
#custom_test_image = cv2.resize(custom_test_image, (48, 48))
#custom_test_image = custom_test_image.astype('float32') / 255.0

# Reshape the image to match the model input shape
#custom_test_image = np.expand_dims(custom_test_image, axis=0)
#custom_test_image = np.expand_dims(custom_test_image, axis=-1)

######new below
#print("Input an image!:")
# Load a custom test image
# Make this into a command in the terminal

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


