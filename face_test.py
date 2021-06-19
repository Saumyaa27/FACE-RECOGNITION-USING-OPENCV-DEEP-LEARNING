# Importing the required libraries
from PIL import Image
import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

# Load the model trained and saved before
model = load_model('E:\\Saumyaa\\sem 6\\ai app\\facefeatures_new_model\\facefeatures_new_model')

# Load Cascade Classifier - a pre-trained frontal_face haar classifier (FACE DETECTION)
face_cascade = cv2.CascadeClassifier('E:\\Saumyaa\\sem 6\\ai app\\Deep-Learning-Face-Recognition-master\\haarcascade_frontalface_default.xml')


# Define a function which detects the face and labels it, if the face is known
# Else returns the input image
def face_extractor(img):

    # MultiScale uses features from the face_cascade to detect objects of different sizes in the input image
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))

    # An empty list indicates that no faces were identified in the given input image
    if faces is ():
        return None

    # If face != () then loop over each face identified
    # Calculate the probability of the labels and assign the label with maximum probability to the face detected
    for (x, y, w, h) in faces:

        # Draws a bounding rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        # 'cface' image in the form of an array of pixel values
        cface = img[y:y + h, x:x + w]
        if type(cface) is np.ndarray:
            # Resizing the input image to 224 x 224 resolution which is the same as the training images
            cface = cv2.resize(cface, (224, 224))
            # Convert the resized image into the PIL image format
            im = Image.fromarray(cface, 'RGB')
            img_array = np.array(im)
            # Our keras model used a 4D tensor, (images x height x width x channel)
            # So changing dimension of the image into 1 x 224 x 224 x 3
            img_array = np.expand_dims(img_array, axis=0)
            # Predicting the probability of each label for the input image
            pred = model.predict(img_array)
            print(f"The predicted probabilities are: {pred}")
            # Determing the label values
            if (pred[0][0] > 0.5):
                name = 'Akriti'
                cv2.putText(frame, name, (x, y+2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            elif (pred[0][1] > 0.5):
                name = 'Saumyaa'
                cv2.putText(frame, name, (x, y+2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            elif (pred[0][2] > 0.5):
                name = 'Shilpa'
                cv2.putText(frame, name, (x, y+2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            elif (pred[0][3] > 0.5):
                name = 'Shilpi'
                cv2.putText(frame, name, (x, y+2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face found", (x, y+2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    # Returning the array containing each face detected in the form of numpy array
    return cface


# Initialize the Web Cam of the device - '0' indicates the webCam index (START CAPTURING FACE)
video_capture = cv2.VideoCapture(0)

# Loop over the frames received and recognize the presence of a 'known' face
while True:
    _, frame = video_capture.read()
    face = face_extractor(frame)                    # Function call to 'face_extractor'
    cv2.imshow('Video', frame)                      # Method used to display image captured in a window
    # This command lets the program wait for 1 millisecond for the user to press 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()                             # Closes the capturing device (works as a destructor)
cv2.destroyAllWindows()                             # Destroys all the windows created by imshow method of cv2
print("Successfully completed the detection and recognition of face.")