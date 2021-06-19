import cv2
import numpy as np

# Load Cascade Classifier - a pre-trained frontal_face haar classifier
face_classifier = cv2.CascadeClassifier(
    'E:\\Saumyaa\\sem 6\\ai app\\Deep-Learning-Face-Recognition-master\\haarcascade_frontalface_default.xml')


# Defining a function which identifies the face and extracts it
# If face is detected, it returns a cropped image
# Else, the input image is returned
def face_extractor(img):
    # Converting input image from RGB color space to GRAY SCALE
    # This is done because haar casscade works only on GrayScale images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # MultiScale uses features from the face_classifier to detect objects of different sizes in the input image
    faces = face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    # 'faces' containes the coordinates of the bounding boxes enclosing the faces
    # If this list is empty then return None - 'No face is detected'
    if faces is ():
        return None

    # Else, crop all faces found
    # 'faces' contains a tuple for each box - top-left coordinates of bounding box (x, y), width(w) and height(h) of the box
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y + h + 50, x:x + w + 50]

    return cropped_face


# Initialize the Web Cam of the device - '0' indicates the webCam index
cap = cv2.VideoCapture(0)
count = 0

# Start collecting samples of your face from web cam input
while True:

    # 'cap.read()' returns bool (True/False). If frame is read correctly, it will be True.
    # 'ret' is a boolean variable that returns true if the frame is available
    # 'frame' is an image array vector captured based on the default frames per second
    ret, frame = cap.read()
    cv2.imshow('Frame', frame)  # Method used to display image captured in a window
    if face_extractor(frame) is not None:
        count += 1  # Count of images captured
        face = cv2.resize(face_extractor(frame), (400, 400))  # Resize the image frame - (new_width=400, new_height=400)

        # Save image in a specified directory with unique name
        file_name_path = 'E:\\Saumyaa\\sem 6\\ai app\\Images\\' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Display the live count of the images captured
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Cropped face', face)  # Dislay the resized image frame

    else:
        print("Face not found")
        pass

    # This loop runs until the count equals 200 or the program waits for 1 millisecond for the user to press 'Enter' key to quit
    if cv2.waitKey(5) == 13 or count == 50:
        break

cap.release()  # Closes the capturing device (works as a destructor)
cv2.destroyAllWindows()  # Destroys all the windows created by imshow method of cv2
print("Successfully completed the collection of samples.")