import cv2
from tensorflow.keras.models import model_from_json
import numpy as np

# Load the model architecture from JSON file
json_file = open("emotiondetector_model.json", "r")
model_json = json_file.read()
json_file.close()

# Load the model from JSON
model = model_from_json(model_json)

# Load the model weights
model.load_weights("emotiondetector_weights.h5")

# Initialize Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Dictionary mapping label indices to emotion names
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Open webcam
webcam = cv2.VideoCapture(0)

# Main loop for real-time emotion detection
while True:
    # Read frame from webcam
    ret, frame = webcam.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    try:
        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Extract face region from grayscale frame
            face_image = gray[y:y + h, x:x + w]

            # Resize face image to match model input size
            face_image_resized = cv2.resize(face_image, (48, 48))

            # Preprocess the face image
            processed_image = extract_features(face_image_resized)

            # Make prediction using the model
            prediction = model.predict(processed_image)

            # Get the index of the predicted emotion
            predicted_emotion_index = np.argmax(prediction)

            # Get the label corresponding to the predicted emotion index
            predicted_emotion = labels[predicted_emotion_index]

            # Display the predicted emotion label on the frame
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the frame with annotations
        cv2.imshow("Emotion Detection", frame)

        # Check for key press to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    except cv2.error:
        pass

# Release resources
webcam.release()
cv2.destroyAllWindows()
