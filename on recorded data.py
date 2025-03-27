import cv2
import numpy as np
import os
import mediapipe as mp
import imutils
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose.flatten(), face.flatten(), lh.flatten(), rh.flatten()])

# Data path & actions
data_path = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30
sequence_length = 30
label_map = {label: num for num, label in enumerate(actions)}

# Load existing data instead of collecting it
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            try:
                res = np.load(os.path.join(data_path, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            except FileNotFoundError:
                print(f"Missing file: {data_path}/{action}/{sequence}/{frame_num}.npy")
                continue
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = keras.utils.to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Define model
model = keras.models.Sequential([
    keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)),
    keras.layers.LSTM(128, return_sequences=True, activation='relu'),
    keras.layers.LSTM(64, return_sequences=False, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(actions.shape[0], activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train model
model.fit(X_train, y_train, epochs=2000)

# Start video capture for sign detection
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture frame. Check camera connection.")
            continue  # Skip this loop iteration
        frame = imutils.resize(frame, width=1280)
        image, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        
        # Display frame
        cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()
