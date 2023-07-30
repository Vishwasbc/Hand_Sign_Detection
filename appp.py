import cv2
from flask import Flask, render_template, Response
import tensorflow as tf
import numpy as np
import mediapipe as mp

loaded_model=tf.keras.models.load_model('highaccuracymodel.h5')
mp_holistic=mp.solutions.holistic
actions=np.array(['hello','thanks','iloveyou'])
sequence=[]

def mediapipe_detection(image,model):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #color convertion brg2rgb
    image.flags.writeable=False #image is no longer writiable
    results=model.process(image) #make prediction
    image.flags.writeable=True #image is now writiable
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #color convertion rgb2brg
    return image,results

def exract_keypoints(results):
    pose=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    face=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    return np.concatenate([pose,face,lh,rh])

app = Flask(__name__)
cap = cv2.VideoCapture(0)

def generate_frames():
    sequence = []  # Initialize the 'sequence' variable outside the loop
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
        # Read feed
            ret, frame = cap.read()

        # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
        
        
        # 2. Prediction logic in order 
            keypoints = exract_keypoints(results)
        
            sequence.append(keypoints)
            sequence = sequence[-30:]
        
            if len(sequence) == 30:
                res = loaded_model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                cv2.putText(image,actions[np.argmax(res)] , (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0 , 255), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame_data = buffer.tobytes()

            #Yield the frame data as a response
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/')
def index():
    return render_template('in.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
