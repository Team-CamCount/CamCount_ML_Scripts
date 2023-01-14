import tensorflow as tf
import numpy as np
import cv2


text = ""
count = 0
wait = 0
person_there = False

left_thresh = 100
right_thresh = 380
false_negative_hold_time = 10


# interpreter = tf.lite.Interpreter(model_path="lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite")
interpreter = tf.lite.Interpreter(model_path="lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite")
interpreter.allocate_tensors()

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for edge, color in edges.items():
        p1, p2 = edge

        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)

    y_r_ear, x_r_ear, r_ear_conf = shaped[3]
    
    if r_ear_conf > confidence_threshold:
        cv2.circle(frame, (int(x_r_ear), int(y_r_ear)), 4, (0,255,0), -1)

    y_l_ear, x_l_ear, l_ear_conf = shaped[4]
    
    if l_ear_conf > confidence_threshold:
        cv2.circle(frame, (int(x_l_ear), int(y_l_ear)), 4, (0,255,255), -1)
    
    if r_ear_conf > confidence_threshold and l_ear_conf > confidence_threshold:
        return x_r_ear, x_l_ear
    elif l_ear_conf > confidence_threshold:
        return -1, x_l_ear
    elif r_ear_conf > confidence_threshold:
        return -1, x_r_ear
    else:
        return -1, -1


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    frame = frame[:, 80:560, :]
    
    # Reshape image
    img = frame.copy()
    
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256,256)
    input_image = tf.cast(img, dtype=tf.uint8)
    
    # Setup input and output 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    # Rendering 
    right_ear, left_ear = draw_connections(frame, keypoints_with_scores, EDGES, 0.5)

    if left_thresh < left_ear < right_thresh or left_thresh < right_ear < right_thresh:
        person_there = True
        wait = 0
        text = "Person in Region = Yes"

    else:
        wait += 1
        if wait >= false_negative_hold_time:
            text = "Person in Region = No"
            if person_there == True:
                count += 1
                person_there = False



    cv2.line(frame, (int(left_thresh), int(480)), (int(left_thresh), int(0)), (255,0,255), 2)
    cv2.line(frame, (int(right_thresh), int(480)), (int(right_thresh), int(0)), (255,0,255), 2)
    cv2.putText(frame, text, (20, 50), 0, 1, (0,0,0), 2)
    cv2.putText(frame, f"Count: {count}", (150, 100), 0, 1, (0,0,0), 2)

    
    cv2.imshow('MoveNet Lightning', frame)
    print(count)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()