######## proto -5 
# date : 27/3/24

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import threading

# Load pre-trained face detection model
prototxt_path = r'.\computer_vision-master\CAFFE_DNN\deploy.prototxt'
caffemodel_path = r'.\computer_vision-master\CAFFE_DNN\res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Initialize webcam
cam = cv2.VideoCapture(0)

# Set frame width, height, and frame rate
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 380)
cam.set(cv2.CAP_PROP_FPS, 30)

shirt_img = None

def select_shirt_image():
    global shirt_img
    file_path = filedialog.askopenfilename()
    if file_path:
        shirt_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if shirt_img is not None:
            print('Selected shirt:', file_path)
        else:
            print('Error: Unable to load shirt image')

def show_gui():
    root = tk.Tk()
    root.title("Select Shirt Image")
    select_btn = tk.Button(root, text="Select Shirt Image", command=select_shirt_image)
    select_btn.pack()
    root.mainloop()

gui_thread = threading.Thread(target=show_gui)
gui_thread.start()

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        print('Error: Failed to open webcam or read frame')
        break

    # Detect faces in the frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        threshold_value = 0.4  # Adjust this threshold value as needed

        if confidence > threshold_value:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype(int)

            # Calculate shirt position
            y_shirt = y + int(0.7 * h)  # Adjust multiplier as needed
            h_shirt = int(0.5 * h)  # Adjust multiplier as needed
            
            # Calculate horizontal position with offset
            x_offset = int(0.5 * w)  # Adjust the offset percentage as needed
            x_shirt = x - x_offset  # Start slightly to the left of the face
            w_shirt = w + x_offset  # Extend to the right by adding the offset
            
            # Inside the face detection loop
            if shirt_img is not None:
                aspect_ratio = shirt_img.shape[1] / shirt_img.shape[0]
                new_h_shirt = int(w_shirt / aspect_ratio)
                
                # Ensure the scaling factor is valid
                if new_h_shirt > 0:
                    resized_shirt = cv2.resize(shirt_img, (w_shirt, new_h_shirt), interpolation=cv2.INTER_LINEAR)

                    # Convert RGBA to RGB (remove alpha channel)
                    if resized_shirt.shape[2] == 4:
                        resized_shirt = resized_shirt[:, :, :3]

                    # Apply the resized shirt to the frame
                    frame[y_shirt:y_shirt + new_h_shirt, x_shirt:x_shirt + w_shirt] = resized_shirt[:min(new_h_shirt, frame.shape[0] - y_shirt),
                                                                                                 :min(w_shirt, frame.shape[1] - x_shirt)]

    cv2.imshow('Virtual Try-On', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

# Wait for the GUI thread to finish
gui_thread.join()
