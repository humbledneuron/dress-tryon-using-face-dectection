#!/usr/bin/python3

# date : 20/3/24

import cv2
import tkinter as tk
from PIL import Image, ImageTk

class DressTryOnApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Dress Try-On App")

        self.video_feed_label = tk.Label(root)
        self.video_feed_label.pack()

        self.dress_selection_label = tk.Label(root, text="Select Dress:")
        self.dress_selection_label.pack()

        self.dress_options = ["Dress 1", "Dress 2", "Dress 3"]
        self.selected_dress = tk.StringVar()
        self.selected_dress.set(self.dress_options[0])

        self.dress_dropdown = tk.OptionMenu(root, self.selected_dress, *self.dress_options)
        self.dress_dropdown.pack()

        self.start_button = tk.Button(root, text="Start Try-On", command=self.start_try_on)
        self.start_button.pack()

        self.capture = cv2.VideoCapture(0)
        self.update_video_feed()

    def update_video_feed(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.video_feed_label.config(image=self.photo)
        self.video_feed_label.after(10, self.update_video_feed)

    def start_try_on(self):
        selected_dress = self.selected_dress.get()
        # Apply dress overlay on the live video feed using image processing
        # You'll need to implement this part based on your image processing logic

if __name__ == "__main__":
    root = tk.Tk()
    app = DressTryOnApp(root)
    root.mainloop()

