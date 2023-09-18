import cv2
import dlib
from scipy.spatial import distance
from datetime import datetime, timedelta
import pygame
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection App")
        self.root.geometry("700x620")

        self.video_label = ttk.Label(root)
        self.video_label.pack()
        
        # Add a title label
        self.title_label = ttk.Label(root, text="Driver Drowsiness Monitoring App", font=("Helvetica", 26, "bold"), foreground="red")
        self.title_label.pack()
        
        # Load and display the image
        image_path = "App_img.jpeg"  # Replace with your image filename
        self.image = Image.open(image_path)
        self.image = self.image.resize((500, 333))  # Resize the image if needed
        self.image = ImageTk.PhotoImage(self.image)
        self.image_label = ttk.Label(root, image=self.image)
        self.image_label.pack()

        self.status_label = ttk.Label(root, text="Status: ", font=("Helvetica", 16))
        
        self.ear_label = ttk.Label(root, text="EAR: ", font=("Helvetica", 16))

        self.start_button = ttk.Button(root, text="Start Detection", command=self.start_detection, style="Custom.TButton")
        self.start_button.pack()
        
        self.quit_button = ttk.Button(root, text="Quit Detection", command=self.quit_detection, style="Custom.TButton")
        
        self.quit_app_button = ttk.Button(root, text="Quit App", command=self.on_closing, style="Custom.TButton")
        self.quit_app_button.pack()
        
        self.root.style = ttk.Style()
        self.root.style.configure("Custom.TButton", font=("Helvetica", 16), width=35)

        self.cap = cv2.VideoCapture(0)
        self.hog_face_detector = dlib.get_frontal_face_detector()
        self.dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.drowsy_timer_start = None
        self.drowsy_duration_threshold = timedelta(seconds=2)

        pygame.mixer.init()
        self.alarm_sound_path = "alarm.wav"
        self.alarm_sound = pygame.mixer.Sound(self.alarm_sound_path)
        self.alarm_playing = False

        self.running = False

    def start_detection(self):
        if not self.running:
            self.running = True
            self.quit_button.pack()  # No need to recreate the button
            self.quit_button.config(command=self.quit_detection)  # Update the command
            self.quit_app_button.pack_forget()  # Hide the Quit App button
            self.title_label.pack_forget()
            self.image_label.pack_forget()
            self.start_button.config(state="disabled")
            self.status_label.pack()
            self.ear_label.pack()
            
            # Reset variables for drowsiness detection
            self.drowsy_timer_start = None
            self.alarm_playing = False
            self.status_label.config(text="Status:")
            self.ear_label.config(text="EAR:")
            
            # Reopen the camera
            self.cap = cv2.VideoCapture(0)

            self.detect_drowsiness()

    def detect_drowsiness(self):
        if not self.running:
            return
        
        _, frame = self.cap.read()
        if frame is None:
            return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.hog_face_detector(gray)
        drowsy = False
        EAR = None

        for face in faces:
            face_landmarks = self.dlib_facelandmark(gray, face)
            leftEye = []
            rightEye = []

            for n in range(36,42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x,y))
                next_point = n+1
                if n == 41:
                    next_point = 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

            for n in range(42,48):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                rightEye.append((x,y))
                next_point = n+1
                if n == 47:
                    next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)

            EAR = (left_ear + right_ear) / 2
            EAR = round(EAR, 2)

            if EAR < 0.26:
                if self.drowsy_timer_start is None:
                    self.drowsy_timer_start = datetime.now()
                elif datetime.now() - self.drowsy_timer_start >= self.drowsy_duration_threshold:
                    drowsy = True
            else:
                self.drowsy_timer_start = None
                
        if EAR is not None:  # Check if EAR is a valid value
            self.ear_label.config(text=f"EAR: {EAR}")
        else:
            self.ear_label.config(text="EAR: No eyes detected")  # Display "No eyes detected" message

        if drowsy:
            if not self.alarm_playing:
                self.alarm_sound.play()
                self.alarm_playing = True
            self.status_label.config(text="Status: Drowsy - Wake Up!")
        else:
            self.alarm_playing = False
            pygame.mixer.stop()
            self.status_label.config(text="Status: Awake")

        # Display the video feed in the Tkinter window
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)
        self.video_label.config(image=frame)
        self.video_label.image = frame

        if self.running:
            self.root.after(10, self.detect_drowsiness)

    def quit_detection(self):
        self.running = False
        self.cap.release()
        self.quit_button.pack_forget()  # Hide the button temporarily
        ## this is a stupid thing to do, but i didnt want to recreate the whole app
        self.start_button.pack_forget()
        self.status_label.pack_forget()
        self.ear_label.pack_forget()
        ##
        self.title_label.pack()
        self.image_label.pack()
        self.start_button.pack()
        self.quit_app_button.pack()
        ##
        self.start_button.config(state="enabled")  # Re-enable the "Start Detection" button
        self.status_label.config(text="Status:")
        self.ear_label.config(text="EAR:")
        self.video_label.config(image=None)  # Clear the video feed
        self.video_label.image = None
        
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Bind the close event to on_closing method
        self.root.mainloop()
    
    def on_closing(self):
        self.running = False  # Stop the detection loop
        self.cap.release()    # Release the camera
        self.root.destroy()   # Close the Tkinter window

if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessApp(root)
    app.run()
