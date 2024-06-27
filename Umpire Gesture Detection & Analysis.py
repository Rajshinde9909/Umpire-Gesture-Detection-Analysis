import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class GestureRecognitionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack(side=tk.LEFT)

        self.figure, self.ax = plt.subplots(figsize=(5, 4), tight_layout=True)
        self.graph_canvas = FigureCanvasTkAgg(self.figure, master=window)
        self.graph_widget = self.graph_canvas.get_tk_widget()

        self.btn_recognize = tk.Button(window, text="Recognize Gesture", width=20, command=self.recognize_gesture)
        self.btn_recognize.pack(padx=20, pady=10)

        self.btn_show_graph = tk.Button(window, text="Show Graph", width=20, command=self.show_graph)
        self.btn_show_graph.pack(padx=20, pady=10)

        self.label_result = tk.Label(window, text="")
        self.label_result.pack(pady=10)

        self.is_recognizing = False
        self.detected_gestures = []
        self.gesture_history = {"Out": [], "Four": [], "Six": [], "New Ball": [], "Wide Ball": []}

        self.hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.update()
        self.window.mainloop()

    def recognize_gesture(self):
        self.is_recognizing = True

    def process_gesture(self, hand_landmarks, frame_width, frame_height):
        x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame_width
        y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame_height

        out_region = (0, int(frame_height * 0.7), int(frame_width * 0.3), frame_height)
        four_region = (int(frame_width * 0.3), int(frame_height * 0.7), int(frame_width * 0.6), frame_height)
        six_region = (int(frame_width * 0.6), int(frame_height * 0.7), frame_width, frame_height)
        new_ball_region = (0, 0, int(frame_width * 0.5), int(frame_height * 0.3))
        wide_ball_region = (int(frame_width * 0.5), 0, frame_width, int(frame_height * 0.3))

        if (
            x > out_region[0]
            and y > out_region[1]
            and x < out_region[2]
            and y < out_region[3]
        ):
            return "Out"
        elif (
            x > four_region[0]
            and y > four_region[1]
            and x < four_region[2]
            and y < four_region[3]
        ):
            return "Four"
        elif (
            x > six_region[0]
            and y > six_region[1]
            and x < six_region[2]
            and y < six_region[3]
        ):
            return "Six"
        elif (
            x > new_ball_region[0]
            and y > new_ball_region[1]
            and x < new_ball_region[2]
            and y < new_ball_region[3]
        ):
            return "New Ball"
        elif (
            x > wide_ball_region[0]
            and y > wide_ball_region[1]
            and x < wide_ball_region[2]
            and y < wide_ball_region[3]
        ):
            return "Wide Ball"
        else:
            return None

    def show_graph(self):
        self.ax.clear()
        for gesture, values in self.gesture_history.items():
            self.ax.plot(values, label=gesture)

        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Count')
        self.ax.legend()
        self.ax.set_title('Gesture Recognition History')
        self.graph_canvas.draw()
        self.graph_widget.pack(side=tk.RIGHT)  # Show the graph canvas on the right side

    def update(self):
        ret, frame = self.vid.read()

        if self.is_recognizing:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    gesture_result = self.process_gesture(hand_landmarks, frame.shape[1], frame.shape[0])

                    if gesture_result:
                        self.detected_gestures.append(gesture_result)
                        self.gesture_history[gesture_result].append(len(self.gesture_history[gesture_result]))

        if self.detected_gestures:
            self.label_result["text"] = f"Gesture: {self.detected_gestures[-1]}"

        if ret:
            self.photo = self.convert_frame_to_photo(frame)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(10, self.update)

    def convert_frame_to_photo(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(image=Image.fromarray(frame))


root = tk.Tk()
app = GestureRecognitionApp(root, "Cricket Umpire Gesture Recognition")
