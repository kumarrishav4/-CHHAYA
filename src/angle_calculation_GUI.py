# import cv2
# import mediapipe as mp
# import pandas as pd
# import tkinter as tk
# from tkinter import filedialog
# import math

# # Function to calculate angles between three points (p1, p2, p3) in 3D space
# def calculate_3d_angle(p1, p2, p3):
#     a = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
#     b = math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2 + (p2[2] - p3[2]) ** 2)
#     c = math.sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2 + (p3[2] - p1[2]) ** 2)
#     angle = math.degrees(math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))
#     return angle

# # Function to draw lines and calculate angles between points in 3D
# def lines_and_box_3d(image, points):
#     fingers = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
#     angles = []

#     for finger in fingers:
#         # Calculate each angle along the finger
#         for i in range(len(finger) - 2):
#             p1 = points[finger[i]]
#             p2 = points[finger[i + 1]]
#             p3 = points[finger[i + 2]]
#             angle = calculate_3d_angle(p1, p2, p3)
#             angles.append(angle)
#             cv2.putText(image, f'{angle:.1f}', (p2[0], p2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         # Calculate angle with respect to the palm point (bottom-most joint calculation)
#         palm_point = points[finger[0]]
#         bottom_joint = points[finger[1]]
#         next_joint = points[finger[2]]
#         bottom_angle = calculate_3d_angle(palm_point, bottom_joint, next_joint)
#         angles.append(bottom_angle)

#         # Draw lines between points
#         for i in range(len(finger) - 1):
#             point1 = points[finger[i]]
#             point2 = points[finger[i + 1]]
#             cv2.line(image, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 255, 0), 2)

#     return angles

# class HandTrackerApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Hand Tracker")

#         self.start_button = tk.Button(root, text="Start", command=self.start_tracking)
#         self.start_button.pack(pady=10)

#         self.stop_button = tk.Button(root, text="Stop", command=self.stop_tracking, state=tk.DISABLED)
#         self.stop_button.pack(pady=10)

#         self.cap = None
#         self.csv_data = []
#         self.running = False

#     def start_tracking(self):
#         self.start_button.config(state=tk.DISABLED)
#         self.stop_button.config(state=tk.NORMAL)
#         self.running = True
#         self.capture_and_track()

#     def stop_tracking(self):
#         self.running = False
#         self.stop_button.config(state=tk.DISABLED)
#         self.start_button.config(state=tk.NORMAL)
#         self.save_to_csv()

#     def capture_and_track(self):
#         self.cap = cv2.VideoCapture(0)
#         mp_hands = mp.solutions.hands
#         hands = mp_hands.Hands()

#         while self.running and self.cap.isOpened():
#             ret, frame = self.cap.read()
#             if not ret:
#                 break

#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(rgb_frame)

#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     landmarks = [
#                         (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]), lm.z)
#                         for lm in hand_landmarks.landmark
#                     ]
#                     angles = lines_and_box_3d(frame, landmarks)
#                     self.csv_data.append(angles)

#             cv2.imshow("Hand Tracking with Box and Angles", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         self.cap.release()
#         cv2.destroyAllWindows()

#     def save_to_csv(self):
#         if self.csv_data:
#             # Column names updated to include bottom joint angles per finger
#             column_names = [f'Finger_{i}_Joint_{j}_Angle' for i in range(1, 6) for j in range(1, 4)]
#             bottom_joint_angles = [f'Finger_{i}_Bottom_Angle' for i in range(1, 6)]
#             column_names.extend(bottom_joint_angles)

#             df = pd.DataFrame(self.csv_data, columns=column_names)

#             file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])

#             if file_path:
#                 df.to_csv(file_path, index=False)
#             else:
#                 print("Save operation canceled.")
#         else:
#             print("No data to save.")

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = HandTrackerApp(root)
#     root.mainloop()
import cv2
import mediapipe as mp
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import math

# Function to calculate angles between three points (p1, p2, p3) in 3D space
def calculate_3d_angle(p1, p2, p3):
    a = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
    b = math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2 + (p2[2] - p3[2]) ** 2)
    c = math.sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2 + (p3[2] - p1[2]) ** 2)
    angle = math.degrees(math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))
    return angle

# Function to calculate the angle between a point and the vertical y-axis in 3D space
def calculate_angle_with_y_axis(p1, p2):
    # Vector between the two points
    vector = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    
    # Vertical axis is represented by the vector (0, 1, 0) in y-axis direction
    vertical_axis = (0, 1, 0)
    
    # Dot product of the vector and the vertical axis
    dot_product = vector[1]  # Only the y-component affects the dot product since the vertical is along y-axis
    
    # Magnitude of the vector and vertical axis
    magnitude_vector = math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    magnitude_vertical_axis = 1  # Vertical axis vector magnitude is 1
    
    # Angle between the vector and the vertical axis
    angle = math.degrees(math.acos(dot_product / magnitude_vector))
    
    return angle

# Function to draw lines and calculate angles between points in 3D
def lines_and_box_3d(image, points):
    fingers = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
    angles = []

    for finger in fingers:
        # Calculate each angle along the finger
        for i in range(len(finger) - 2):
            p1 = points[finger[i]]
            p2 = points[finger[i + 1]]
            p3 = points[finger[i + 2]]
            angle = calculate_3d_angle(p1, p2, p3)
            angles.append(angle)
            cv2.putText(image, f'{angle:.1f}', (p2[0], p2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate bottom joint angle with respect to vertical y-axis
        palm_point = points[finger[0]]
        bottom_joint = points[finger[1]]
        bottom_angle = calculate_angle_with_y_axis(palm_point, bottom_joint)
        angles.append(bottom_angle)

        # cv2.putText(image, f'{bottom_angle:.1f}', (bottom_joint[0], bottom_joint[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw lines between points
        for i in range(len(finger) - 1):
            point1 = points[finger[i]]
            point2 = points[finger[i + 1]]
            cv2.line(image, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 255, 0), 2)

    return angles

class HandTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Tracker")

        self.start_button = tk.Button(root, text="Start", command=self.start_tracking)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_tracking, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.cap = None
        self.csv_data = []
        self.running = False

    def start_tracking(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.running = True
        self.capture_and_track()

    def stop_tracking(self):
        self.running = False
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        self.save_to_csv()

    def capture_and_track(self):
        self.cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [
                        (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]), lm.z)
                        for lm in hand_landmarks.landmark
                    ]
                    angles = lines_and_box_3d(frame, landmarks)
                    self.csv_data.append(angles)

            cv2.imshow("Hand Tracking with Box and Angles", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def save_to_csv(self):
        if self.csv_data:
            # Column names updated to include bottom joint angles per finger
            column_names = [f'Finger_{i}_Joint_{j}_Angle' for i in range(1, 6) for j in range(1, 4)]
            bottom_joint_angles = [f'Finger_{i}_Bottom_Angle' for i in range(1, 6)]
            column_names.extend(bottom_joint_angles)

            df = pd.DataFrame(self.csv_data, columns=column_names)

            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])

            if file_path:
                df.to_csv(file_path, index=False)
            else:
                print("Save operation canceled.")
        else:
            print("No data to save.")

if __name__ == "__main__":
    root = tk.Tk()
    app = HandTrackerApp(root)
    root.mainloop()
