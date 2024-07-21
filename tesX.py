import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import pygame  # Import pygame

# Kamus untuk memetakan nilai numerik kelas ke nilai string kelas yang sesuai
class_mapping = {
    0: "tidak ngantuk",
    1: "ngantuk"
    # Tambahkan kelas lainnya sesuai dengan model Anda
}

class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, frame):
        results = self.model.predict(frame)
        return results

class YOLOGUI:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("YOLO Real-Time Detection")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.frame = ttk.Frame(root)
        self.frame.grid(row=0, column=0, padx=10, pady=10)

        self.canvas = tk.Canvas(self.frame, width=640, height=480)
        self.canvas.grid(row=0, column=0)

        self.label_frame = ttk.Frame(root)
        self.label_frame.grid(row=1, column=0, padx=10, pady=10)

        self.title_label = ttk.Label(self.label_frame, text="Deteksi", font=("Arial", 18))
        self.title_label.grid(row=0, column=0, pady=5)

        self.class_labels = []
        self.cap = cv2.VideoCapture(0)
        self.running = True

        self.last_open_time = time.time()  # Waktu terakhir mata terbuka
        self.closed_start_time = None  # Waktu mulai mata tertutup
        self.closed_duration = 3.0  # Durasi tertutupnya mata sebelum dianggap "ngantuk"

        pygame.mixer.init()  # Inisialisasi mixer pygame
        self.alarm_sound = pygame.mixer.Sound('alarm.mp3')  # Load sound

        self.update_frame()

    def update_frame(self):
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            self.root.after(10, self.update_frame)
            return

        # Perform prediction
        results = self.model.predict(frame)
        
        # Draw bounding boxes on the frame and update class labels
        detected_classes = set()
        eyes_closed = True

        for result in results:
            for box in result.boxes:
                conf = box.conf[0]
                if conf >= 0.6:  # Check if confidence score is above 60%
                    x1, y1, x2, y2 = box.xyxy[0].int().numpy()
                    cls = int(box.cls[0])
                    label_text = class_mapping.get(cls, "Unknown")
                    label = f"{label_text}: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if label_text == "tidak ngantuk":
                        eyes_closed = False

        current_time = time.time()
        if eyes_closed:
            if self.closed_start_time is None:
                self.closed_start_time = current_time
            if current_time - self.closed_start_time > self.closed_duration:
                detected_classes.add("ngantuk")
                self.alarm_sound.play()  # Play alarm sound
            else:
                detected_classes.add("tidak ngantuk")
                self.alarm_sound.stop()  # Stop alarm sound
        else:
            self.closed_start_time = None
            detected_classes.add("tidak ngantuk")
            self.alarm_sound.stop()  # Stop alarm sound

        # Hapus label-label kelas yang tidak terdeteksi pada frame ini
        for label in self.class_labels:
            if label.cget("text") not in detected_classes:
                label.destroy()
                self.class_labels.remove(label)

        # Tambahkan label kelas baru yang terdeteksi pada frame ini
        for cls in detected_classes:
            if cls not in [label.cget("text") for label in self.class_labels]:
                new_label = ttk.Label(self.label_frame, text=cls)
                new_label.grid(row=len(self.class_labels) + 1, column=0, pady=2, sticky="w")
                self.class_labels.append(new_label)

        # Convert the frame to an image that can be displayed in the GUI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk

        # Schedule the next frame update
        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.running = False
        self.cap.release()
        pygame.mixer.quit()  # Stop pygame mixer
        self.root.destroy()

if __name__ == "__main__":
    model_path = 'best.pt'  # Ganti dengan jalur model Anda
    yolo_model = YOLOModel(model_path)

    root = tk.Tk()
    gui = YOLOGUI(root, yolo_model)
    root.mainloop()
