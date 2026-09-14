import cv2
import threading
from ultralytics import YOLO
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|framedrop;1"

class LiveStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

model = YOLO("best.pt")
model.to("cuda")

stream = LiveStream("rtsp://localhost:8554/drone?tcp")

while True:
    frame = stream.read()
    if frame is None:
        continue

    results = model(
        frame,
        imgsz=(704, 1280),
        device="cuda",
        verbose=False
    )

    annotated = results[0].plot()
    cv2.imshow("Drone Feed", annotated)
    if cv2.waitKey(1) == ord("q"):
        break

stream.stop()
cv2.destroyAllWindows()