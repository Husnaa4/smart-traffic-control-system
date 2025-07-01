 #!/usr/bin/env python3
# pi_stream_and_capture.py - Capture ONLY when number plate is detected and still

import os
import cv2
import zmq
import time
import numpy as np
import base64
import paramiko
import logging
from datetime import datetime
from threading import Thread, Lock
from picamera2 import Picamera2

# === CONFIGURATION ===
PI_POSITION = "left"
MAC_IP = "192.168.4.20"
STREAM_PORT = 5555
COOLDOWN_SECONDS = 60
MAC_USERNAME = ""
MAC_PASSWORD = ""
IMAGE_DIR = "/home/pi/captured_images"
REMOTE_FOLDER = f"/Users/{MAC_USERNAME}/Desktop/received_images/"
PLATE_CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_russian_plate_number.xml"

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/pi/license_plate_detector.log')
    ]
)
logger = logging.getLogger(__name__)

class LicensePlateDetector:
    def __init__(self):
        self.socket = zmq.Context().socket(zmq.PUB)
        self.socket.bind(f"tcp://0.0.0.0:{STREAM_PORT}")

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (1280, 720)},
            lores={"size": (640, 480), "format": "YUV420"}
        )
        self.picam2.configure(config)

        os.makedirs(IMAGE_DIR, exist_ok=True)
        self.frame_lock = Lock()
        self.current_frame = None
        self.last_capture_time = 0
        # Load Haar cascade for plates
        if not os.path.exists(PLATE_CASCADE_PATH):
            raise FileNotFoundError("Haar cascade for plate detection not found.")
        self.plate_cascade = cv2.CascadeClassifier(PLATE_CASCADE_PATH)

    def start(self):
        logger.info(f"Starting plate detector for {PI_POSITION}...")
        self.picam2.start()
        time.sleep(2)

        Thread(target=self.stream_video, daemon=True).start()
        Thread(target=self.detect_and_capture, daemon=True).start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.picam2.stop()
            self.socket.close()
            logger.info("Stopped.")

    def stream_video(self):
        while True:
            try:
                frame = self.picam2.capture_array("main")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.frame_lock:
                    self.current_frame = frame_rgb.copy()

                _, buffer = cv2.imencode('.jpg', frame_rgb)
                encoded = base64.b64encode(buffer)
                self.socket.send_multipart([PI_POSITION.encode(), encoded])
                logger.info("[ZMQ] Frame sent to Mac")
                time.sleep(0.033)
            except Exception as e:
                logger.error(f"Streaming error: {e}")

    def detect_and_capture(self):
        while True:
            try:
                with self.frame_lock:
                    if self.current_frame is None:
                        continue
                    frame = self.current_frame.copy()

                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                plates = self.plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 20))

                if len(plates) > 0 and (time.time() - self.last_capture_time) > COOLDOWN_SECONDS:
                    x, y, w, h = plates[0]
                    plate_crop = frame[y:y+h, x:x+w]
                    logger.info("[DETECTED] Number plate detected and stable")
                    self.capture_and_transfer(frame)
                    self.last_capture_time = time.time()

                time.sleep(0.3)
            except Exception as e:
                logger.error(f"Detection error: {e}")

    def capture_and_transfer(self, frame):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{PI_POSITION}_{timestamp}.jpg"
            local_path = os.path.join(IMAGE_DIR, filename)
            cv2.imwrite(local_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            logger.info(f"[CAPTURED] {filename}")

            self.transfer_image(local_path, filename)
            if os.path.exists(local_path):
                os.remove(local_path)
        except Exception as e:
            logger.error(f"Capture error: {e}")

    def transfer_image(self, local_path, filename):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(MAC_IP, username=MAC_USERNAME, password=MAC_PASSWORD, timeout=10)

            with ssh.open_sftp() as sftp:
                remote_path = os.path.join(REMOTE_FOLDER, filename)
                sftp.put(local_path, remote_path)
                logger.info(f"[SENT] {filename} to Mac")
            ssh.close()
        except Exception as e:
            logger.error(f"SFTP transfer error: {e}")

if __name__ == "__main__":
    detector = LicensePlateDetector()
    detector.start()
def capture_and_transfer(self, frame):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{PI_POSITION}_{timestamp}.jpg"
        local_path = os.path.join(IMAGE_DIR, filename)
        cv2.imwrite(local_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        logger.info(f"[CAPTURED] {filename}")

        # Now, send the image to the comms hub
        self.send_image_to_comms_hub(local_path, filename)
       
        # Clean up the captured image after transfer
        if os.path.exists(local_path):
            os.remove(local_path)

    except Exception as e:
        logger.error(f"Capture error: {e}")

def send_image_to_comms_hub(self, local_path, filename):
    try:
        # Assuming you already have an MQTT topic for image upload (e.g., 'image/upload')
        image_url = "http://192.168.0.194:5000/upload_image"  # This is your comms hub URL for the image endpoint
        with open(local_path, 'rb') as img_file:
            files = {'image': img_file}
            response = requests.post(image_url, files=files)

            if response.status_code == 200:
                logger.info(f"[SENT] {filename} to Comms Hub")
            else:
                logger.error(f"[SENT ERROR] Status {response.status_code}")

    except Exception as e:
        logger.error(f"Error while sending image to Comms Hub: {e}")

