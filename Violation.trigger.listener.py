#!/usr/bin/env python3
# pi_plate_detector.py - Simplified capture-only version with HTTP trigger

import os
import cv2
import zmq
import time
import numpy as np
import base64
import paramiko
import logging
import requests
from datetime import datetime
from threading import Thread, Lock
from flask import Flask, request, jsonify
from picamera2 import Picamera2

# === CONFIGURATION ===
PI_POSITION = "left"
MAC_IP = ""
STREAM_PORT = 5555
MAC_USERNAME = ""
MAC_PASSWORD = ""
IMAGE_DIR = "/home/pi/captured_images"
REMOTE_FOLDER = f"/Users/{MAC_USERNAME}/Desktop/received_images/"
HUB_UPLOAD_URL = "http://192.168.0.194:5000/upload_image"

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

# === FLASK APP ===
app_image = Flask(__name__)

@app_image.route("/capture", methods=["GET"])
def capture_image_api():
    direction = request.args.get("direction", "unknown")
    reason = request.args.get("reason", "manual")
    logger.info(f"[API] Received capture request: direction={direction}, reason={reason}")

    if direction != PI_POSITION and direction != "unknown":
        return jsonify({"status": "ignored", "message": f"This camera is for {PI_POSITION} side, not {direction}"}), 200

    try:
        with detector.frame_lock:
            if detector.current_frame is not None:
                frame = detector.current_frame.copy()
                capture_thread = Thread(target=detector.capture_and_send_to_hub, args=(frame, reason))
                capture_thread.daemon = True
                capture_thread.start()
                return jsonify({"status": "capturing", "message": f"Image capture initiated for {reason}"}), 200
            else:
                return jsonify({"status": "error", "message": "No frame available for capture"}), 503
    except Exception as e:
        logger.error(f"[API ERROR] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

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

    def start(self):
        logger.info(f"Starting detector for {PI_POSITION}...")
        self.picam2.start()
        time.sleep(2)

        Thread(target=self.stream_video, daemon=True).start()
        Thread(target=lambda: app_image.run(host='0.0.0.0', port=5001, debug=False), daemon=True).start()

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
                time.sleep(0.033)
            except Exception as e:
                logger.error(f"Streaming error: {e}")

    def capture_and_send_to_hub(self, frame, reason):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{PI_POSITION}_{reason}_{timestamp}.jpg"
            local_path = os.path.join(IMAGE_DIR, filename)

            cv2.imwrite(local_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            logger.info(f"[CAPTURED] {filename}")

            with open(local_path, 'rb') as img_file:
                files = {'image': (filename, img_file, 'image/jpeg')}
                data = {'reason': reason, 'side': PI_POSITION}
                response = requests.post(HUB_UPLOAD_URL, files=files, data=data)

                if response.status_code == 200:
                    logger.info(f"[SENT] {filename} to hub successfully")
                else:
                    logger.error(f"Failed to send image to hub: {response.status_code}")

            self.transfer_image(local_path, filename)
            if os.path.exists(local_path):
                os.remove(local_path)

        except Exception as e:
            logger.error(f"Capture and send error: {e}")

    def transfer_image(self, local_path, filename):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(MAC_IP, username=MAC_USERNAME, password=MAC_PASSWORD, timeout=10)
            with ssh.open_sftp() as sftp:
                remote_path = os.path.join(REMOTE_FOLDER, filename)
                sftp.put(local_path, remote_path)
                logger.info(f"[SFTP] {filename} sent to Mac")
            ssh.close()
        except Exception as e:
            logger.error(f"SFTP transfer error: {e}")

if __name__ == "__main__":
    detector = LicensePlateDetector()
    detector.start()
