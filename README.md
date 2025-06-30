# Smart Traffic Control System 

This project detects and manages vehicle access using computer vision and microcontrollers. It includes:

## Structure
- `mac_controller/`: Image processing and license plate detection using EasyOCR and YOLOv8
- `pi_detector/`: Raspberry Pi module for automatic detection and image capture
- `pi_trigger_listener/`: Trigger-based capture and image forwarding

## Technologies
- Raspberry Pi, ESP32, Python, Flask, Streamlit
- OpenCV, EasyOCR, YOLOv8, ZMQ, paramiko, watchdog

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run each script in its respective module on the appropriate device

## Features
- Real-time plate detection & classification
- Smart gate logic & side-based control
- Image logging and violation reporting

For full deployment instructions, see `/docs/`.
