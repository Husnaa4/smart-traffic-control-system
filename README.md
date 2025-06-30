# Smart Traffic Control System

This project is a real-time vehicle access control system using AI, embedded systems, and microcontrollers. Built as a final-year university project, it uses license plate detection, Raspberry Pi devices, and automated boom gate logic to control traffic in a single-lane road.

---

## Project Structure

- `mac_controller/`: Processes incoming images, detects and reads license plates using EasyOCR and OpenCV
- `pi_detector/`: Raspberry Pi module that captures images when a plate is detected
- `pi_trigger_listener/`: REST API + Flask service to trigger manual or remote image capture
- `docs/`: For reports, diagrams, and setup guides

---

## Technologies Used

- Python, OpenCV, EasyOCR, YOLOv8
- Flask, Streamlit, MQTT, ZMQ
- Raspberry Pi, ESP32, Paramiko
- Hardware: Ultrasonic sensors, MPU6050, LCD displays

---

## Features

- Real-time number plate detection (ND format handling)
- Smart decision logic for one-lane boom gate control
- Automatic and manual image capture
- MQTT & HTTP communication between subsystems
- Violation logging with CSV export and screenshot capture
- Compliant with South Africa’s POPIA requirements

---

## Setup

```bash
# Clone the repo
git clone https://github.com/Husnaa4/smart-traffic-control-system.git
cd smart-traffic-control-system

# Install dependencies
pip install -r requirements.txt

