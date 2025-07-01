#!/usr/bin/env python3
"""
License Plate Detection and Processing System
"""

import os
import cv2
import time
import csv
import json
import requests
import numpy as np
import easyocr
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shutil
import logging
import re
import base64
processed_files = set()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
RECEIVED_FOLDER = os.path.expanduser("~/Desktop/received_images/")
PROCESSED_FOLDER = os.path.expanduser("~/Desktop/processed_images/")
VIOLATIONS_FOLDER = os.path.expanduser("~/Desktop/violations/")
CSV_PATH = os.path.expanduser("~/Desktop/plate_detections.csv")
COMM_HUB_URL = "http://192.168.15.55:5000/validate_plate"  # Updated communication hub URL
COMM_HUB_DENIED_URL = "http://192.168.15.55:5000/upload_image"  # New URL for denied access
MAX_RETRY_COUNT = 3
RETRY_DELAY = 5  # seconds
MIN_PLATE_CHARS = 5
MAX_PLATE_CHARS = 10

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)
# Ensure all required directories exist
for folder in [RECEIVED_FOLDER, PROCESSED_FOLDER, VIOLATIONS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize CSV if needed
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Timestamp', 'Filename', 'Plate Number', 'Vehicle Type', 'Side (left/right)', 'Confidence Score'])

def preprocess_image(image):
    """
    Improve image processing for better license plate detection
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to connect edge components
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    return dilated, enhanced


def find_license_plate(processed_image, original_image, enhanced_image):
    """
    Improved license plate detection with multiple approaches
    """
    result_image = original_image.copy()
    height, width = original_image.shape[:2]
    
    # First approach: contour detection
    contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    potential_plates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:  # Filter small contours
            continue
            
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if 4 <= len(approx) <= 6:  # Look for rectangular shapes
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            
            # Most license plates have aspect ratios between 2 and 5
            if 1.5 <= aspect_ratio <= 6:
                # Calculate relative size and position
                rel_size = (w * h) / (width * height)
                rel_position = y / height
                
                # Score the potential plate
                score = area * (0.3 + 0.2 * (1 - rel_position))  # Prefer larger areas, lower in the image
                potential_plates.append((x, y, w, h, score))
    
    # Sort by score
    potential_plates.sort(key=lambda x: x[4], reverse=True)
    
    # If we found some plates with the contour method
    if potential_plates:
        x, y, w, h = potential_plates[0][:4]
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plate_region = original_image[y:y+h, x:x+w]
        return result_image, plate_region, (x, y, w, h)
    
    # Second approach: try detecting the whole image as a fallback
    return result_image, original_image, None


def clean_plate_text(text):
    """
    Clean and format the detected license plate text
    """
    if not text:
        return None      

def extract_plate_text(plate_image):
    """
    Enhanced OCR for license plate text extraction
    """
    if plate_image is None:
        return None, 0
        
    try:
        # Try OCR with standard settings
        results = reader.readtext(plate_image)
        
        if not results:
            # Try with different preprocessings
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) > 2 else plate_image
            
            # Apply various preprocessing techniques
            preprocessing_methods = [
                lambda img: img,  # Original
                lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # Otsu threshold
                lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),  # Adaptive threshold
                lambda img: cv2.equalizeHist(img)  # Histogram equalization
            ]
            
            for preprocess in preprocessing_methods:
                processed = preprocess(gray)
                results = reader.readtext(processed)
                if results:
                    break
        
        # Process results
        if results:
            # Sort text by confidence
            results.sort(key=lambda x: x[2], reverse=True)
            
            # Get the most confident text
            plate_text = results[0][1]
            confidence = results[0][2]
            
            # Clean and normalize
            plate_text = clean_plate_text(plate_text)
            
            if plate_text and MIN_PLATE_CHARS <= len(plate_text) <= MAX_PLATE_CHARS + 5:  # Increased max length to account for spaces
                return plate_text, confidence
                
    except Exception as e:
        logger.error(f"OCR error: {e}")
        
    return None, 0


def determine_side(filename):
    filename_lower = filename.lower()
    if "left" in filename_lower:
        return "left"
    elif "right" in filename_lower:
        return "right"
    return "Unknown"

def validate_plate(plate_text, direction):
    if not plate_text:
        return "deny"
    payload = {
        "plate": plate_text,
        "direction": direction
    }
    try:
        response = requests.post(COMM_HUB_URL, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            return result.get("result", "deny")  # Get the 'result' key which is 'allow' or 'deny'
        else:
            logger.error(f"Error from comm hub: {response.status_code}")
    except Exception as e:
        logger.error(f"Comm hub connection failed: {e}")
    return "deny"

def send_denied_image(image_path, plate_text, side):
    """
    Send the image file (as multipart/form-data) to the communication subsystem when access is denied.
    """
    try:
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(COMM_HUB_DENIED_URL, files=files)

        if response.status_code == 200:
            logger.info(f"Successfully sent denied image for plate {plate_text} to communication hub")
            return True
        else:
            logger.error(f"Failed to send denied image: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error sending denied image: {e}")
        return False


def log_detection(filename, plate_text, side, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vehicle_type = ""
    if not plate_text:
        plate_text = "Not Found"
        confidence = 0
    with open(CSV_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, filename, plate_text, vehicle_type, side, f"{confidence:.2f}"])


def process_image(file_path):
    filename = os.path.basename(file_path)
    

    for _ in range(MAX_RETRY_COUNT):
        try:
            image = cv2.imread(file_path)
            if image is None:
                logger.warning(f"Could not read {filename}, retrying...")
                time.sleep(RETRY_DELAY)
                continue

            side = determine_side(filename)
            preprocessed, enhanced = preprocess_image(image)
            result_image, plate_region, _ = find_license_plate(preprocessed, image, enhanced)

            plate_text, confidence = extract_plate_text(plate_region)

            if not plate_text:
                height, width = image.shape[:2]
                regions = [
                    image,
                    image[height//3:2*height//3, width//4:3*width//4],
                    image[height//4:3*height//4, :]
                ]
                for region in regions:
                    if region.size == 0:
                        continue
                    plate_text, confidence = extract_plate_text(region)
                    if plate_text:
                        break

            if plate_text:
                logger.info(f"[DETECTED] Plate: {plate_text} | Side: {side} | Confidence: {confidence:.2f}")
            else:
                logger.warning(f"[WARNING] No valid plate detected in {filename}")

            processed_path = os.path.join(PROCESSED_FOLDER, filename)
            cv2.imwrite(processed_path, result_image)

            log_detection(filename, plate_text, side, confidence)

            

            # Decide access based on plate validation
            if plate_text:
                access_result = validate_plate(plate_text, side)
                if access_result == "allow":
                    logger.info(f"Access granted for {plate_text}")
                else:
                    logger.info(f"Access denied for {plate_text}. Sending to comm hub.")
                    if send_denied_image(file_path, plate_text, side):
                        logger.info("Denied image successfully sent to comm hub.")
                    else:
                        logger.warning("Failed to send denied image. Saving to violations folder.")
                        try:
                            violation_path = os.path.join(VIOLATIONS_FOLDER, filename)
                            shutil.copy(file_path, violation_path)
                        except Exception as e:
                            logger.error(f"Failed to save to violations: {e}")
            else:
                logger.info("No valid plate found. Sending unknown image to comm hub.")
                if send_denied_image(file_path, None, side):
                    logger.info("Unknown plate image successfully sent to comm hub.")
                else:
                    logger.warning("Failed to send unknown image. Saving to violations folder.")
                    try:
                        violation_path = os.path.join(VIOLATIONS_FOLDER, filename)
                        shutil.copy(file_path, violation_path)
                    except Exception as e:
                        logger.error(f"Failed to save to violations: {e}")

            # Final cleanup: delete the original image file
            if os.path.exists(file_path):
                try:
                    #os.remove(file_path)
                    logger.info(f"Deleted original image file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete file {file_path}: {e}")

            processed_files.add(filename)
            return True

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            time.sleep(RETRY_DELAY)

    logger.error(f"Failed to process {filename} after retries")
    return False


class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and (event.src_path.lower().endswith(('.jpg', '.jpeg'))):
            logger.info(f"New image detected: {event.src_path}")
            time.sleep(1)  # Small delay to ensure file is completely written
            process_image(event.src_path)

    def on_moved(self, event):
        if not event.is_directory and (event.dest_path.lower().endswith(('.jpg', '.jpeg'))):
            if os.path.dirname(event.dest_path) == RECEIVED_FOLDER:
                logger.info(f"Image moved: {event.dest_path}")
                process_image(event.dest_path)


def run_plate_detection_on_single_image(image_path):
    """
    Process a single image file (useful for testing)
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read {image_path}")
            return None
            
        filename = os.path.basename(image_path)
        logger.info(f"Processing image: {filename}")
        
        preprocessed, enhanced = preprocess_image(image)
        result_image, plate_region, _ = find_license_plate(preprocessed, image, enhanced)
        plate_text, confidence = extract_plate_text(plate_region)
        
        # If no plate detected in the region, try the whole image
        if not plate_text:
            plate_text, confidence = extract_plate_text(image)
            
        if plate_text:
            logger.info(f"[DETECTED] Plate: {plate_text} | Confidence: {confidence:.2f}")
            # Draw the license plate text on the image
            cv2.putText(result_image, plate_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            logger.warning(f"No valid plate detected in {filename}")
            
        return plate_text, result_image
            
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None


def main():
    logger.info("Starting License Plate Detection System")
    
    # Process existing files in RECEIVED_FOLDER
    for filename in os.listdir(RECEIVED_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg')):  # Accept both JPG and JPEG
            file_path = os.path.join(RECEIVED_FOLDER, filename)
            logger.info(f"Processing existing file: {filename}")
            process_image(file_path)

    # Start watching for new files
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path=RECEIVED_FOLDER, recursive=False)
    observer.start()

    logger.info(f"Monitoring {RECEIVED_FOLDER} for new images...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    logger.info("License Plate Detection System stopped")


if __name__ == "__main__":
    main()
