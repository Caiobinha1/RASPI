#neste programa estamos detectando rosto, atravez de uma foto sendo tirada usando picamera e inferida por um modelo treinado. Alem disso enquanto esta tirando a foto estamos acendendo um led conectado ao gpio 8
from picamera2 import Picamera2
from ultralytics import YOLO
import RPi.GPIO as GPIO
import cv2
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)
# Initialize Picamera2 and YOLO model globally to avoid reinitializing each time
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
model = YOLO('data_eu.pt')

def capture_image():
    """Captures an image from the camera and returns it as an RGB array."""
    picam2.start()
    img = picam2.capture_array()
    picam2.stop()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def detect_objects(img):
    results = model(img)
    annotated_img = results[0].plot()
    return annotated_img

if __name__ == "__main__":
    print("Type 'o' to capture an image and detect objects, 'q' to quit.")
    
    while True:
        # Prompt the user for input
        key = input("Press 'o' to capture, 'q' to quit: ").strip().lower()

        if key == 'o':
            # Start timing
            start_time = time.time()
            GPIO.output(8, GPIO.HIGH)
            # Capture image
            img = capture_image()

            # Calculate and print time taken for image capture
            capture_time = time.time() - start_time
            print(f"Time taken to capture image: {capture_time:.4f} seconds")
            GPIO.output(8,GPIO.LOW)
            # Detect objects
            annotated_img = detect_objects(img)
            img_resize = cv2.resize(annotated_img,(640,480))
            # Start timing for display
           # display_start_time = time.time()
            # Display the annotated image
            cv2.imshow("YOLOv8 Object Detection", img_resize)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()
            #display_time = time.time() - display_start_time
            #print(f"Time taken to render and display image: {display_time:.4f} seconds")
        elif key == 'q':
            # Exit loop
            print("Exiting...")
            break





