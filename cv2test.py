from picamera2 import Picamera2
import cv2
from ultralytics import YOLO

#Tirar a foto
picam2 =Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()
picam2.capture_file("test.jpg")
picam2.stop()

#Abrir a foto usando cv2
image_path="test.jpg"
image =cv2.imread(image_path)
img_resized = cv2.resize(image,(640,480))


cv2.imshow("teste", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
