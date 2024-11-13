from ultralytics import YOLO
import cv2

# Load a YOLO11n PyTorch model
model = YOLO("yolov8n.pt")


# Run inference
results = model("https://ultralytics.com/images/bus.jpg")
annotated_img =results[0].plot()
img_resize=cv2.resize(annotated_img,(640,480))
cv2.imshow("Inferencia", img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()