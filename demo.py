import numpy
import cv2 
from ultralytics import YOLOv10

model = YOLOv10('/home/kylai/llm/yolov10/runs/detect/train72/weights/last.pt')
# Open a live webcam feed
cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	if not ret:
		break
	results = model.predict(frame)
	img = results.plot()
	img = numpy.asarray(img)
	cv2.imshow('demo', img)
	if cv2.waitKey(1) == 27:
		break
cap.release()
cv2.destroyAllWindows()