from ultralytics import YOLO

model = YOLO("./models/best_done.pt")

print(model.model.names)
