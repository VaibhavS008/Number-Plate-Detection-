import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
results = model("D:/Project/Number_Plate_Detection-Main/media/image1.jpg")
print(results.pandas().xyxy[0])