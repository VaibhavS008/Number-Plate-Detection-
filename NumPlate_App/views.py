#Using Yolov5
from django.shortcuts import render
from django.conf import settings
import cv2
import os
import torch
from PIL import Image
import numpy as np
import pytesseract
from .models import Image
from ultralytics import YOLO

def Home(request):
    return render(request, "Home.html", {})

# Load YOLOv5 model 
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

def Detect_Image(request):
    text = ""
    img_file = None
    result_img_name = ""
    cropped_img_name = ""

    if request.method == "POST":
        try:
            # Check if image is uploaded
            if 'img' not in request.FILES:
                return render(request, "Detect_Image.html", {"error": "No image uploaded", "detected_text": ""})
            
            img_file = request.FILES['img']
            
            # Validate file type
            if not img_file.content_type.startswith('image/'):
                return render(request, "Detect_Image.html", {"error": "Uploaded file is not an image", "detected_text": ""})

            # Save the uploaded image temporarily
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, img_file.name)

            with open(temp_path, 'wb+') as destination:
                for chunk in img_file.chunks():
                    destination.write(chunk)

            # Load image for YOLO detection
            img = cv2.imread(temp_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # YOLOv5 Detection
            results = yolo_model(img_rgb)
            results.print()
            detections = results.pandas().xyxy[0]

            if detections.empty:
                return render(request, "Detect_Image.html", {"error": "No license plate detected", "detected_text": ""})

            # Crop and save image with bounding box
            x1, y1, x2, y2 = map(int, detections.iloc[0][['xmin', 'ymin', 'xmax', 'ymax']])
            cropped_plate = img[y1:y2, x1:x2]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            result_img_name = 'result_' + img_file.name
            cropped_img_name = 'cropped_' + img_file.name

            result_img_path = os.path.join(temp_dir, result_img_name)
            cropped_img_path = os.path.join(temp_dir, cropped_img_name)

            cv2.imwrite(result_img_path, img)
            cv2.imwrite(cropped_img_path, cropped_plate)

            # OCR using tesseract
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            text = pytesseract.image_to_string(cropped_plate, config='--psm 11')

            return render(request, "Detect_Image.html", {
                "detected_text": text.strip(),
                "original_img": os.path.join(settings.MEDIA_URL, 'temp', img_file.name),
                "result_img": os.path.join(settings.MEDIA_URL, 'temp', result_img_name),
                "cropped_img": os.path.join(settings.MEDIA_URL, 'temp', cropped_img_name),
                "error": None
            })

        except Exception as e:
            print(f"Error: {str(e)}")
            return render(request, "Detect_Image.html", {
                "error": f"Error processing image: {str(e)}",
                "detected_text": "",
                "original_img": None,
                "result_img": None,
                "cropped_img": None
            })

    return render(request, "Detect_Image.html", {
        "detected_text": "",
        "original_img": None,
        "result_img": None,
        "cropped_img": None,
        "error": None
    })

def Real_Time(request):
    if request.method == "POST":
        width, height = 640, 480  # Set resolution for video capture
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Start video stream from default webcam
        video_stream = cv2.VideoCapture(0)
        video_stream.set(3, width)
        video_stream.set(4, height)
        video_stream.set(10, 150)  # Brightness setting

        while True:
            ret, frame = video_stream.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLOv5 Detection
            results = yolo_model(img_rgb)
            results.print()  # To show results in the console
            detections = results.pandas().xyxy[0]  # Extract bounding boxes

            if not detections.empty:
                # Assume the first detection is the license plate
                x1, y1, x2, y2 = map(int, detections.iloc[0][['xmin', 'ymin', 'xmax', 'ymax']])
                cropped_plate = frame[y1:y2, x1:x2]

                # Preprocessing for OCR
                kernel = np.ones((1, 1), np.uint8)
                cropped_plate = cv2.dilate(cropped_plate, kernel, iterations=1)
                cropped_plate = cv2.erode(cropped_plate, kernel, iterations=1)
                gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                _, binary_plate = cv2.threshold(gray_plate, 127, 255, cv2.THRESH_BINARY)

                # OCR recognition
                text = pytesseract.image_to_string(binary_plate)
                text = ''.join(filter(str.isalnum, text))

                # Drawing bounding box and text on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (51, 51, 255), 2)
                cv2.rectangle(frame, (x1 - 1, y1 - 40), (x2 + 1, y1), (51, 51, 255), -1)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Display live video with detections
            cv2.imshow("Live Detection", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                # Save the current frame as an image
                cv2.imwrite("C:/Python Projects/Number_Plate_Detection/images/saved_frame.jpg", frame)
                cv2.rectangle(frame, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                cv2.imshow("Live Detection", frame)
            elif key & 0xFF == ord('q'):
                break

        video_stream.release()
        cv2.destroyAllWindows()
        return render(request, "Real_Time.html", {})
    else:
        return render(request, "Real_Time.html", {})


def Detect_Video(request):
    if request.method == "POST":
        video = request.FILES['video']
        print(video)

        # Save uploaded video to a known location
        save_path = os.path.join(settings.MEDIA_ROOT, 'videos')
        os.makedirs(save_path, exist_ok=True)

        video_filename = str(video.name)
        full_video_path = os.path.join(save_path, video_filename)

        with open(full_video_path, 'wb+') as destination:
            for chunk in video.chunks():
                destination.write(chunk)

        # Set up video capture and processing
        frameWidth, frameHeight = 640, 480
        pytesseract.pytesseract.tesseract_cmd = getattr(settings, 'TESSERACT_PATH', r'C:\Program Files\Tesseract-OCR\tesseract.exe')
        
        cap = cv2.VideoCapture(full_video_path)
        cap.set(3, frameWidth)
        cap.set(4, frameHeight)
        cap.set(10, 150)

        count = 0
        detected_text = []

        while True:
            success, img = cap.read()
            if not success or img is None:
                print("No more frames or failed to read video.")
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Run YOLO object detection
            results = yolo_model(img_rgb)
            results.print()
            detections = results.pandas().xyxy[0]  # Convert results to pandas DataFrame
            
            # Iterate over detected license plates 
            for _, row in detections.iterrows():
                if row['name'] == 'license_plate': 
                    x1, y1, x2, y2 = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']])
                    cropped_plate = img[y1:y2, x1:x2]

                    # Preprocess the cropped license plate for OCR
                    gray_cropped = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                    _, binary_cropped = cv2.threshold(gray_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Run OCR to detect text
                    license_plate_number = pytesseract.image_to_string(binary_cropped, config='--psm 11')

                    # Draw rectangle around detected plate and display recognized text
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, license_plate_number.strip(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                    detected_text.append(license_plate_number.strip())

            # Display results
            cv2.imshow("Detected License Plates", img)

            # Handle frame-saving and exiting via keys
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return render(request, "Detect_Video.html", {"detected_text": "\n".join(detected_text)})

    return render(request, "Detect_Video.html", {})



#Yolo v8
yolo_model_v8 = YOLO('best.pt')


def detect_image_v8(request):
    text = ""
    img_file = None
    result_img_name = ""
    cropped_img_name = ""

    if request.method == "POST":
        try:
            if 'img' not in request.FILES:
                return render(request, "Detect_Image.html", {"error": "No image uploaded", "detected_text": ""})
            
            img_file = request.FILES['img']
            if not img_file.content_type.startswith('image/'):
                return render(request, "Detect_Image.html", {"error": "Uploaded file is not an image", "detected_text": ""})

            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, img_file.name)

            with open(temp_path, 'wb+') as destination:
                for chunk in img_file.chunks():
                    destination.write(chunk)

            img = cv2.imread(temp_path)
            results = yolo_model_v8(img)
            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                return render(request, "Detect_Image.html", {"error": "No license plate detected", "detected_text": ""})

            b = boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = b
            cropped_plate = img[y1:y2, x1:x2]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            result_img_name = 'result_' + img_file.name
            cropped_img_name = 'cropped_' + img_file.name
            result_img_path = os.path.join(temp_dir, result_img_name)
            cropped_img_path = os.path.join(temp_dir, cropped_img_name)

            cv2.imwrite(result_img_path, img)
            cv2.imwrite(cropped_img_path, cropped_plate)

            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            text = pytesseract.image_to_string(cropped_plate, config='--psm 11')

            return render(request, "Detect_Image.html", {
                "detected_text": text.strip(),
                "original_img": os.path.join(settings.MEDIA_URL, 'temp', img_file.name),
                "result_img": os.path.join(settings.MEDIA_URL, 'temp', result_img_name),
                "cropped_img": os.path.join(settings.MEDIA_URL, 'temp', cropped_img_name),
                "error": None
            })

        except Exception as e:
            print(f"Error: {str(e)}")
            return render(request, "Detect_Image.html", {
                "error": f"Error processing image: {str(e)}",
                "detected_text": "",
                "original_img": None,
                "result_img": None,
                "cropped_img": None
            })

    return render(request, "Detect_Image.html", {
        "detected_text": "",
        "original_img": None,
        "result_img": None,
        "cropped_img": None,
        "error": None
    })


def real_time_detection_v8(request):
    if request.method == "POST":
        width, height = 640, 480
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        cap = cv2.VideoCapture(0)
        cap.set(3, width)
        cap.set(4, height)
        cap.set(10, 150)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model_v8(frame)
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                b = boxes.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = b
                cropped_plate = frame[y1:y2, x1:x2]

                gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

                text = pytesseract.image_to_string(binary, config='--psm 11')
                text = ''.join(filter(str.isalnum, text))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Live Detection V8", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                cv2.imwrite("C:/Python Projects/Number_Plate_Detection/images/saved_frame.jpg", frame)
            elif key & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return render(request, "Real_Time.html", {})
    else:
        return render(request, "Real_Time.html", {})


def detect_video_v8(request):
    if request.method == "POST":
        video = request.FILES['video']
        save_path = os.path.join(settings.MEDIA_ROOT, 'videos')
        os.makedirs(save_path, exist_ok=True)

        video_filename = str(video.name)
        full_video_path = os.path.join(save_path, video_filename)

        with open(full_video_path, 'wb+') as destination:
            for chunk in video.chunks():
                destination.write(chunk)

        pytesseract.pytesseract.tesseract_cmd = getattr(settings, 'TESSERACT_PATH', r'C:\Program Files\Tesseract-OCR\tesseract.exe')
        cap = cv2.VideoCapture(full_video_path)

        detected_text = []

        while True:
            success, frame = cap.read()
            if not success:
                break

            results = yolo_model_v8(frame)
            boxes = results[0].boxes

            if boxes is not None:
                for box in boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box.astype(int)
                    cropped = frame[y1:y2, x1:x2]

                    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

                    text = pytesseract.image_to_string(binary, config='--psm 11')
                    cleaned = text.strip()

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, cleaned, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    detected_text.append(cleaned)

            cv2.imshow("Video Detection V8", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return render(request, "Detect_Video.html", {"detected_text": "\n".join(detected_text)})

    return render(request, "Detect_Video.html", {})