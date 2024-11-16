import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("best.pt")

# Initialize video capture from the default camera
cap = cv2.VideoCapture(2)  # Change the index if you have multiple cameras

# Check if the video capture is successful
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define colors for each class (e.g., blue and red)
colors = [(255, 0, 0), (0, 0, 255)]  # Blue for class 0, Red for class 1

# Run the camera feed
try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform YOLOv8 inference on the frame
        results = model(frame)

        # Draw detections on the frame
        for result in results:
            boxes = result.boxes  # Bounding boxes
            for box in boxes:
                # Extract the box coordinates and class
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates for the box
                confidence = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class ID, converted to int

                # Select color based on class ID
                color = colors[cls % 2]  # Use color 0 for class 0, color 1 for class 1

                # Draw the bounding box and label
                label = f"{model.names[cls]} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Display the resulting frame
        cv2.imshow("Camera Feed", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Process interrupted.")

# Release resources
cap.release()
cv2.destroyAllWindows()
