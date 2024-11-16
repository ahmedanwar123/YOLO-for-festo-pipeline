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

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.6

# Object counting variables
detected_objects = {}  # Dictionary to track objects between frames
object_count = 0  # Counter for objects in the current frame

# File path for output
output_file = "object_count.txt"


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Intersection area
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    # Union area
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


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

        # Initialize a new dictionary for the current frame
        current_frame_objects = {}

        # Process detected objects
        for result in results:
            boxes = result.boxes  # Bounding boxes
            for box in boxes:
                # Extract box coordinates, confidence, and class ID
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates for the box
                confidence = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class ID, converted to int

                # Only consider detections above the confidence threshold
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Draw bounding box and label
                    color = colors[cls % 2]
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

                    # Add object to the current frame's dictionary
                    current_frame_objects[(x1, y1, x2, y2)] = cls

        # Set object_count to the number of objects in the current frame
        object_count = len(current_frame_objects)

        # Write object count to file
        with open(output_file, "w") as file:
            file.write(f"Object Count: {object_count}\n")

        # Update detected objects for the next frame
        detected_objects = current_frame_objects.copy()

        # Display the count on the frame
        cv2.putText(
            frame,
            f"Object Count: {object_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
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
