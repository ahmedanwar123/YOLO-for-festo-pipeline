import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("best.pt")

# Load an image for testing
image_path = "dataset/1.jpg"  # Replace with the path to your test image
image = cv2.imread(image_path)

# Check if the image was loaded properly
if image is None:
    print("Error: Could not load image.")
    exit()

# Perform YOLOv8 inference on the image
results = model(image)

# Define colors for each class
colors = [(255, 0, 0), (0, 0, 255)]  # Blue for class 0, Red for class 1

# Draw detections on the image
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
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

# Display the resulting image
cv2.imshow("Inference Result", image)

# Wait for a key press to close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
