import time
import cv2
import os

# File path for the output text file
input_file = "object_count.txt"

# Define the directory where you want to save the photos
# This will work for both Windows and Linux
if os.name == "nt":  # For Windows
    save_dir = "C:\\path\\to\\your\\folder"  # Use double backslashes for Windows paths
else:  # For Linux and macOS
    save_dir = "/path/to/your/folder"  # Forward slashes for Linux and macOS

# Ensure the directory exists, if not, create it
os.makedirs(save_dir, exist_ok=True)

# Start capturing images
cap = cv2.VideoCapture(
    0
)  # Open the default camera (you can change the index if needed)

try:
    photo_counter = 0  # Initialize a counter for naming photos

    while True:
        # Open the file and read the last line
        with open(input_file, "r") as file:
            lines = file.readlines()
            if lines:
                # Get the most recent object count line
                latest_line = lines[-1]
                print(f"Latest Object Count: {latest_line.strip()}")

        # Capture a frame (photo) from the camera
        ret, frame = cap.read()
        if ret:
            # Construct a filename based on the counter
            filename = os.path.join(save_dir, f"photo_{photo_counter}.jpg")

            # Save the photo to the defined path
            cv2.imwrite(filename, frame)
            print(f"Photo saved as {filename}")

            # Increment the counter for the next photo
            photo_counter += 1

        # Wait for a short time before reading again
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Process interrupted.")

finally:
    # Release the camera when done
    cap.release()
    print("Camera released.")
