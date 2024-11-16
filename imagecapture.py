import time
import cv2
import os

# File path for the output text file
input_file = "object_count.txt"

# Define the directory where you want to save the photos
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

photo_counter = 0  # Initialize a counter for naming photos

try:
    while True:
        # Capture a frame (photo) from the camera
        ret, frame = cap.read()
        if ret:
            # Display the frame in a window
            cv2.imshow("Camera", frame)

        # Open the file and read the last line
        with open(input_file, "r") as file:
            lines = file.readlines()
            if lines:
                # Get the most recent object count line
                latest_line = lines[-1]
                print(f"Latest Object Count: {latest_line.strip()}")

        # Check for key press (wait for 1 ms)
        key = cv2.waitKey(1) & 0xFF  # Mask to handle all platforms

        if key == ord("s"):  # If 's' is pressed, save the photo
            filename = os.path.join(save_dir, f"photo_{photo_counter}.jpg")
            cv2.imwrite(filename, frame)  # Save the photo
            print(f"Photo saved as {filename}")
            photo_counter += 1  # Increment the counter for the next photo

        elif key == ord("q"):  # If 'q' is pressed, quit the loop
            print("Exiting...")
            break

        # Wait for a short time before reading again (used by cv2.waitKey)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Process interrupted.")

finally:
    # Release the camera and close the window when done
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released.")
