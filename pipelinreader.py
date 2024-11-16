import time

# File path for the output text file
input_file = "object_count.txt"

try:
    while True:
        # Open the file and read the last line
        with open(input_file, "r") as file:
            lines = file.readlines()
            if lines:
                # Get the most recent object count line
                latest_line = lines[-1]
                print(f"Latest Object Count: {latest_line.strip()}")

        # Wait for a short time before reading again
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Process interrupted.")
