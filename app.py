import cv2
import numpy as np

# Load the video file
video_path = 'video.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Define minimum vehicle size for detection 
min_width_rect = 80
min_height_rect = 80

# Define the position of the counting line 
count_line_position = 500

# Create background subtractor for motion detection
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=450, varThreshold=60, detectShadows=False)

# Define a kernel for morphological operations (noise removal)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Initialize vehicle counter and detection list
counter = 0
detect = []
offset = 6  # Allowable error for counting line crossing

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (1000, 700))  # Output file, codec, FPS, frame size

def center_handle(x, y, w, h):
    """Calculate the center of a bounding box."""
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break  # Exit if the video ends or cannot read the frame

    # Resize frame for consistent processing 
    frame = cv2.resize(frame, (1000, 700))

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(3,3),5)
    img_sub = background_subtractor.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) 
    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    # Find contours of detected objects
    contours, _ = cv2.findContours(dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the counting line
    cv2.line(frame, (25, count_line_position), (950, count_line_position), (255, 127, 0), 3)

    # Process each detected contour
    for contour in contours:
        # Ignore small contours 
        if cv2.contourArea(contour) > 800:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter based on minimum vehicle size
            if w >= min_width_rect and h >= min_height_rect:
                # Draw bounding box around the vehicle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Vehicle" + str(counter), (x,y-20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)

                # Calculate the center of the bounding box
                center = center_handle(x, y, w, h)
                detect.append(center)
                
                

    # Count vehicles crossing the line
        for (x, y) in detect:
            if count_line_position - offset < y < count_line_position + offset:
                counter += 1
                # Highlight the counting line when a vehicle crosses
                cv2.line(frame, (25, count_line_position), (950, count_line_position), (0, 0, 255), 3)
                detect.remove((x, y))
                print("Vehicle Counter:", counter)  # Log the count 

    # Display the vehicle count on the frame
    cv2.putText(frame, "VEHICLE Counter: " + str(counter), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Write the frame into the output video file
    out.write(frame)

    # Show the processed frame
    cv2.imshow("Vehicle Detection", frame)

    # Exit the loop if 'Esc' key is pressed
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release video capture, video writer, and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()
