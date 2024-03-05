import cv2
import pyautogui

# Get screen size
screen_info = pyautogui.screen_info()
screen_size = (screen_info["width"], screen_info["height"])

# Define the codec using VideoWriter_fourcc() and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 20.0, screen_size)

while True:
    try:
        # Capture screen
        img = pyautogui.screenshot()
        # Write the RBG image to file
        out.write(np.array(img))
        # Display screen/frame being recorded
        cv2.imshow('Screen', np.array(img))
        # Wait for the user to press 'q' key to stop the recording
        if cv2.waitKey(1) == ord('q'):
            break
    except Exception as e:
        break

out.release()  # Close the video file
cv2.destroyAllWindows()  # Close all OpenCV windows