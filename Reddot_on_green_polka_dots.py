import cv2
import numpy as np

lower_green = np.array([35, 50, 50])
upper_green = np.array([90, 255, 255])

dot_size = 5
dot_color = (0, 0, 255)


def calculate_center(rect):
    x, y, w, h = rect
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy

cap = cv2.VideoCapture("C:\\Users\\Trisha\\Downloads\\AAA\\task_2_video.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('task_2_output.mp4', fourcc, frame_fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_green, upper_green)


    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    threshold_area = 500
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > threshold_area:
            x, y, w, h = cv2.boundingRect(cnt)
            center = calculate_center((x, y, w, h))
            cv2.circle(frame, center, dot_size, dot_color, -1)

    output_video.write(frame)


    cv2.imshow('Output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()

print("Output video saved as 'task_2_output.mp4'")
