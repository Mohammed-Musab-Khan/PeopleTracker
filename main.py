import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

video_width = 845
video_height = 480

# -------------Activity 2 and 3-------------
# Calculate the mid-horizontal line
mid_y = video_height // 2

# Define the regions
# Upper region (top half of the video)
upper_region = np.array([(0, 0), (0, mid_y), (video_width, mid_y), (video_width, 0)])
upper_region = upper_region.reshape((-1, 1, 2))

# Lower region (bottom half of the video)
lower_region = np.array([(0, mid_y), (0, video_height), (video_width, video_height), (video_width, mid_y)])
lower_region = lower_region.reshape((-1, 1, 2))

# -------------Activity3-------------
total_upper = set()
total_lower = set()

count_in = set()
count_out = set()

cap = cv2.VideoCapture("people.mp4")

while True:
    # -------------Activity1-------------
    # people_count=set()

    # -------------Activity2-------------
    # total_upper = set()
    # total_lower = set()

    ret, frame = cap.read()
    if not ret:
        break

    # -------------Activity 2 and 3-------------
    cv2.line(frame, (0, mid_y), (video_width, mid_y), (255, 255, 255), 3)

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.track(rgb_img, persist=True, verbose=False)

    for i in range(len(results[0].boxes)):

        x1, y1, x2, y2 = results[0].boxes.xyxy[i]

        score = results[0].boxes.conf[i]

        cls = results[0].boxes.cls[i]

        ids = results[0].boxes.id[i]

        x1, y1, x2, y2, score, cls, ids = int(x1), int(y1), int(x2), int(y2), float(score), int(cls), int(ids)

        if score < 0.5 or cls != 0:
            continue

        cx, cy = int(x1 / 2 + x2 / 2), int(y1 / 2 + y2 / 2)

        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        # people_count.add(ids)

        # -------------Activity 2 and 3-------------
        inside_upper_region = cv2.pointPolygonTest(upper_region, (cx, cy), False)
        if inside_upper_region > 0:
            if ids in total_lower:
                count_out.add(ids)
                total_lower.remove(ids)
                cv2.line(frame, (0, mid_y), (video_width, mid_y), (0,0,255), 3)
            total_upper.add(ids)

        inside_lower_region = cv2.pointPolygonTest(lower_region, (cx, cy), False)
        if inside_lower_region > 0:
            if ids in total_upper:
                count_in.add(ids)
                total_upper.remove(ids)
                cv2.line(frame, (0, mid_y), (video_width, mid_y), (0,255,0), 3)
            total_lower.add(ids)

    # -------------Activity 1-------------
    # people_count_text= 'People count: ' + str(len(people_count))
    # cv2.putText(frame, people_count_text, (0, video_height - 10), 0, 1, (0, 128, 0), 1)

    # -------------Activity 2-------------
    # people_upper_text = 'People in upper region: ' + str(len(total_upper))
    # cv2.putText(frame, people_upper_text, (0, 30), 0, 1, (0, 128, 0), 1)
    # people_lower_text = 'People in lower region: ' + str(len(total_lower))
    # cv2.putText(frame, people_lower_text, (0, video_height - 10), 0, 1, (0, 128, 0), 1)

    # -------------Activity 3-------------
    people_upper_text = 'People out: ' + str(len(count_out))
    cv2.putText(frame, people_upper_text, (0, 30), 0, 1, (0, 0, 255), 1)
    people_lower_text = 'People in: ' + str(len(count_in))
    cv2.putText(frame, people_lower_text, (0, video_height - 10), 0, 1, (0, 128, 0), 1)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
