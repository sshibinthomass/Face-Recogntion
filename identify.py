import face_recognition
import cv2, time, pandas
from datetime import datetime

video_capture = cv2.VideoCapture(0)
# Loading Images
shibin_image = face_recognition.load_image_file("photos/shibin1.jpg")
sakthi_image = face_recognition.load_image_file("photos/sakthi1.jpg")

# Recognition of images
shibin_face_encoding = face_recognition.face_encodings(shibin_image)[0]
sakthi_face_encoding = face_recognition.face_encodings(sakthi_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [shibin_face_encoding, sakthi_face_encoding]
known_face_names = ["Shibin", "Sakthi"]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
# Csv file
status_list = [None, None]
times = []
df = pandas.DataFrame(columns=["Start", "End"])

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    status = 0
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        # face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=0, model="cnn")
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        status = 1
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 1)

        # Draw a label with a name below the face
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    # Display the resulting image
    cv2.imshow("Video", frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord("q"):
        if status == 1:
            times.append(datetime.now())
        break

# Release handle to the webcam
print(status_list)
print(times)

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i + 1]}, ignore_index=True)
df.to_csv("times.csv")

video_capture.release()
cv2.destroyAllWindows()
