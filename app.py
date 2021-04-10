import numpy as np
import face_recognition
import os
import cv2
import time
import pandas
from datetime import datetime
import csv
import sys
import winsound
from csv import reader
import time

path = 'E:/Projects/Python/Face-Recogntion/photos'       #Path of photo is mentioned here by Reg_No and Name
images = []
classNames = []
allList = []
regNo = []
myList = os.listdir(path)


#Find most frequent name
def most_frequent(List):
    return max(set(List), key=List.count)

#Write to CSV file
def markAttendance(std_name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if std_name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{std_name},{dtString}')

#Check the Name by 'Y'- yes and 'N'- No
def check(name_list):
    if len(name_list) % 100 == 0 and most_frequent(name_list) != "Unknown":
        std_name = most_frequent(name_list)                                     #Most frequent in 100 values
        print("Detected as:", std_name)
        while True:
            check = input("Please say Y or N or Q")
            check = check.upper()
            if check == "Y":                                                    #When Y is given as input
                markAttendance(std_name)
                print("Thank You Next Person please")
                name_list.clear()
                break
            elif check == "N":                                                  #When N is given as input
                print("Please try again")
                name_list.clear()
                break

#Format the name to Name and RegNo
for name in myList:
    curImg = cv2.imread(f'{path}/{name}')
    images.append(curImg)
    all = name
    name = name.split('_')
    reg = name[0]
    name = name[1]
    regNo.append(os.path.splitext(reg)[0])
    classNames.append(os.path.splitext(name)[0])
    allList.append(os.path.splitext(all)[0])

#Face Encoding using cv2 and FaceRecogniting library
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("Completed Encoding")

known_face_encodings = encodeListKnown
known_face_names = classNames
print(known_face_names)


def face_rec():
    # Initialize some variables
    video_capture = cv2.VideoCapture(0)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    # Csv file
    df = pandas.DataFrame(columns=["Start", "End"])
    count = "Unknown"
    name_list = []

    while True:
        # Grab a single frame of video
        _, frame = video_capture.read()

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
                rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                # name check
                face_names.append(name)
                if count != name:
                    count = name

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
                frame, (left, bottom - 35), (right,
                                             bottom), (0, 0, 0), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        font, 1.0, (255, 255, 255), 1)
            name_list.append(name)
            winsound.Beep(2500, 50)

            # Check Function to check whether the person is right and Append to CSV
            check(name_list)
        # Display the resulting image
        cv2.imshow("Video", frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


face_rec()
