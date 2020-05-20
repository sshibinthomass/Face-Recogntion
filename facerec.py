import face_recognition
from PIL import Image
import cv2

image = face_recognition.load_image_file("my_pics/photo2.jpg")
face_locations = face_recognition.face_locations(image)

for face_location in face_locations:
    top, right, bottom, left = face_location

    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
