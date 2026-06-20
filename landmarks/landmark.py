import dlib
import cv2
from PIL import Image
import numpy as np

class Landmarks:

    def __init__(self, number_of_landmarks):

        ### instantiating the detector
        self.detector = dlib.get_frontal_face_detector()

        ### loading the corresponding dlib descriptor
        if number_of_landmarks == 5:
            self.predictor = dlib.shape_predictor("landmarks/shape_predictor_5_face_landmarks.dat")
        else:
            self.predictor = dlib.shape_predictor("landmarks/shape_predictor_68_face_landmarks.dat")


    def generate_landmarks(self, image, channels: int, width: int, height: int):

        ### converting to an array
        image = Image.fromarray(image)

        ### computing scales
        scale_X = width / image.width
        scale_Y = height / image.height

        if channels == 3:
            image = image.convert('L')

        ### detecting face in image
        image = np.array(image)
        faces = self.detector(image)

        if len(faces) == 0:
            raise ValueError("No face detected in image.")

        ### predicting the landmarks
        shape = self.predictor(image, faces[0])

        landmarks = np.array([
            (shape.part(i).x, shape.part(i).y)
            for i in range(shape.num_parts)])

        ### resizing image
        image = Image.fromarray(image).resize((width, height))

        ### rescaling the landmarks
        landmarks = landmarks.astype(np.float32)
        landmarks[:, 0] *= scale_X
        landmarks[:, 1] *= scale_Y
        landmarks = np.round(landmarks).astype(np.int32)

        return image, landmarks