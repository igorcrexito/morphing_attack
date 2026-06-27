import dlib
import cv2
from PIL import Image
import numpy as np

class Landmarks:

    # Canonical face template: where the two eye centers should land (as fractions
    # of the output W/H) after similarity alignment. Smaller EYE_DX zooms in more.
    EYE_Y = 0.40        # vertical position of the eye line
    EYE_DX = 0.16       # half the horizontal eye-to-eye distance

    def __init__(self, number_of_landmarks, max_yaw=None, align=False):

        ### instantiating the detector
        self.detector = dlib.get_frontal_face_detector()

        ### optional frontality gate: faces whose yaw asymmetry exceeds max_yaw are
        ### rejected (dlib's frontal detector still accepts mildly turned faces,
        ### which then ghost when warped). None disables the check.
        self.max_yaw = max_yaw

        ### optional similarity alignment: rotate/scale/translate each face so the
        ### eyes land on a canonical template, then crop to WxH. Centers & scales
        ### the face consistently (better detail + FaceNet identity), and removes
        ### in-plane roll. Needs the 68-point predictor. See _align_face.
        self.align = align

        ### loading the corresponding dlib descriptor
        if number_of_landmarks == 5:
            self.predictor = dlib.shape_predictor("landmarks/shape_predictor_5_face_landmarks.dat")
        else:
            self.predictor = dlib.shape_predictor("landmarks/shape_predictor_68_face_landmarks.dat")

    def _align_face(self, color, landmarks, width, height):
        """Similarity-align a face so its eyes hit the canonical template.

        color      : HxWx3 uint8 RGB image at original resolution.
        landmarks  : 68x2 float landmarks in the same coordinate space.
        Returns (aligned_rgb_uint8 [height x width], transformed_landmarks 68x2).

        Builds the 4-DOF similarity (rotation + uniform scale + translation) that
        maps the two eye centers onto the template points, warps the colour image
        with it, and applies the same matrix to the landmarks so they stay in sync.
        """
        # eye centers: 36-41 = image-left eye, 42-47 = image-right eye.
        eye_l = landmarks[36:42].mean(axis=0)
        eye_r = landmarks[42:48].mean(axis=0)
        dst_l = np.array([(0.5 - self.EYE_DX) * width, self.EYE_Y * height])
        dst_r = np.array([(0.5 + self.EYE_DX) * width, self.EYE_Y * height])

        # closed-form similarity from the two eye correspondences.
        dp, dq = eye_r - eye_l, dst_r - dst_l
        s = (np.linalg.norm(dq) + 1e-8) / (np.linalg.norm(dp) + 1e-8)
        ang = np.arctan2(dq[1], dq[0]) - np.arctan2(dp[1], dp[0])
        cos, sin = s * np.cos(ang), s * np.sin(ang)
        R = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
        t = dst_l - R @ eye_l
        M = np.array([[cos, -sin, t[0]], [sin, cos, t[1]]], dtype=np.float32)

        aligned = cv2.warpAffine(color, M, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
        pts = np.hstack([landmarks, np.ones((len(landmarks), 1), np.float32)])
        return aligned, (M @ pts.T).T.astype(np.float32)

    @staticmethod
    def frontality(landmarks):
        """Yaw-asymmetry score for a 68-point face (0 = perfectly frontal).

        Measures how far the nose tip (point 30) sits from the midpoint of the two
        outer eye corners (36, 45), normalized by inter-eye distance. A frontal
        face keeps the nose centered (~0); a turned head pushes it sideways. Returns
        0.0 for non-68 shapes (no frontality information)."""
        lm = np.asarray(landmarks, dtype=np.float32)
        if len(lm) < 68:
            return 0.0
        eye_mid = (lm[36] + lm[45]) / 2.0
        eye_dist = np.linalg.norm(lm[45] - lm[36]) + 1e-6
        return float(abs(lm[30][0] - eye_mid[0]) / eye_dist)


    def generate_landmarks(self, image, channels: int, width: int, height: int):
        """Detect 68 landmarks and return (output_rgb_image, landmarks).

        The returned image is the colour face at model resolution (width x height):
        similarity-aligned & cropped when ``align`` is set, otherwise a plain
        resize. Callers must use *this* image for warping/saving so the pixels and
        the landmarks stay in the same coordinate space.
        """
        pil = Image.fromarray(image)
        orig_w, orig_h = pil.width, pil.height

        ### detect on a grayscale copy at original resolution
        gray = np.array(pil.convert('L')) if channels == 3 else np.array(pil)
        faces = self.detector(gray)
        if len(faces) == 0:
            raise ValueError("No face detected in image.")

        ### predicting the landmarks (original-resolution coordinates)
        shape = self.predictor(gray, faces[0])
        landmarks = np.array([
            (shape.part(i).x, shape.part(i).y)
            for i in range(shape.num_parts)], dtype=np.float32)

        ### colour image at original resolution (what we actually warp/crop)
        color = np.array(pil.convert('RGB'))

        if self.align and shape.num_parts >= 68:
            ### similarity-align & crop: eyes -> canonical template, landmarks
            ### transformed by the same matrix.
            out_img, landmarks = self._align_face(color, landmarks, width, height)
        else:
            ### legacy path: plain resize + proportional landmark rescale
            out_img = np.array(Image.fromarray(color).resize((width, height)))
            landmarks[:, 0] *= width / orig_w
            landmarks[:, 1] *= height / orig_h

        ### frontality gate: skip faces that are too turned to warp cleanly.
        if self.max_yaw is not None:
            yaw = self.frontality(landmarks)
            if yaw > self.max_yaw:
                raise ValueError(
                    f"Face too non-frontal (yaw asymmetry {yaw:.3f} > "
                    f"max_yaw {self.max_yaw}). Use a frontal photo.")

        landmarks = np.round(landmarks).astype(np.int32)
        return Image.fromarray(out_img), landmarks


    def generate_heatmaps(self, landmarks: np.ndarray, width: int, height: int, sigma: int = 3):

        num_landmarks = landmarks.shape[0]

        heatmaps = np.zeros((height, width, num_landmarks), dtype=np.float32)

        yy, xx = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing="ij")

        for i, (x, y) in enumerate(landmarks):
            heatmaps[:, :, i] = np.exp(
                -((xx - x) ** 2 + (yy - y) ** 2
                ) / (2 * sigma ** 2))

        return heatmaps