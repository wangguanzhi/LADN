import numpy as np
from .api_util import FacePPAPI

'''
facePartsCoordinates using API landmark
    n_local
        3: left_eye, right_eye, mouth
        6: left_eye, right_eye, mouth, nose, left_cheek, right_cheek
        9: left_eye, right_eye, mouth, nose, left_cheek, right_cheek, left_eyebrow, right_eyebrow, nose_upper
        10: left_eye, right_eye, mouth, nose, left_cheek, right_cheek, left_eyebrow, right_eyebrow, nose_upper, forehead
        12: left_eye, right_eye, mouth, nose, left_cheek, right_cheek, left_eyebrow, right_eyebrow, nose_upper, forehead, mouth_left, mouth_right
'''
def facePartsCoordinatesAPI(img, landmarks,
                            n_local = 3,
                            general_ratio = 0.2,
                            scaling_factor = 1,
                            ):
    api = FacePPAPI()
    img_size = img.shape[0]

    left_eye = api.getLandmarkByName(landmarks, "left_eye_lower_left_quarter")[0]

    right_eye = api.getLandmarkByName(landmarks, "right_eye_lower_right_quarter")[0]

    mouth = api.getLandmarkByName(landmarks, "mouth_lower_lip_top")[0]

    nose = api.getLandmarkByName(landmarks, "nose_bridge3")[0]

    left_cheek = api.getLandmarkByName(landmarks, ["nose_left_contour2", "contour_left8"])
    left_cheek = np.round(np.array(left_cheek).mean(axis = 0))

    right_cheek = api.getLandmarkByName(landmarks, ["nose_right_contour2", "contour_right8"])
    right_cheek = np.round(np.array(right_cheek).mean(axis = 0))

    left_eyebrow = api.getLandmarkByName(landmarks, "left_eyebrow_upper_middle")[0]

    right_eyebrow = api.getLandmarkByName(landmarks, "right_eyebrow_upper_middle")[0]

    nose_upper = api.getLandmarkByName(landmarks, "nose_tip")[0]

    forehead = api.getLandmarkByName(landmarks, ['nose_bridge1', 'nose_bridge3'])
    forehead = forehead[0] + forehead[0] - forehead[1]

    mouth_left = api.getLandmarkByName(landmarks, "mouth_left_corner")[0]
    mouth_right = api.getLandmarkByName(landmarks, "mouth_right_corner")[0]

    n_local = int(n_local)
    if n_local == 3:
        parts = [left_eye, right_eye, mouth]
    elif n_local == 6:
        parts = [left_eye, right_eye, mouth,
            nose, left_cheek, right_cheek]
    elif n_local == 9:
        parts = [left_eye, right_eye, mouth,
            nose, left_cheek, right_cheek,
            left_eyebrow, right_eyebrow, nose_upper]
    elif n_local == 10:
        parts = [left_eye, right_eye, mouth,
            nose, left_cheek, right_cheek,
            left_eyebrow, right_eyebrow, nose_upper, forehead]
    elif n_local == 12:
        parts = [left_eye, right_eye, mouth,
            nose, left_cheek, right_cheek,
            left_eyebrow, right_eyebrow, nose_upper, forehead,
            mouth_left, mouth_right]
    else:
        raise Exception("Unknown number of local parts")

    rects = []
    for i, part in enumerate(parts):
        part = (part * scaling_factor).round().astype(int)

        center = part
        radius = int(img_size * general_ratio / 2)

        rects.append([
            center[1]-radius, center[1]+radius,
            center[0]-radius, center[0]+radius,
        ])

    return rects
