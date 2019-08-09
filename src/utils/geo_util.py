import numpy as np

'''
Expand the conves hull from the center to the given ratio

@params:
points: points given by a convex hull to be expanded
ratio: the ratio of the expanding
'''
def expandConvex(points, ratio, center = None):
    pts = np.array(points)
    if center is None:
        center = pts.mean(axis = 0)
    new_pts = []
    vec = pts - center
    new_pts = center + ratio * vec
    new_pts = new_pts.astype(int)
    if not isinstance(points, np.ndarray):
        new_pts = new_pts.tolist()
    return new_pts

def cropFaceBBox(img, landmarks, expand = 1.0, change_landmarks = False):
    face_bbox = cv2.boundingRect(landmarks)
    x,y,w,h = face_bbox
    cx = x + w/2
    cy = y + h/2
    l = max(w,h) * expand
    img = img[int(cy - l/2): int(cy + l/2),
           int(cx - l/2): int(cx + l/2)]
    if change_landmarks:
        new_landmarks = landmarks - np.array([int(cx - l/2), int(cy - l/2)])
        return img, new_landmarks
    return img
