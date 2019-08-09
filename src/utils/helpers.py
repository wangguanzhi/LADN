import numpy as np
import sys
import matplotlib
# Only import the matplotlib.pyplot module if not already done so
# To suppress warning
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use('ps')
    import matplotlib.pyplot as plt
import dlib
import imutils
from imutils import face_utils
import cv2
from PIL import Image

import facemorpher
from facemorpher import locator, aligner, blender
from facemorpher.warper import *

from scipy.stats import multivariate_normal

from .constants import *
from .api_util import FacePPAPI
from .geo_util import expandConvex

LEFT_EYE = [16,17,18,19,20,21, 29,30,31,32,33,34,35,36,37,38]
RIGHT_EYE = [22,23,24,25,26,27,28, 39,40,41,42,43,44,45,46,47]
MOUTH = [59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76]

def listSub(l1, l2):
    return [item for item in l1 if item not in l2]

def combineApiLandmark(lm_stasm, lm_api, include_forehead=True):
    landmark = [[v['x'], v['y']] for k, v in lm_api['faces'][0]['landmark'].items()]
    landmark = np.array(landmark)
    if include_forehead:
        landmark = np.vstack((landmark, lm_stasm[STASM_FOREHEAD]))
    return landmark

def adjustGamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

# https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
def histEqualicationColor(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def cropImageByPointsConvex(img, points):
    mask = np.zeros_like(img)
    convex = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convex, [1,1,1])
    return img * mask

def getConvexMask(img, points):
    mask = np.zeros_like(img)
    convex = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convex, [1,1,1])
    return mask

def warpImageTPS(source,target,img):
    tps = cv2.createThinPlateSplineShapeTransformer()

    source=source.reshape(-1,len(source),2)
    target=target.reshape(-1,len(target),2)

    matches=list()
    for i in range(0,len(source[0])):

        matches.append(cv2.DMatch(i,i,0))

    tps.estimateTransformation(target,source,matches)
    new_img = tps.warpImage(img)
    return new_img

def findResultDir(before_dir, after_dir, warp_name = 'warp'):
    part_no = before_dir.split("/")[-1].split('_')[-1].split('.')[0]

    before_fn = before_dir.split("/")[-1]
    after_fn = after_dir.split("/")[-1]

    before_no = before_fn.split('_')[0]
    after_no = after_fn.split('_')[0]

    prefix = "/".join(before_dir.split("/")[:-2])
    warp_dir = "%s/%s/%s_%s_%s.jpg" % (prefix, warp_name, before_no, after_no, part_no)
    return warp_dir

def mostFrequentColor(img):
    img = img.reshape((-1, img.shape[2]))
    values, counts = np.unique(img, axis = 0, return_counts = True)
    index = np.argmax(counts)
    while values[index].sum() == 0:
        counts[index] = 0
        index = np.argmax(counts)

    return counts[index], values[index]

def getAvgColorWithMask(img, mask):
    img = img.reshape(-1)
    mask = mask.reshape(-1)
    masked_img = np.ma.masked_array(img, mask)
    masked_img = masked_img.reshape((-1,3))

    return np.mean(masked_img, axis = 0)

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

'''The input image must be in RGB color space'''
def faceMaskByHue(img, threshold = 0.15):
    img_hsv = matplotlib.colors.rgb_to_hsv(img)
    # mask = img_hsv[:,:,0] < threshold
    # To remove the block border
    mask = np.logical_or(img_hsv[:,:,0] < threshold, img_hsv[:,:,2] < 6)
    return mask

'''Class with some useful functions for this project'''
class FaceCropper():
    '''
    @params
    predictor_dir: director of shape_predictor_68_face_landmarks.dat
    desiredLeftEye, desiredFaceWidth: passed to face_utils.FaceAligner()
    '''
    def __init__(self,
                 predictor_dir = './shape_predictor_68_face_landmarks.dat',
                 desiredLeftEye = (0.40, 0.40),
                 desiredFaceWidth = 500
                 ):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_dir)
        self.fa = face_utils.FaceAligner(self.predictor,
                                    desiredLeftEye=desiredLeftEye,
                                    desiredFaceWidth=desiredFaceWidth
                                    )

    '''
    Return the face landmark as a (m*2) array using dlib
    '''
    def facePoints(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray)

        # There should be only one face in the image
        if len(rects) != 1:
            return None
        rect = rects[0]

        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        shape = np.vstack((np.array([0,0]), shape))

        return shape

    def facePointsApiStasm(self, img):
        api = FacePPAPI()
        temp_dir = "./temp.jpg"
        cv2.imwrite(temp_dir, img)

        lm_stasm = self.facePointsStasm(img)
        lm_api = api.faceLandmarkDetector(temp_dir)
        lm = combineApiLandmark(lm_stasm = lm_stasm, lm_api = lm_api, include_forehead = True)

        return lm

    '''
    Return the face landmark as a (m*2) array using stasm (It will include the forehead)
    '''
    def facePointsStasm(self, img):
        return locator.face_points(img)


    def isValidStasmLandmarks(self, landmark):
        if np.arrar(landmark).size == 0:
            return False
        else:
            return True

    def facePointsCombined(self, img):
        landmark_dlib = self.facePoints(img)
        landmark_stasm = self.facePointsStasm(img)
        landmark_combined = np.vstack((
            landmark_dlib[DLIB_ALL],
            # landmark_dlib[listSub(DLIB_ALL, DLIB_LEFT_EYEBROW + DLIB_RIGHT_EYEBROW)],
            # landmark_stasm[STASM_LEFT_EYEBROW + STASM_RIGHT_EYEBROW],
            landmark_stasm[STASM_FOREHEAD]
        ))
        return landmark_combined, landmark_dlib, landmark_stasm


    '''
    warp the after_mask(template extracted from DualCNN) to the a before-mask face, with landmarks extracted from the after makeup face
    and then add the warped mask to the before
    '''
    def warpAddTemplate(self, before_img, after_img, after_mask,
                        use_stasm = True,
                        ):
        before_shape = before_img.shape
        after_shape = after_img.shape
        if use_stasm:
            before_points = locator.face_points(before_img)
            after_points = locator.face_points(after_img)
        else:
            before_points = self.facePoints(before_img)
            after_points = self.facePoints(after_img)

        before_points = np.clip(before_points, 0, min(np.array(before_shape[:2]) - 2))
        after_points = np.clip(after_points, 0, min(np.array(after_shape[:2]) - 2))

        mask_transformed = warp_image(after_mask, after_points, before_points, before_shape)
        result = before_img.astype(float) + (mask_transformed.astype(float) - 127) * 2
        result = np.clip(result, 0, 255).astype(np.uint8)
        # return result
        return result, mask_transformed

    '''
    Put the face of the after_image to the position of the before_img
    The input image must be in BGR space (loaded from cv2.imread)

    @params:
    before_img: the image before makeup
    after_img: the image after the makeup
    use_poisson: whether to use pissson blending or the simple Gaussian blending
    use_stasm: whether to use the facial landmark detector in the stasm library or the dlib library
    additional_gaussian; when use possion blending, whether to use additional Gaussian mask to smooth the blended image
    '''
    def warpFace(self, before_img, after_img,
                 before_points = None, after_points = None,
                 use_poisson = True,
                 use_stasm = True,
                 additional_gaussian = False,
                 clone_method = cv2.NORMAL_CLONE,
                 use_tps = False,
                 extra_blending_for_extreme = False,
                 hue_threshold = 0.15,
                 extra_blending_weight = 0.6,
                 adjust_value = False
                ):
        before_shape = before_img.shape
        after_shape = after_img.shape
        if use_stasm:
            if before_points is None: before_points = locator.face_points(before_img)
            if after_points is None: after_points = locator.face_points(after_img)
        else:
            if before_points is None: before_points = self.facePoints(before_img)
            if after_points is None: after_points = self.facePoints(after_img)

        # before_points = np.clip(before_points, 0, min(np.array(before_shape[:2]) - 2))
        # after_points = np.clip(after_points, 0, min(np.array(after_shape[:2]) - 2))
        result_points = before_points

        if additional_gaussian:
            norm = before_points[52]
            std = np.linalg.norm(before_points[14] - before_points[6]) / 3

            x, y = np.mgrid[0:before_shape[0]:1, 0:before_shape[1]:1]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x; pos[:, :, 1] = y
            rv = multivariate_normal(norm[::-1], [[std**2,0], [0,std**2]])
            gauss_mask = rv.pdf(pos)
            gauss_mask /= gauss_mask.max()
            gauss_mask = np.expand_dims(gauss_mask, axis=2)

        x,y,w,h = cv2.boundingRect(np.array([before_points], np.int32)) # (x, y, w, h) from the top-left coordinates
        size = (w,h)

        if use_tps:
            if before_shape[0] > after_shape[0] or before_shape[1] > after_shape[1]:
                expanded = np.zeros( (max(before_shape[0], after_shape[0]), max(before_shape[1], after_shape[1]), 3), dtype = np.uint8)
                expanded[:after_shape[0], :after_shape[1], :] = after_img
            else:
                expanded = after_img

            after_transformed = warpImageTPS(after_points, result_points, cropImageByPointsConvex(expanded, after_points))
            after_transformed = after_transformed[:before_shape[0], :before_shape[1], :]
            self.after_transformed = after_transformed
        else:
            after_transformed = warp_image(after_img, after_points, result_points, before_shape)
            self.after_transformed = after_transformed

        # black_mask = after_transformed.astype(np.float).sum(axis=2) == 0
        # face_mask = after_transformed.astype(np.float).sum(axis=2) != 0
        black_mask = after_transformed.astype(np.float).sum(axis=2) == 0
        black_mask = np.logical_or(black_mask, before_img.astype(np.float).sum(axis=2) == 0)
        face_mask = np.logical_not(black_mask)

        nonzero_x, nonzero_y = face_mask.nonzero()
        x1, x2 = nonzero_x.min(), nonzero_x.max()
        y1, y2 = nonzero_y.min(), nonzero_y.max()

        # print(y1, x1, y2, x2)
        # print(x, y, x+w, y+h)
        x = y1
        y = x1
        w = (y2-y1)
        h = (x2-x1)

        poisson_face = cv2.seamlessClone(after_transformed, before_img, face_mask.astype(np.uint8) * 255, (x+w//2,y+h//2), clone_method)
        self.poisson_face = poisson_face

        self.face_mask_binary = face_mask
        self.face_mask = np.repeat(np.expand_dims(face_mask, 2), 3, axis = 2).astype(np.uint8) * 255

        black_mask = np.expand_dims(black_mask, axis=2)
        face_mask = np.expand_dims(face_mask, axis=2)

        if additional_gaussian:
            result_img = before_img.astype(np.float) * (black_mask + face_mask * (1-gauss_mask)) \
                     + poisson_face.astype(np.float) * gauss_mask * face_mask
        else:
            result_img = before_img * (black_mask) + poisson_face * face_mask

        result_img = result_img.astype(np.uint8)

        '''extra image blending to fix the color blending effect'''
        if extra_blending_for_extreme:
            before_hue_mask = faceMaskByHue(before_img[:,:,::-1], threshold = hue_threshold)
            after_hue_mask = faceMaskByHue(after_transformed[:,:,::-1], threshold = hue_threshold)
            before_face_mask = getConvexMask(before_img, before_points)[:,:,0]
            after_face_mask = getConvexMask(after_transformed, after_points)[:,:,0]

            before_combined_mask = np.logical_and(before_hue_mask, before_face_mask)
            after_combined_mask = ~after_hue_mask

            self.after_combined_mask = after_combined_mask

            before_combined_mask = np.expand_dims(before_combined_mask, axis = 2)
            after_combined_mask = np.expand_dims(after_combined_mask, axis = 2)

            combined = (after_combined_mask * extra_blending_weight) * after_transformed.astype(float) + \
                (after_combined_mask * (1 - extra_blending_weight) + (~after_combined_mask)) * result_img.astype(float)
            combined = combined.astype(np.uint8)

            result_img = combined

        if adjust_value:
            result_hsv = matplotlib.colors.rgb_to_hsv(result_img[:,:,::-1])
            before_hsv = matplotlib.colors.rgb_to_hsv(before_img[:,:,::-1])
            result_value = result_hsv[:,:,2]
            before_value = before_hsv[:,:,2]
            result_value[result_value > before_value] = before_value[result_value > before_value]
            result_hsv[:,:,2] = result_value
            # Convert back to the BGR space
            result_img = matplotlib.colors.hsv_to_rgb(result_hsv).astype(np.uint8)[:,:,::-1]

        return result_img, result_points

    '''
    Get a mask that only the regions of inner eyes and inner mouth are 1 (others are 0)
        if detect == 'api', the api_util will be imported, and the landmark passed in should be the response diectionary returned by api
    '''
    def innerMouthEyeMask(self, img, detector = "stasm", landmark = None):
        assert detector in ['stasm', 'api']

        if detector == 'api':
            api = FacePPAPI()

        if landmark is None:
            if detector == "stasm":
                landmark = self.facePointsStasm(img)
            elif detector == "api":
                landmark = api.faceLandmarkDetector(img)

        mask = np.zeros_like(img)

        if detector == "stasm":
            for part_index in [STASM_LEFT_EYEBALL, STASM_RIGHT_EYEBALL, STASM_MOUTH_INNER]:
                part = landmark[part_index]
                # convex = cv2.convexHull(part)
                # cv2.fillConvexPoly(mask, convex, [1] * 3)
                cv2.fillPoly(mask, [np.vstack((part, part[0]))], [1] * 3)

        if detector == 'api':
            api = FacePPAPI()
            left_eye_inner = api.getLeftEyeInner(landmark)
            right_eye_inner = api.getRightEyeInner(landmark)
            mouth_inner = api.getMouthInner(landmark)
            for part in [left_eye_inner, right_eye_inner, mouth_inner]:
                # convex = cv2.convexHull(part)
                # cv2.fillConvexPoly(mask, convex, [1] * 3)
                cv2.fillPoly(mask, [np.vstack((part, part[0]))], [1] * 3)

        return mask


    '''
    Expand the conves hull from the center to the given ratio

    @params:
    points: points given by a convex hull to be expanded
    ratio: the ratio of the expanding
    '''
    def expandConvex(self, points, ratio, center = None):
        return expandConvex(points, ratio, center)
        # pts = np.array(points)
        # if center is None:
        #     center = pts.mean(axis = 0)
        # new_pts = []
        # vec = pts - center
        # new_pts = center + ratio * vec
        # new_pts = new_pts.astype(int)
        # if not isinstance(points, np.ndarray):
        #     new_pts = new_pts.tolist()
        # return new_pts

    '''
    Return the face parts (left eye, right eye, mouth and the remaining part with proper mask

    @params:
    result_img: image of the face
    result_points: facial landmark returned by stasm (in warpFace() function)
    landmark_type: "api" or "stasm"
    use_circle: whether to crop the face parts as circles on the main face
    expand_ratio: passed to expandConvex (useful only when use_circle == False)
    color: The background color for this image

    @return:
    parts: the cropped face part (image used by pyplot)
    rects: the position of the cropped parts in the given image in [x,y,w,h]
    '''
    def faceParts(self, result_img, result_points, landmark_type="stasm", use_circle=True, expand_ratio=1.0, color = None):
        parts = []
        rects = []
        if landmark_type == 'stasm':
            left_eye = result_points[STASM_LEFT_EYE]
            right_eye = result_points[STASM_RIGHT_EYE]
            mouth = result_points[STASM_MOUTH]
        elif landmark_type == "api":
            left_eye = result_points['left_eye']
            right_eye = result_points['right_eye']
            mouth = result_points['mouth']
        img = np.copy(result_img)
        for i, part in enumerate([left_eye, right_eye, mouth]):
            x,y,w,h = cv2.boundingRect(part)
            center = (x + w//2, y + h//2)
            radius = ((w ** 2 + h ** 2) ** 0.5) / 2
            radius = int(radius)
            rects.append([center[0], center[1], radius*2, radius*2])

            img_part = np.copy(result_img[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius])
            if color is None:
                _, c = mostFrequentColor(img_part)
                    # print(img_part, center[1]-radius, center[1]+radius, center[0]-radius, center[0]+radius)
                c = np.array(c)
            else:
                c = np.array(color)

            # Mask the whole face image
            if use_circle:
                X,Y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
                part_convex = (X-center[1]) ** 2 + (Y - center[0]) ** 2 <= radius ** 2
                img[part_convex] = c
            else:
                if landmark_type == "stasm":
                    if i == 0:
                        expand_center = result_points[16]
                    elif i == 1:
                        expand_center = result_points[23]
                    else:
                        expand_center = None
                else:
                    expand_center = None
                part_convex = cv2.convexHull(part)
                part_convex = self.expandConvex(part_convex, expand_ratio, center=expand_center)
                cv2.fillConvexPoly(img, part_convex, c.tolist())

            # Mask the partial image of the face
            X,Y = np.ogrid[0:2*radius, 0:2*radius]
            try:
                background_mask = (X-radius) ** 2 + (Y - radius) ** 2 > radius ** 2
                img_part[background_mask] = c
            except:
                print("ERROR in faceParts!")

                print(center, radius)
                print(background_mask)
                print(background_mask.shape)
                print(img_part.shape)
            parts.append(img_part)
        parts.append(img)
        return parts, rects


    '''
    Return the coordinates of the bounding boxes of the facial parts (eyes and mouth)
    Return a list of the facial parts as [left_eye, right_eye, mouth], with each being a 4-number list of [x1, x2, y1, y2]
    '''
    def facePartsCoordinates(self, img, landmarks = None, given_eye_ratio = 0.2, given_mouth_ratio = 0.25):
        if landmarks is None:
            landmarks = self.facePointsStasm(img)
        landmarks = np.array(landmarks)
        img_size = img.shape[0]
        left_eye = landmarks[STASM_LEFT_EYE]
        right_eye = landmarks[STASM_RIGHT_EYE]
        mouth = landmarks[STASM_MOUTH]
        rects = []
        for i, part in enumerate([left_eye, right_eye, mouth]):
            x,y,w,h = cv2.boundingRect(part)
            center = (x + w//2, y + h//2)

            if i < 2:
                if given_eye_ratio is None:
                    radius = max(w//2, h//2)
                else:
                    radius = int(img_size * given_eye_ratio / 2)
            if i == 2:
                if given_mouth_ratio is None:
                    radius = max(w//2, h//2)
                else:
                    radius = int(img_size * given_mouth_ratio / 2)

            rects.append([
                center[1]-radius, center[1]+radius,
                center[0]-radius, center[0]+radius,
            ])
        return rects

    '''
    facePartsCoordinates using API landmark
        n_local
            3: left_eye, right_eye, mouth
            6: left_eye, right_eye, mouth, nose, left_cheek, right_cheek
            9: left_eye, right_eye, mouth, nose, left_cheek, right_cheek, left_eyebrow, right_eyebrow, nose_upper
            10: left_eye, right_eye, mouth, nose, left_cheek, right_cheek, left_eyebrow, right_eyebrow, nose_upper, forehead
            12: left_eye, right_eye, mouth, nose, left_cheek, right_cheek, left_eyebrow, right_eyebrow, nose_upper, forehead, mouth_left, mouth_right
    '''
    def facePartsCoordinatesAPI(self, img, landmarks,
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


    '''
    Crop the face parts, fill the background with the most dominant color in the facial area
    Return the images as a list [left_eye, right_eye, mouth, face]

    @params:
    image: image with face in it
    landmarks: facial landmarks
    landmark_type: "api" or "stasm"
    eye_size, mouth_size, face_size: resolutions of the output images
    '''
    def generateTrainingData(self,
                     image,
                     landmarks,
                     landmark_type="stasm",
                     eye_size=(256,256),
                     mouth_size=(256,256),
                     face_size=(256,256)):

        sizes = [eye_size, eye_size, mouth_size, face_size]

        # Get the landmarks for the whole face
        if landmark_type == "stasm":
            landmarks_all = landmarks
        else:
            landmarks_all = landmarks['all']

        face_convex = cv2.convexHull(landmarks_all)
        face_mask = np.zeros_like(image)
        # facial area are white (1,1,1)
        cv2.fillConvexPoly(face_mask, face_convex, [1,1,1])
        masked_face = face_mask * image

        # bg_color = getAvgColorWithMask(image, face_mask)
        _, bg_color = mostFrequentColor(masked_face)
        bg_color = np.array(bg_color, dtype = np.uint8)

        parts, rects = self.faceParts(image, landmarks, landmark_type=landmark_type, use_circle=False, expand_ratio=1.0, color=bg_color)
        results = []

        for i in range(4):
            part = parts[i]
            # If it is the whole face
            if i == 3:
                part = face_mask * part
                part = part + (1-face_mask) * bg_color.reshape((1,1,3))
                part = cropFaceBBox(part, landmarks_all)

            part = cv2.resize(part, sizes[i], interpolation=cv2.INTER_CUBIC)
            results.append(part)

        return results, rects

    def alignFace(self, img):
        img.setflags(write=True)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray)

        # There should be only one face in the image
        if len(rects) != 1:
            return None
        rect = rects[0]

        face_aligned_image = self.fa.align(img, gray, rect)
        return face_aligned_image

    '''
    Cut the image of the major facial area that extends the jaw area from dlib
    to include the forehead.
    (Can only operate on the image containing a single face)

    @params
    img: Numpy.Array of the image containing faces
    output_size: tuple of square size
    '''
    def cutAlignedFace(self,
                   img,
                   output_size = (512, 512)
                   ):
        img.setflags(write=True)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray)

        # There should be only one face in the image
        if len(rects) != 1:
            return None
        rect = rects[0]

        face_aligned_image = self.fa.align(img, gray, rect)
        face_aligned_gray = cv2.cvtColor(face_aligned_image, cv2.COLOR_BGR2GRAY)
        face_aligned_rects = self.detector(face_aligned_gray)

        # There should be only one face in the image
        if len(face_aligned_rects) != 1:
            return None
        face_aligned_rect = face_aligned_rects[0]

        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(face_aligned_gray, face_aligned_rect)
        shape = face_utils.shape_to_np(shape)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name != "jaw":
                continue
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            # To include the forehead
            y = y - h/2
            h = h * 3/2

            center = (int(y+h/2), int(x+w/2))
            half_size = int(np.max([w,h])/2)

            roi = face_aligned_image[center[0]-half_size:center[0]+half_size, center[1]-half_size:center[1]+half_size]
            roi = cv2.resize(roi, output_size, interpolation=cv2.INTER_CUBIC)

        return roi

    def smoothBorder(self, img, points = None):
        if points is None:
            points = self.facePointsStasm(img)

        mask = blender.mask_from_points(img.shape[:2], points)
        inner_points = self.expandConvex(points, ratio=0.9)
        inner_mask = blender.mask_from_points(img.shape[:2], inner_points)

        mask = mask-inner_mask

        return cv2.textureFlattening(img, mask, low_threshold=30 , high_threshold=45)


    def faceMask(self,
             img,
             landmarks = None,
             white = 1,
             grey = 0.1,
             black = 0,
             n_channel = 3
             ):

        if landmarks is None:
            landmarks = self.facePointsStasm(img)

        landmarks = np.array(landmarks)

        if landmarks.size == 0:
            return np.repeat(np.zeros_like(img[:, :, :1]), n_channel, axis = 2)

        # print(landmarks)

        face_convex = cv2.convexHull(landmarks)
        face_mask = np.zeros_like(img).astype(float)

        cv2.fillConvexPoly(face_mask, face_convex, [grey] * 3)

        face_main = face_mask.copy() / grey

        upper_lip = landmarks[STASM_UPPER_LIP]
        lower_lip = landmarks[STASM_LOWER_LIP]

        left_eyeshape = landmarks[STASM_LEFT_EYESHADE]
        right_eyeshape = landmarks[STASM_RIGHT_ETESHADE]

        left_eye = self.expandConvex(landmarks[STASM_LEFT_EYE], 1.2)
        right_eye = self.expandConvex(landmarks[STASM_RIGHT_EYE], 1.2)

        left_eye_outer = self.expandConvex(landmarks[STASM_LEFT_EYEBALL], 2.5)
        right_eye_outer = self.expandConvex(landmarks[STASM_RIGHT_EYEBALL], 2.5)

        left_eye_outer = np.vstack([left_eye_outer, left_eye])
        right_eye_outer = np.vstack([right_eye_outer, right_eye])

        for i, part in enumerate([upper_lip, lower_lip]):
            cv2.fillPoly(face_mask, [part], [white] * 3)

        for part in [left_eye_outer, right_eye_outer]:
            convex = cv2.convexHull(self.expandConvex(part, 1.3))
            cv2.fillConvexPoly(face_mask, convex, [white] * 3)

        for part in [landmarks[STASM_LEFT_EYEBALL], landmarks[STASM_RIGHT_EYEBALL]]:
            convex = cv2.convexHull(part)
            cv2.fillConvexPoly(face_mask, convex, [grey] * 3)

        face_mask *= face_main
        face_mask = np.repeat(face_mask[:,:,:1], n_channel, axis = 2)
        return face_mask

    '''
    return the face mask for post-processing:

    Only keep the facical area and crop out the inner eye and inner mouth areas
    '''
    def faceMaskforPostProcessing(self, img, after_landmarks, after_landmarks_dlib, after_landmarks_stasm):

        for landmarks in [after_landmarks_stasm, after_landmarks_dlib, after_landmarks]:
            landmarks = np.array(landmarks)

            if landmarks.size == 0:
                raise Exception("Something wrong with the landmarks detection")

        face_mask = np.zeros_like(img)
        face_convex = cv2.convexHull(after_landmarks)
        cv2.fillConvexPoly(face_mask, face_convex, [1,1,1])

        for part in [DLIB_LEFT_EYEBALL, DLIB_RIGHT_EYEBALL, DLIB_MOUTH_INNER]:
            part_landmarks = after_landmarks_dlib[part + part[:1]]
            cv2.fillPoly(face_mask, [part_landmarks], [0,0,0])

        return face_mask
