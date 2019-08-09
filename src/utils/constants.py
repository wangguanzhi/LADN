'''
The indexes of the stasm and dlib landmarks
'''

API_KEY = "2RIMQKQge97okAtBUFTR0FXRWpuGeDlh"
API_SECRET = "TCSe4TedTYvmxzONO4_jUNUXlY4-yYZ1"

DLIB_DAT_PATH = "../models/shape_predictor_68_face_landmarks.dat"
DUMMY_BLEND_PATH = "../dataset/train/dummp.jpg"

STASM_LEFT_EYEBALL = list(range(30, 38))
STASM_LEFT_EYEBROW = list(range(16, 22))
STASM_LEFT_EYE_CENTER = [38, 29]
STASM_LEFT_EYE = STASM_LEFT_EYEBALL + STASM_LEFT_EYEBROW + STASM_LEFT_EYE_CENTER
STASM_LEFT_EYESHADE = [18,19,20,21,30,31,32,33,34]

STASM_RIGHT_EYEBALL = list(range(40, 48))
STASM_RIGHT_EYEBROW = list(range(22, 28))
STASM_RIGHT_EYE_CENTER = [39, 28]
STASM_RIGHT_EYE = STASM_RIGHT_EYEBALL + STASM_RIGHT_EYEBROW + STASM_RIGHT_EYE_CENTER
STASM_RIGHT_ETESHADE = [25,26,27,22,40,41,42,43,44]

STASM_UPPER_LIP = list(range(59, 69))
STASM_LOWER_LIP = [59] + list(range(69, 72)) + [65] + list(range(72, 77))
STASM_MOUTH_INNER = [59] + list(range(65, 72))
STASM_MOUTH = STASM_UPPER_LIP + STASM_LOWER_LIP

STASM_FOREHEAD = list(range(13,16)) + list(range(77, 79))

STASM_ALL = list(range(79))

DLIB_LEFT_EYEBALL = list(range(37, 43))
DLIB_LEFT_EYEBROW = list(range(18, 23))
DLIB_LEFT_EYE = DLIB_LEFT_EYEBALL + DLIB_LEFT_EYEBROW

DLIB_RIGHT_EYEBALL = list(range(43, 49))
DLIB_RIGHT_EYEBROW = list(range(23, 28))
DLIB_RIGHT_EYE = DLIB_RIGHT_EYEBALL + DLIB_RIGHT_EYEBROW

DLIB_UPPER_LIP = list(range(49, 56)) + list(range(61,66))
DLIB_LOWER_LIP = list(range(55, 62)) + list(range(65,69)) + [49]
DLIB_MOUTH = DLIB_UPPER_LIP + DLIB_LOWER_LIP
DLIB_MOUTH_INNER = list(range(61, 69))

DLIB_ALL = list(range(1, 69))
