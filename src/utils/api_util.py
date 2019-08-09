# -*- coding: utf-8 -*-
# import urllib2
import requests
from json import JSONDecoder
import numpy as np

from .constants import API_KEY, API_SECRET

class FacePPAPI():
    def __init__(self):
        self.key = API_KEY
        self.secret = API_SECRET

        self.landmark_all = ['contour_chin', 'left_eye_upper_left_quarter', 'mouth_lower_lip_right_contour1', 'left_eye_bottom', 'mouth_lower_lip_right_contour2', 'contour_left7', 'contour_left6', 'contour_left5', 'contour_left4', 'contour_left3', 'contour_left2', 'contour_left1', 'left_eye_lower_left_quarter', 'contour_right1', 'contour_right3', 'contour_right2', 'contour_right5', 'contour_right4', 'contour_right7', 'left_eyebrow_left_corner', 'right_eye_right_corner', 'nose_bridge1', 'nose_bridge3', 'nose_bridge2', 'right_eyebrow_upper_left_corner', 'mouth_upper_lip_right_contour4', 'mouth_upper_lip_right_contour1', 'right_eye_left_corner', 'left_eyebrow_upper_right_corner', 'left_eyebrow_upper_middle', 'mouth_lower_lip_right_contour3', 'nose_left_contour3', 'mouth_lower_lip_bottom', 'mouth_upper_lip_right_contour2', 'left_eye_top', 'nose_left_contour1', 'mouth_upper_lip_bottom', 'mouth_upper_lip_left_contour2', 'mouth_upper_lip_top', 'mouth_upper_lip_left_contour1', 'mouth_upper_lip_left_contour4', 'right_eye_top', 'mouth_upper_lip_right_contour3', 'right_eye_bottom', 'right_eyebrow_lower_left_corner', 'mouth_left_corner', 'nose_middle_contour', 'right_eye_lower_right_quarter', 'right_eyebrow_lower_right_quarter', 'contour_right9', 'mouth_right_corner', 'right_eye_lower_left_quarter', 'right_eye_center', 'left_eye_upper_right_quarter', 'right_eyebrow_lower_left_quarter', 'left_eye_pupil', 'contour_right8', 'contour_left13', 'left_eyebrow_lower_right_quarter', 'left_eye_right_corner', 'left_eyebrow_lower_right_corner', 'mouth_upper_lip_left_contour3', 'left_eyebrow_lower_left_quarter', 'mouth_lower_lip_left_contour1', 'mouth_lower_lip_left_contour3', 'mouth_lower_lip_left_contour2', 'contour_left9', 'left_eye_lower_right_quarter', 'contour_right6', 'nose_tip', 'right_eyebrow_upper_middle', 'right_eyebrow_lower_middle', 'left_eye_center', 'right_eyebrow_upper_left_quarter', 'right_eyebrow_right_corner', 'right_eyebrow_upper_right_quarter', 'contour_left16', 'contour_left15', 'contour_left14', 'left_eyebrow_upper_right_quarter', 'contour_left12', 'contour_left11', 'contour_left10', 'left_eyebrow_lower_middle', 'left_eyebrow_upper_left_quarter', 'right_eye_upper_right_quarter', 'nose_right_contour4', 'nose_right_contour5', 'nose_left_contour4', 'nose_left_contour5', 'nose_left_contour2', 'nose_right_contour1', 'nose_right_contour2', 'nose_right_contour3', 'left_eye_left_corner', 'contour_right15', 'contour_right14', 'contour_right16', 'contour_right11', 'contour_right10', 'contour_right13', 'contour_right12', 'contour_left8', 'mouth_lower_lip_top', 'right_eye_upper_left_quarter', 'right_eye_pupil']

        # self.left_eye_inner = ['left_eye_bottom', 'left_eye_top', 'left_eye_left_corner', 'left_eye_right_corner']
        # self.left_eye_inner += ['left_eye_upper_left_quarter', 'left_eye_lower_left_quarter', 'left_eye_lower_right_quarter', 'left_eye_upper_right_quarter']
        self.left_eye_inner = ['left_eye_bottom', 'left_eye_lower_left_quarter', 'left_eye_left_corner', 'left_eye_upper_left_quarter']
        self.left_eye_inner += ['left_eye_top', 'left_eye_upper_right_quarter', 'left_eye_right_corner', 'left_eye_lower_right_quarter']

        self.left_eyebrow = [l for l in self.landmark_all if l.startswith("left_eyebrow_")]
        self.left_eye = self.left_eyebrow + self.left_eye_inner

        # self.right_eye_inner = ['right_eye_bottom', 'right_eye_top', 'right_eye_left_corner', 'right_eye_right_corner']
        # self.right_eye_inner += ['right_eye_upper_left_quarter', 'right_eye_lower_left_quarter', 'right_eye_lower_right_quarter', 'right_eye_upper_right_quarter']
        self.right_eye_inner = ['right_eye_bottom', 'right_eye_lower_left_quarter', 'right_eye_left_corner', 'right_eye_upper_left_quarter']
        self.right_eye_inner += ['right_eye_top', 'right_eye_upper_right_quarter', 'right_eye_right_corner', 'right_eye_lower_right_quarter']

        self.right_eyebrow = [l for l in self.landmark_all if l.startswith("right_eyebrow_")]
        self.right_eye = self.right_eyebrow + self.right_eye_inner

        # self.mouth_inner = ['mouth_lower_lip_top', 'mouth_upper_lip_bottom', 'mouth_left_corner', 'mouth_right_corner']
        # self.mouth_inner += ['mouth_upper_lip_left_contour4', 'mouth_upper_lip_right_contour4']
        # self.mouth_inner += ['mouth_lower_lip_left_contour1', 'mouth_lower_lip_right_contour1']
        self.mouth_inner = ['mouth_left_corner', 'mouth_upper_lip_left_contour4', 'mouth_upper_lip_bottom', 'mouth_upper_lip_right_contour4']
        self.mouth_inner += ['mouth_right_corner', 'mouth_lower_lip_right_contour1']
        self.mouth_inner += ['mouth_lower_lip_top', 'mouth_lower_lip_left_contour1']

        self.mouth = [l for l in self.landmark_all if l.startswith("mouth_")]

        self.nose = [l for l in self.landmark_all if l.startswith("nose_")]

        self.contour = [l for l in self.landmark_all if l.startswith("contour_")]

    def faceLandmarkDetector(self, filepath):
        http_url ="https://api-us.faceplusplus.com/facepp/v3/detect"
        data = {
            "api_key": self.key,
            "api_secret": self.secret,
            "return_landmark":"2"
        }
        files = {"image_file": open(filepath, "rb")}
        response = requests.post(http_url, data=data, files=files)
        res_con = response.content.decode('utf-8')
        res_dict = JSONDecoder().decode(res_con)
        return res_dict

    def getLeftEyeInner(self, res_dict):
        return self.getLandmarkByName(res_dict, self.left_eye_inner)

    def getRightEyeInner(self, res_dict):
        return self.getLandmarkByName(res_dict, self.right_eye_inner)

    def getLeftEyebrow(self, res_dict):
        return self.getLandmarkByName(res_dict, self.left_eyebrow)

    def getRightEyebrow(self, res_dict):
        return self.getLandmarkByName(res_dict, self.right_eyebrow)

    def getLeftEye(self, res_dict):
        return self.getLandmarkByName(res_dict, self.left_eye)

    def getRightEye(self, res_dict):
        return self.getLandmarkByName(res_dict, self.right_eye)

    def getMouthInner(self, res_dict):
        return self.getLandmarkByName(res_dict, self.mouth_inner)

    def getMouth(self, res_dict):
        return self.getLandmarkByName(res_dict, self.mouth)

    def getNose(self, res_dict):
        return self.getLandmarkByName(res_dict, self.nose)

    def getContour(self, res_dict):
        return self.getLandmarkByName(res_dict, self.contour)

    def getAll(self, res_dict):
        return self.getLandmarkByName(res_dict, self.landmark_all)

    def getLandmarkByName(self, res_dict, name):
        result = []
        landmark = res_dict['faces'][0]['landmark']

        if isinstance(name, str):
            name = [name]
        else:
            assert isinstance(name, list)

        for k in name:
            result.append((landmark[k]['x'],landmark[k]['y']))
        return np.array(result)












#
