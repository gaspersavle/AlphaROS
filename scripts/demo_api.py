# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu,Hao-Shu Fang
# -----------------------------------------------------

"""Script for single-image demo."""
import argparse
import torch
import os
import platform
import sys
import math
import time
import statistics as stat
import queue
import yaml
import sys
#from helpers import path
from transformations import * 

import cv2
import numpy as np

##################################################################
import rospy
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge   
import colorama
from colorama import Fore, Style
##################################################################

from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.models import builder
from alphapose.utils.config import update_config
from detector.apis import get_detector
from alphapose.utils.vis import getTime

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Single-Image Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
parser.add_argument('--circles', default = False, action = 'store_true',
                    help='draw circles arround wrists')
parser.add_argument('--depth', default = False, action = 'store_true',
                    help='display wrist proximity')
parser.add_argument('--dict', type=str, help='Specify which aruco dictionary is used to determine camera pose',
                    default='DICT_5X5_100')
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)


args = parser.parse_args()
cfg = update_config(args.cfg)

args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

class DetectionLoader():
    def __init__(self, detector, cfg, opt):
        self.cfg = cfg
        self.opt = opt
        self.device = opt.device
        self.detector = detector

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = cfg.DATA_PRESET.SIGMA

        if cfg.DATA_PRESET.TYPE == 'simple':
            pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
            self.transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)
        elif cfg.DATA_PRESET.TYPE == 'simple_smpl':
            # TODO: new features
            from easydict import EasyDict as edict
            dummpy_set = edict({
                'joint_pairs_17': None,
                'joint_pairs_24': None,
                'joint_pairs_29': None,
                'bbox_3d_shape': (2.2, 2.2, 2.2)
            })
            self.transformation = SimpleTransform3DSMPL(
                dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
                color_factor=cfg.DATASET.COLOR_FACTOR,
                occlusion=cfg.DATASET.OCCLUSION,
                input_size=cfg.MODEL.IMAGE_SIZE,
                output_size=cfg.MODEL.HEATMAP_SIZE,
                depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=(2.2, 2,2, 2.2),
                rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
                train=False, add_dpg=False, gpu_device=self.device,
                loss_type=cfg.LOSS['TYPE'])

        self.image = (None, None, None, None)
        self.det = (None, None, None, None, None, None, None)
        self.pose = (None, None, None, None, None, None, None)

    def process(self, im_name, image):
        # start to pre process images for object detection
        self.image_preprocess(im_name, image)
        # start to detect human in images
        self.image_detection()
        # start to post process cropped human image for pose estimation
        self.image_postprocess()
        return self

    def image_preprocess(self, im_name, image):
        # expected image shape like (1,3,h,w) or (3,h,w)
        img = self.detector.image_preprocess(image)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        # add one dimension at the front for batch if image shape (3,h,w)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        orig_img = image # scipy.misc.imread(im_name_k, mode='RGB') is depreciated
        im_dim = orig_img.shape[1], orig_img.shape[0]

        im_name = os.path.basename(im_name)

        with torch.no_grad():
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        self.image = (img, orig_img, im_name, im_dim)

    def image_detection(self):
        imgs, orig_imgs, im_names, im_dim_list = self.image
        if imgs is None:
            self.det = (None, None, None, None, None, None, None)
            return

        with torch.no_grad():
            dets = self.detector.images_detection(imgs, im_dim_list)
            if isinstance(dets, int) or dets.shape[0] == 0:
                self.det = (orig_imgs, im_names, None, None, None, None, None)
                return
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            ids = torch.zeros(scores.shape)

        boxes = boxes[dets[:, 0] == 0]
        if isinstance(boxes, int) or boxes.shape[0] == 0:
            self.det = (orig_imgs, im_names, None, None, None, None, None)
            return
        inps = torch.zeros(boxes.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)

        self.det = (orig_imgs, im_names, boxes, scores[dets[:, 0] == 0], ids[dets[:, 0] == 0], inps, cropped_boxes)

    def image_postprocess(self):
        with torch.no_grad():
            (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes) = self.det
            if orig_img is None:
                self.pose = (None, None, None, None, None, None, None)
                return
            if boxes is None or boxes.nelement() == 0:
                self.pose = (None, orig_img, im_name, boxes, scores, ids, None)
                return

            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            self.pose = (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes)

    def read(self):
        return self.pose


class DataWriter():
    def __init__(self, cfg, opt):
        self.cfg = cfg
        self.opt = opt

        self.eval_joints = list(range(cfg.DATA_PRESET.NUM_JOINTS))
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        self.item = (None, None, None, None, None, None, None)
        
        loss_type = self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss')
        num_joints = self.cfg.DATA_PRESET.NUM_JOINTS
        if loss_type == 'MSELoss':
            self.vis_thres = [0.4] * num_joints
        elif 'JointRegression' in loss_type:
            self.vis_thres = [0.05] * num_joints
        elif loss_type == 'Combined':
            if num_joints == 68:
                hand_face_num = 42
            else:
                hand_face_num = 110
            self.vis_thres = [0.4] * (num_joints - hand_face_num) + [0.05] * hand_face_num

        self.use_heatmap_loss = (self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss')

    def start(self):
        # start to read pose estimation results
        return self.update()

    def update(self):
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE

        # get item
        (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.item
        if orig_img is None:
            return None
        # image channel RGB->BGR
        orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
        self.orig_img = orig_img
        if boxes is None or len(boxes) == 0:
            return None
        else:
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            assert hm_data.dim() == 4
            if hm_data.size()[1] == 136:
                self.eval_joints = [*range(0,136)]
            elif hm_data.size()[1] == 26:
                self.eval_joints = [*range(0,26)]
            elif hm_data.size()[1] == 133:
                self.eval_joints = [*range(0,133)]
            pose_coords = []
            pose_scores = []

            for i in range(hm_data.shape[0]):
                bbox = cropped_boxes[i].tolist()
                if isinstance(self.heatmap_to_coord, list):
                    pose_coords_body_foot, pose_scores_body_foot = self.heatmap_to_coord[0](
                        hm_data[i][self.eval_joints[:-110]], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coords_face_hand, pose_scores_face_hand = self.heatmap_to_coord[1](
                        hm_data[i][self.eval_joints[-110:]], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                    pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
                else:
                    pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)

            boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area, use_heatmap_loss=self.use_heatmap_loss)

            _result = []
            for k in range(len(scores)):
                _result.append(
                    {
                        'keypoints':preds_img[k],
                        'kp_score':preds_scores[k],
                        'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx':ids[k],
                        'bbox':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                    }
                )

            result = {
                'imgname': im_name,
                'result': _result
            }

            if hm_data.size()[1] == 49:
                from alphapose.utils.vis import vis_frame_dense as vis_frame
            elif self.opt.vis_fast:
                from alphapose.utils.vis import vis_frame_fast as vis_frame
            else:
                from alphapose.utils.vis import vis_frame
            self.vis_frame = vis_frame

        return result

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        self.item = (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name)


class SingleImageAlphaPose():
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.image = None
        self.enablePose = False
        self.enableCamPose = True
        self.poseNode = '/realsense/alphapose/enable'
        self.camPoseNode = '/realsense_top/get_pose'
        rospy.Service(self.poseNode, SetBool, self.enablePose_CB)
        rospy.Service(self.camPoseNode, SetBool, self.enableCamPose_CB)

        self.create_service_client(self.poseNode)
        self.create_service_client(self.camPoseNode)
        self.arucoInit(args.dict)
        self.rvec = np.zeros((3,1))
        self.tvec = np.zeros((3,1)) 
        
        #self.config = update_config(config_file='config.yaml')

        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

        print(f'Loading pose model from {args.checkpoint}...')
        self.pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        self.pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)

        self.pose_model.to(args.device)
        self.pose_model.eval()
        
        self.det_loader = DetectionLoader(get_detector(self.args), self.cfg, self.args)
        #init_p, D, K, self.P, w, h = self.parse_calib_yaml(self.config.realsense.calibration_file)
        ####################################################################################
        ## Begin: Body dict
        ####################################################################################
        self.body ={'R_ankle': {'x': None, 'y': None, 'z': None, 'pf': 'r_knee_default', 
                        'cf': 'r_ankle_default','rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'R_knee',
                        'transj': [0, 0, -0.44], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_ankle': {'x': None, 'y': None, 'z': None, 'pf': 'l_knee_default', 
                        'cf': 'l_ankle_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'L_knee',
                        'transj': [0, 0, -0.44], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_knee': {'x': None, 'y': None, 'z': None, 'pf': 'r_hip_default',
                        'cf': 'r_knee_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'R_ankle', 'parent': 'R_hip', 'neg': False,
                        'transj': [0, 0, -0.33], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_knee': {'x': None, 'y': None, 'z': None, 'pf': 'l_hip_default',
                        'cf': 'l_knee_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'L_ankle', 'parent': 'L_hip', 'neg': False,
                        'transj': [0, 0, -0.33], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10), 
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_hip_yaw': {'x': None, 'y': None, 'z': None, 'pf': 'waist_default',
                        'cf': 'r_y_hip_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': 'R_knee', 'parent': 'body', 'neg': True,
                        'transj': [0, -0.132, 0], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_hip_pitch': {'x': None, 'y': None, 'z': None, 'pf': 'r_y_hip_default',
                        'cf': 'r_p_hip_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': 'R_knee', 'parent': 'R_hip_yaw', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_hip': {'x': None, 'y': None, 'z': None, 'pf': 'r_p_hip_default',
                        'cf': 'r_hip_default', 'rot_x': True, 'rot_y': False, 'rot_z': False, 'lower_j': 'R_knee', 'parent': 'R_hip_pitch', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_hip_yaw': {'x': None, 'y': None, 'z': None, 'pf': 'waist_default',
                        'cf': 'l_y_hip_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': 'L_knee', 'parent': 'body', 'neg': True, 
                        'transj': [0, 0.132, 0], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_hip_pitch': {'x': None, 'y': None, 'z': None, 'pf': 'l_y_hip_default',
                        'cf': 'l_p_hip_default', 'rot_x': True, 'rot_y': False, 'rot_z': False, 'lower_j': 'L_knee', 'parent': 'L_hip_yaw', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_hip': {'x': None, 'y': None, 'z': None, 'pf': 'l_p_hip_default',
                        'cf': 'l_hip_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'L_knee', 'parent': 'L_hip_pitch', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_wrist': {'x': None, 'y': None, 'z': None, 'pf': 'r_elbow_default',
                        'cf': 'r_wrist_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'R_elbow', 
                        'transj': [0, 0, -0.22], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_wrist': {'x': None, 'y': None, 'z': None, 'pf': 'l_elbow_default',
                        'cf': 'l_wrist_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'L_elbow', 
                        'transj': [0, 0, -0.22], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_elbow': {'x': None, 'y': None, 'z': None, 'pf': 'r_shoulder_default',
                        'cf': 'r_elbow_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'R_wrist', 'parent': 'R_shoulder', 'neg': False, 
                        'transj': [0, 0, -0.352], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_elbow': {'x': None, 'y': None, 'z': None, 'pf': 'l_shoulder_default',
                        'cf': 'l_elbow_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'L_wrist', 'parent': 'L_shoulder', 'neg': False, 
                        'transj': [0, 0, -0.352], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_shoulder_yaw': {'x': None, 'y': None, 'z': None, 'pf': 'torso_default',
                        'cf': 'r_y_shoulder_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': 'R_elbow', 'parent': 'torso', 'neg': False, 
                        'transj' : [0, -0.176, 0], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_shoulder_pitch': {'x': None, 'y': None, 'z': None, 'pf': 'r_y_shoulder_default',
                        'cf': 'r_p_shoulder_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'R_elbow', 'parent': 'R_shoulder_yaw', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_shoulder': {'x': None, 'y': None, 'z': None, 'pf': 'r_p_shoulder_default',
                        'cf': 'r_shoulder_default', 'rot_x': True, 'rot_y': False, 'rot_z': False, 'lower_j': 'R_elbow', 'parent': 'R_shoulder_pitch', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_shoulder_yaw': {'x': None, 'y': None, 'z': None, 'pf': 'torso_default',
                        'cf': 'l_y_shoulder_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': 'L_elbow', 'parent': 'torso', 'neg': True, 
                        'transj' : [0, 0.176, 0], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_shoulder_pitch': {'x': None, 'y': None, 'z': None, 'pf': 'l_y_shoulder_default',
                        'cf': 'l_p_shoulder_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'L_elbow', 'parent': 'L_shoulder_yaw', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_shoulder': {'x': None, 'y': None, 'z': None, 'pf': 'l_p_shoulder_default',
                        'cf': 'l_shoulder_default', 'rot_x': True, 'rot_y': False, 'rot_z': False, 'lower_j': 'L_elbow', 'parent': 'L_shoulder_pitch', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_ear': {'x': None, 'y': None, 'z': None, 'cf': None,
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'transj' : None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_ear': {'x': None, 'y': None, 'z': None, 'cf': None,
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'transj' : None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_eye': {'x': None, 'y': None, 'z': None, 'cf': None,
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'transj' : None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_eye': {'x': None, 'y': None, 'z': None, 'cf': None,
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'transj' : None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'head': {'x': None, 'y': None, 'z': None, 'pf': 'p_head_default', 'cf': 'head_default',
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'head_pitch', 
                        'transj' : [0.055, 0, 0.11], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'head_roll': {'x': None, 'y': None, 'z': None, 'pf': 'torso_default', 'cf': 'r_head_default',
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'torso', 
                        'transj' : [0, 0, 0.055], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'head_yaw': {'x': None, 'y': None, 'z': None, 'pf': 'r_head_default', 'cf': 'y_head_default',
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'head_roll', 
                        'transj' : None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'head_pitch': {'x': None, 'y': None, 'z': None, 'pf':'y_head_default', 'cf': 'p_head_default',
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'head_yaw', 
                        'transj' : None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'body': {'x': None, 'y': None, 'z': None,
                        'pf': 'world', 'cf': 'body_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 
                        'transj' : None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'torso': {'x': None, 'y': None,
                        'z': None, 'pf': 'waist_default', 'cf': 'torso_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'waist', 'neg': True, 
                        'transj': [0, 0, 0.605], 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None},
                    
                    'waist': {'x': None, 'y': None,'z': None, 'pf': 'body_default', 
                        'cf': 'waist_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'torso', 'parent': 'body', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(10), 'qy': np.ndarray(10), 'qz': np.ndarray(10),
                        'worldx': None, 'worldy': None, 'worldz': None}}
        #init rospy
        rospy.init_node("vision", anonymous = True)

        self.pub_TRANS_POSE = tf2_ros.TransformBroadcaster()
        self.transmsg = geometry_msgs.msg.TransformStamped()
        self.tfbuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfbuffer)
        self.camInfo('/realsense_top/color')
        
        self.maxDEPTH = rospy.get_param("/realsense_top/aligned_depth_to_color/image_raw/compressedDepth/depth_max") # Za kasnejse mapiranje globine
        self.sub_POSE = rospy.Subscriber("/realsense_top/color/image_raw", Image, self.pose_CB)
        self.sub_DEPTH = rospy.Subscriber("/realsense_top/aligned_depth_to_color/image_raw", Image, self.depth_CB)
        self.pub_POSE = rospy.Publisher("/alphapose_pose", Image, queue_size=1)
        self.pub_DEPTH = rospy.Publisher("/alphapose_depth", Image, queue_size=1)
        
        
        rospy.spin()
        ####################################################################################

    def pose_CB(self, input):
        if self.enablePose:
            self.img_POSE = CvBridge().imgmsg_to_cv2(input, desired_encoding='rgb8')
            #self.img_POSE = cv2.resize(self.img_POSE, ())
            self.pose = self.process("demo", self.img_POSE)
            self.vis_POSE = self.vis(self.img_POSE, self.pose)

            if self.pose != None:
                self.keypoints = self.pose['result'][0]['keypoints']

                self.secs = str(input.header.stamp)[:10]
                self.nsecs = str(input.header.stamp)[10:]

                #print(f"Sec: {self.secs} | Nsec: {self.nsecs}"
                jIgnore = ['waist', 'torso', 'body']
                tagIgnore = ['roll', 'pitch', 'yaw']
                i = 0
                for key, joint in self.body.items():
                    keySegs = key.split('_')
                    
                    #print(ending)
                    if keySegs[0] not in jIgnore:
                        if keySegs[-1] not in tagIgnore:
                            #print(keySegs)
                            joint['x'] = int(self.keypoints[16-i][0])
                            joint['y'] = int(self.keypoints[16-i][1])
                            #print(f"{Fore.RED}{key}\n{Fore.BLACK}",joint)
                            i+=1
                    
                self.body['R_hip_yaw']['x'] = self.body['R_hip_pitch']['x'] = self.body['R_hip']['x'] 
                self.body['R_hip_yaw']['y'] = self.body['R_hip_pitch']['y'] = self.body['R_hip']['y'] 

                self.body['L_hip_yaw']['x'] = self.body['L_hip_pitch']['x'] = self.body['L_hip']['x'] 
                self.body['L_hip_yaw']['y'] = self.body['L_hip_pitch']['y'] = self.body['L_hip']['y'] 

                self.body['R_shoulder_yaw']['x'] = self.body['R_shoulder_pitch']['x'] = self.body['R_shoulder']['x'] 
                self.body['R_shoulder_yaw']['y'] = self.body['R_shoulder_pitch']['y'] = self.body['R_shoulder']['y'] 
                
                self.body['L_shoulder_yaw']['x'] = self.body['L_shoulder_pitch']['x'] = self.body['L_shoulder']['x'] 
                self.body['L_shoulder_yaw']['y'] = self.body['L_shoulder_pitch']['y'] = self.body['L_shoulder']['y'] 

                self.body['head_yaw']['x'] = self.body['head_roll']['x'] = self.body['head_pitch']['x'] = self.body['head']['x']
                self.body['head_yaw']['y'] = self.body['head_roll']['y'] = self.body['head_pitch']['y'] = self.body['head']['y']

                self.body['body']['x'] = (self.body['L_hip']['x']+self.body['R_hip']['x'])/2
                self.body['body']['y'] = (self.body['L_hip']['y']+self.body['R_hip']['y'])/2

                self.body['waist']['x'] = self.body['body']['x']
                self.body['waist']['y'] = self.body['body']['y']

                self.body['torso']['x'] = (self.body['L_shoulder']['x']+self.body['R_shoulder']['x'])/2
                self.body['torso']['y'] = (self.body['L_shoulder']['y']+self.body['R_shoulder']['y'])/2



                            



                if self.args.circles == True:
                    self.vis_POSE = cv2.circle(self.vis_POSE, (self.body['L_wrist']['x'], self.body['L_wrist']['y']), radius=10, color=(255, 0, 255), thickness=2)
                    self.vis_POSE = cv2.circle(self.vis_POSE, (self.body['R_wrist']['x'], self.body['R_wrist']['y']), radius=10, color=(255, 0, 255), thickness=2)
                
                #print(f"{Fore.GREEN}{self.vis_POSE}")
            else:
                print(f"{Fore.RED} No pose detected...")
                

            self.markerHandler(image=self.vis_POSE)
            
            self.out_POSE = CvBridge().cv2_to_imgmsg(self.vis_POSE, encoding = 'rgb8')
            self.pub_POSE.publish(self.out_POSE)

    def depth_CB(self, input):

        if self.enablePose:
            self.img_DEPTH = CvBridge().imgmsg_to_cv2(input, desired_encoding='16UC1')
            self.img_blur_DEPTH = cv2.GaussianBlur(self.img_DEPTH, (5,5), cv2.BORDER_DEFAULT)
            #print(f"{Fore.GREEN}Depth: {self.img_blur_DEPTH[350, 240]}")
            #self.colourised = cv2.cvtColor(self.img_DEPTH, cv2.COLOR_GRAY2RGB)
            #self.vis_DEPTH = self.vis(self.img_DEPTH, self.pose)
            
            #print(f"{Fore.YELLOW} {self.img_DEPTH}")
            if self.pose != None:
                for key, joint in self.body.items():
                    #print(key)
                    if joint['y'] >= 480:
                        joint['z'] = self.depth_remap(self.img_blur_DEPTH[479, int(joint['x'])])
                        #joint['z'] = self.img_blur_DEPTH[479, int(joint['x'])]
                    elif joint['x'] >= 640:
                        joint['z'] = self.depth_remap(self.img_blur_DEPTH[int(joint['y']), 639])
                    else:
                        joint['z'] = self.depth_remap(self.img_blur_DEPTH[int(joint['y']), int(joint['x'])])
                        #joint['z'] = self.img_blur_DEPTH[int(joint['y']), int(joint['x'])]
                    #print('Joint z: ', joint['z'])
                    np.roll(joint['qx'],-1)
                    np.take(joint['qx'], 0)
                    np.put(joint['qx'], 2, joint['x'])

                    np.roll(joint['qy'],-1)
                    np.take(joint['qy'], 0)
                    np.put(joint['qy'], 2, joint['y'])

                    np.roll(joint['qz'],-1)
                    np.take(joint['qz'], 0)
                    np.put(joint['qz'], 2, joint['z'])
                    #print(joint['y'])
                    #print(f"{Fore.LIGHTBLUE_EX}{key}\n  Qx: {joint['qx']}\n  Qy: {joint['qy']}\n  Qz: {joint['qz']}")

                print(f"{Fore.CYAN}LEFT:\nDEPTH: {self.body['L_wrist']['z']} | LOCATION: {self.body['L_wrist']['x'], self.body['L_wrist']['y']}")
                print(f"{Fore.MAGENTA}RIGHT:\nDEPTH: {self.body['R_wrist']['z']} | LOCATION: {self.body['R_wrist']['x'], self.body['R_wrist']['y']}")
                #print(f"{Fore.GREEN} Max depth: {np.ndarray.max(self.img_DEPTH)} {Fore.RED} | Min depth: {np.ndarray.min(self.img_DEPTH)}")
                """ self.vis_POSE = cv2.putText(self.vis_POSE, 
                                            str(self.body['L_wrist']['z']), 
                                            [self.body['L_wrist']['x'], self.body['L_wrist']['y']],
                                            fontFace=cv2.FONT_HERSHEY_COMPLEX, 
                                            fontScale=1, 
                                            color=(0,255,255), thickness=2)
                self.vis_POSE = cv2.putText(self.vis_POSE,
                                            str(self.body['R_wrist']['z']), 
                                            [self.body['R_wrist']['x'], self.body['R_wrist']['y']], 
                                            fontFace=cv2.FONT_HERSHEY_COMPLEX, 
                                            fontScale=1, 
                                            color=(0,255,255), thickness=2) """
                
                self.maxdepth_loc, self.mindepth_loc = np.unravel_index(np.argmax(self.img_blur_DEPTH),self.img_blur_DEPTH.shape), np.unravel_index(np.argmin(self.img_blur_DEPTH), self.img_blur_DEPTH.shape)


                self.rounddepth_L = str(self.body['L_wrist']['z'])[:4]
                self.rounddepth_R = str(self.body['L_wrist']['z'])[:4]

                print(f"{Fore.GREEN} Max depth: {self.maxdepth_loc} {Fore.RED} | Min depth: {self.mindepth_loc}")
                #print(f"{Fore.LIGHTYELLOW_EX} RAW left: {self.img_blur_DEPTH[self.body['L_wrist']['x'], self.body['L_wrist']['y']]} | RAW right: {self.img_blur_DEPTH[self.body['R_wrist']['x'], self.body['R_wrist']['y']]}")
                
                for key, joint in self.body.items():
                    jointx = self.GetMoveAvg(joint['qx'])
                    #print(jointx)
                    jointy = self.GetMoveAvg(joint['qy'])
                    #print(jointy)
                    jointz = self.GetMoveAvg(joint['qz'])
                    #print(jointz)

                    jointxyz = self.uv_to_XY(joint['x'], joint['y'], joint['z'])
                    #jointxyz = self.uv_to_XY(jointx, jointy, jointz)

                    """ self.SendTransform2tf(p=bodyxyz, parent_frame='panda_2/realsense', child_frame=self.body['body']['cf']) 
                    transform = self.GetCameraTrans('world','body_default')
                    cameratrans = self.GetCameraTrans('world', 'panda_2/realsense')
                    self.camerarot = [cameratrans.rotation.x, cameratrans.rotation.y, cameratrans.rotation.z, cameratrans.rotation.w]
                    #print(f'{Fore.CYAN}{cameratrans}')
                    rotation = q2r([transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z])
                    tr = np.eye(3)
                    tr = np.transpose(rotation) """
                    
                    #TODO: Popravi zanko, da ne bo posiljala translacije
                    
                    """ if joint['cf'] and key != 'body':
                        if joint['pf'] == 'body_default':
                            posfin = np.eye(3)@tr
                            q_fin = r2q(posfin)
                            #trans_joint_xyz = self.transToParent_xyz(parent=joint['parent'], child=key)
                            #trans_joint_xyz = self.uv_to_XY(joint['x'], joint['y'], joint['z'])
                            self.SendTransform2tf(q=q_fin, parent_frame=joint['pf'], child_frame=joint['cf'])
                        else:
                            #and joint['rot_y'] == False and joint['rot_z'] == False
                            if joint['cf'] != None:
                                rotJoint = self.getRot(joint=key)
                                if joint['transj'] != None:
                                    transJoint = joint['transj']
                                else:
                                    transJoint = [0,0,0]
                                self.SendTransform2tf(p=transJoint, q=rotJoint, parent_frame=joint['pf'], child_frame=joint['cf']) """
                    if joint['cf'] != None:
                        self.SendTransform2tf(p=jointxyz, parent_frame='rs_top', child_frame=(joint['cf']+'/rs'))
                        if key == 'body':
                            transToWorld = self.GetCameraTrans('world',joint['cf']+'/rs')
                            transJoint = [transToWorld.translation.x, transToWorld.translation.y, transToWorld.translation.z]
                            self.SendTransform2tf(p=transJoint, parent_frame=joint['pf'], child_frame=joint['cf'])
                        else:
                            #print(f"{Fore.RED}Append to: {key}\n{joint}")
                            #jointxyz = self.uv_to_XY(joint['x'], joint['y'], joint['z'])
                            
                            
                            transToWorld = self.GetCameraTrans(joint['cf']+'/rs', 'world')
                            
                            worldPos = transToWorld.translation
                            #print(key, worldPos)
                            
                            joint['worldx'] = -worldPos.x
                            joint['worldy'] = -worldPos.y
                            joint['worldz'] = -worldPos.z
                            #print(key, joint['worldz'])

                            jointRot = self.getRot(joint=key)
                            if joint['transj'] != None:
                                transJoint = joint['transj']
                            else:
                                transJoint = [0,0,0]
                            self.SendTransform2tf(p=transJoint, q=jointRot, parent_frame=joint['pf'], child_frame=joint['cf'])
                            #print(key,'World Z: ', joint['worldz'])
                if self.args.circles == True:
                    
                    self.circle_DEPTH = cv2.circle(self.img_blur_DEPTH, (self.body['L_wrist']['x'], self.body['L_wrist']['y']), radius=10, color=(255, 0, 255), thickness=2)
                    self.circle_DEPTH = cv2.circle(self.circle_DEPTH, (self.body['R_wrist']['x'], self.body['R_wrist']['y']), radius=10, color=(255, 0, 255), thickness=2)
                    self.out_DEPTH = CvBridge().cv2_to_imgmsg(self.circle_DEPTH, encoding = '16UC1')

            else:
                self.out_DEPTH = CvBridge().cv2_to_imgmsg(self.img_DEPTH, encoding = '16UC1')
            if self.camPose != None:
                self.SendTransform2tf(p=self.camPose, parent_frame="/world", child_frame="/rs_top")
                # q=self.camRot,
                
            self.pub_DEPTH.publish(self.out_DEPTH)

    def arucoInit(self,dict):
        ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }

        self.arucoDict = ARUCO_DICT[dict]

    def markerHandler(self, image:np.ndarray):
        markerLib = cv2.aruco.Dictionary_get(self.arucoDict)
        params = cv2.aruco.DetectorParameters_create()
        #print('Refinement: ', params.cornerRefinementMethod)
        #detImage = cv2.flip(image, -1)
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, markerLib,
	    parameters=params)

        #print(f"{Fore.GREEN}Corner: {corners}{type(corners)}")
        #print(f"{Fore.BLUE}ID: {ids}{type(ids)}")
        #print(f"{Fore.RED}Rej: {rejected}{type(rejected)}")

        if (corners != None) and (len(corners) > 0):
            # loop over the detected ArUCo corners
            i = 0
            marker = []
            ids = np.ndarray.flatten(ids)
            self.cornerdict = {1: None,
                                    2: None,
                                    3: None,
                                    4: None,
                                    5: None,
                                    6: None,
                                    7: None,
                                    8: None}
            #if len(corners) == 8:
            self.corners = np.asfarray(corners)#.reshape(32,2)
                
            print(self.corners.shape, self.corners)
            for (id, corner) in zip(ids,corners):
                corners = corner.reshape((4, 2))
                self.cornerdict[id] = np.roll(np.asfarray(corners), -1)
                
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = [int(topRight[0]), int(topRight[1])]
                bottomRight = [int(bottomRight[0]), int(bottomRight[1])]
                bottomLeft = [int(bottomLeft[0]), int(bottomLeft[1])]
                topLeft = [int(topLeft[0]), int(topLeft[1])]
                #print(topLeft)
                # draw the bounding box of the ArUCo detection
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                cv2.putText(image, str(id),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                i+=1

            #print(f"{Fore.CYAN}{self.corners}")
            #print(self.cornerdict)
            self.camPose, self.camRot = self.getCamPose()
            print(f"{Fore.GREEN}ROTmat: {self.camRot}")
            self.enableCamPose = False
            
            return 

    def camInfo(self, camera_topic:str):
        caminfo = rospy.wait_for_message(camera_topic+'/camera_info', CameraInfo, timeout=20)
        self.camera_fx = caminfo.K[0]
        self.camera_cx = caminfo.K[2]
        self.camera_fy = caminfo.K[4]
        self.camera_cy = caminfo.K[5]

        self.cameramatrix = np.array([[self.camera_fx, 0.0, self.camera_cx],
                                    [0.0, self.camera_fy, self.camera_cy],
                                    [0.0, 0.0, 1.0]],dtype=np.float32)

        self.distCoefs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
    
    def getCamPose(self):
        obj_pts = np.array([[-0.2,0.299, 0.825], # marker 1
                    [-0.2,0.199, 0.825],
                    [-0.3,0.199, 0.825],
                    [-0.3,0.299, 0.825],

                    [0.3,0.299, 0.825], # marker 2
                    [0.3,0.199, 0.825],
                    [0.2,0.199, 0.825],
                    [0.2,0.299,0.825],
                    
                    [0.4,0.899,0.825], # marker 3
                    [0.4,0.799,0.825],
                    [0.3,0.799,0.825],
                    [0.3,0.899,0.825],
                    
                    [0.9,0.399,0.825], # marker 4
                    [0.9,0.299,0.825],
                    [0.8,0.299,0.825],
                    [0.8,0.399,0.825],

                    [-0.2,0.999,0.825], # marker 5
                    [-0.2,0.899,0.825],
                    [-0.3,0.899,0.825],
                    [-0.3,0.999,0.825],

                    [0.3,0.999,0.825], # marker 6
                    [0.3,0.899,0.825],
                    [0.2,0.899,0.825],
                    [0.2,0.999,0.825],

                    [0.4,0.999,0.825], # marker 7
                    [0.4,0.899,0.825],
                    [0.3,0.899,0.825],
                    [0.3,0.999,0.825],

                    [0.9,1.499,0.825], # marker 8
                    [0.9,1.399,0.825],
                    [0.8,1.399,0.825],
                    [0.8,1.499,0.825]
                    ], dtype=np.float32)
        for marker, corners in self.cornerdict.items():
            if corners.any() != None:
                print(f"{Fore.GREEN}{corners}")
            else:
                print(f"{Fore.RED}{corners}")

        #print(self.corners)
        flag = cv2.SOLVEPNP_ITERATIVE 
        retval,  rvec, tvec = cv2.solvePnP(obj_pts, self.corners, self.cameramatrix, self.distCoefs, self.rvec, self.tvec, flags=flag)
        print(f"{Fore.LIGHTCYAN_EX}Entering calib...{rvec}")
        rotm = np.zeros((3,3)) 
        cv2.Rodrigues(rvec, rotm)
        rotq = r2q(rotm)
        print('RotQ: ',rotq, 'RotM: ',rotm )
        # q=[1, rvec[0][0], rvec[1][0], rvec[2][0]],
        return [tvec[1], tvec[0], tvec[2]], rotq
        

    def SendTransform2tf(self, p=[0,0,0],q=[1,0,0,0], parent_frame = "panda_2/realsense",child_frame="Human_Pose"):
        
        self.transmsg.header.stamp = rospy.Time.now()
        self.transmsg.header.frame_id = parent_frame

        self.transmsg.child_frame_id = child_frame

        self.transmsg.transform.translation.x = p[0]
        self.transmsg.transform.translation.y = p[1]
        self.transmsg.transform.translation.z = p[2]

        self.transmsg.transform.rotation.w = q[0]
        self.transmsg.transform.rotation.x = q[1]
        self.transmsg.transform.rotation.y = q[2]
        self.transmsg.transform.rotation.z = q[3]

        self.pub_TRANS_POSE.sendTransform(self.transmsg)

    def GetCameraTrans(self, from_sys:str, to_sys:str):
        trans= self.tfbuffer.lookup_transform(from_sys, to_sys, rospy.Time())
        transform = trans.transform
        return transform

    def GetMoveAvg(self, axis:np.ndarray):
        
        return axis.mean()


    def getRot(self, joint:str) -> np.ndarray:
        curJoint = self.body[joint]
        
        if curJoint['lower_j'] != None:
            lowJoint = self.body[curJoint['lower_j']]
            #print(f"{Fore.MAGENTA}{joint} | Z: {curJoint['worldz']}")
            #print(f"{Fore.LIGHTMAGENTA_EX}{curJoint['lower_j']} | Z: {lowJoint['worldz']}")
            curPos = [curJoint['worldx'], curJoint['worldy'], curJoint['worldz']]
            lowPos = [lowJoint['worldx'], lowJoint['worldy'], lowJoint['worldz']]

            if curJoint['rot_x']:
                dx = curPos[0] - lowPos[0]
                dy = curPos[1] - lowPos[1]
                rot = math.atan(dx/dy)
                if curJoint['neg']:
                    rot = -rot
                return r2q(rot_x(rot))
            
            if curJoint['rot_y']:
                #print('Cur: ', curPos)
                #print('Low: ', lowPos)
                dz = curPos[2] - lowPos[2]
                dy = curPos[1] - lowPos[1]
                rot = math.atan(dz/dy)
                if curJoint['neg']:
                    rot = -rot
                return r2q(rot_y(rot))
            
            if curJoint['rot_z']:
                dx = curPos[0] - lowPos[0]
                dz = curPos[2] - lowPos[2]
                rot = math.atan(dx/dz)
                if curJoint['neg']:
                    rot = -rot
                return r2q(rot_z(rot))
            
            return r2q(np.eye(3))
        else:
            return r2q(np.eye(3))

    def getTrans(self, joint:str) -> list:
        curJoint = self.body[joint]
        parent = self.body[curJoint['parent']]
        #print(lowJoint)
        diff =self.uv_to_XY(curJoint['x']-parent['x'],
                            curJoint['y']-parent['y'],
                            curJoint['z']-parent['z'])    
        #print(diff)
        return [0, diff[0]*10, diff[1]*10]

    def depth_remap(self, depth) -> float:
        self.adj_DEPTH = 65536- depth
        self.range_16b = 65536
        self.range_depth = 300 - 3
        self.remapped = (depth * self.range_depth) / self.range_16b 

        #return self.remapped
        #return self.remapped/20 + 0.3
        return depth/1000

    def uv_to_XY(self, u:int,v:int, z:int) -> list:
        """Convert pixel coordinated (u,v) from realsense camera into real world coordinates X,Y,Z """
        #print(f"{Fore.GREEN}U: {u}\nV: {v}\nZ: {z}")
        x = (u - (self.camera_cx)) / self.camera_fx
        y = (v - (self.camera_cy)) / self.camera_fy

        X = (z * x)
        Y = (z * y)
        Z = z
        return [X, Y, Z]

    def transToParent_xyz(self, parent:str, child:str) -> list:
        parentlink = self.body[parent]
        joint = self.body[child]
        jointxyz = self.uv_to_XY(joint['x'], joint['y'], joint['z'])
        parentxyz = self.uv_to_XY(parentlink['x'], parentlink['y'], parentlink['z'])
        """ delta = [jointxyz[0]-parentxyz[0], jointxyz[1]-parentxyz[1], jointxyz[2]-parentxyz[2]]
        result = []
        print(parent)
        print(child)
        for i in range(len(delta)):
            result.append(parentxyz[i] + delta[i])
        return result """

        dz = jointxyz[2] - parentxyz[2]
        dx = jointxyz[0] - parentxyz[0]
        dy = jointxyz[1] - parentxyz[1]
        
        return [0, 2*dy, 2*dz]

    
    def enablePose_CB(self, req):
        state = req.data
        if state:
            print("AlphaPose: starting...")
            self.enablePose = True
            msg = self.poseNode + " started."
        else:
            print("AlphaPose: stopping...")
            self.enablePose = False
            msg = self.poseNode + " stopped."
        return True, msg

    def enableCamPose_CB(self, req):
        state = req.data
        if state:
            print("Camera pose estimation: starting...")
            self.enableCamPose = True
            msg = self.camPoseNode + " started."
        else:
            print("Camera pose estimation: stopping...")
            self.enableCamPose = False
            msg = self.camPoseNode + " stopped."
        return True, msg

    def create_service_client(self, node):
        try:
            print("waiting for service:" + node)
            rospy.wait_for_service(node, 2) # 2 seconds
        except rospy.ROSException as e:
            print("Couldn't find service! " + node)
        self.camera_service = rospy.ServiceProxy(node, SetBool)

    def process(self, im_name, image):
        # Init data writer
        self.writer = DataWriter(self.cfg, self.args)

        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }
        pose = None
        try:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = self.det_loader.process(im_name, image).read()
                if orig_img is None:
                    raise Exception("no image is given")
                if boxes is None or boxes.nelement() == 0:
                    if self.args.profile:
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    self.writer.save(None, None, None, None, None, orig_img, im_name)
                    if self.args.profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    pose = self.writer.start()
                    if self.args.profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)
                else:
                    if self.args.profile:
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    # Pose Estimation
                    inps = inps.to(self.args.device)
                    if self.args.flip:
                        inps = torch.cat((inps, flip(inps)))
                    hm = self.pose_model(inps)
                    if self.args.flip:
                        hm_flip = flip_heatmap(hm[int(len(hm) / 2):], self.pose_dataset.joint_pairs, shift=True)
                        hm = (hm[0:int(len(hm) / 2)] + hm_flip) / 2
                    if self.args.profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    hm = hm.cpu()
                    self.writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                    pose = self.writer.start()
                    if self.args.profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)

            if self.args.profile:
                print(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
        except Exception as e:
            print(repr(e))
            print('An error as above occurs when processing the images, please check it')
            pass


        return pose

    def getImg(self):
        return self.writer.orig_img

    def vis(self, image, pose):
        if pose is not None:
            image = self.writer.vis_frame(image, pose, self.writer.opt, self.writer.vis_thres)
        return image

    def writeJson(self, final_result, outputpath, form='coco', for_eval=False):
        from alphapose.utils.pPose_nms import write_json
        write_json(final_result, outputpath, form=form, for_eval=for_eval)
        print("Results have been written to json.")

    def parse_calib_yaml(self, fn):
        """Parse camera calibration file (which is hand-made using ros camera_calibration) """

        with open(fn, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        data = data['Realsense']
        init_p = data['init_robot_pos']
        #print(data)
        w  = data['coeffs'][0]['width']
        h = data['coeffs'][0]['height']
    
        D = np.array(data['coeffs'][0]['D'])
        K = np.array(data['coeffs'][0]['K']).reshape(3,3)
        P = np.array(data['coeffs'][0]['P']).reshape(3,4)
    
        return init_p, D, K, P, w, h


if __name__ == "__main__":
    print('version:', sys.version)
    print('info', sys.version_info)
    SingleImageAlphaPose(args, cfg)