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
from sensor_msgs.msg import Image
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
        self.poseNode = '/realsense/alphapose/enable'
        rospy.Service(self.poseNode, SetBool, self.enablePose_CB)
        self.create_service_client()
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
        #init rospy
        rospy.init_node("vision", anonymous = True)

        self.pub_TRANS_POSE = tf2_ros.TransformBroadcaster()
        self.transmsg = geometry_msgs.msg.TransformStamped()
        self.tfbuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfbuffer)

        self.maxDEPTH = rospy.get_param("/realsense/aligned_depth_to_color/image_raw/compressedDepth/depth_max") # Za kasnejse mapiranje globine
        self.sub_POSE = rospy.Subscriber("/realsense/color/image_raw", Image, self.pose_CB)
        self.sub_DEPTH = rospy.Subscriber("/realsense/aligned_depth_to_color/image_raw", Image, self.depth_CB)
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
                self.body ={'R_ankle': {'x': int(self.keypoints[16][0]), 'y': int(self.keypoints[16][1]), 'z': None, 'pf': 'r_knee_default', 
                                'cf': 'r_ankle_default','rot_x': False, 'rot_y': False, 'rot_z': False, 'parent': 'R_knee'},

                            'L_ankle': {'x': int(self.keypoints[15][0]), 'y': int(self.keypoints[15][1]), 'z': None, 'pf': 'l_knee_default', 
                                'cf': 'l_ankle_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'parent': 'L_knee'},

                            'R_knee': {'x': int(self.keypoints[14][0]), 'y': int(self.keypoints[14][1]), 'z': None, 'pf': 'r_hip_default',
                                'cf': 'r_knee_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'parent': 'R_hip'},

                            'L_knee': {'x': int(self.keypoints[13][0]), 'y': int(self.keypoints[14][1]), 'z': None, 'pf': 'l_hip_default',
                                'cf': 'l_knee_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'parent': 'L_hip'},

                            'R_hip_yaw': {'x': int(self.keypoints[12][0]), 'y': int(self.keypoints[12][1]), 'z': None, 'pf': 'waist_default',
                                'cf': 'r_y_hip_default', 'rot_x': False, 'rot_y': False, 'rot_z': True, 'lower_j': 'R_knee', 'parent': 'body'},

                            'R_hip_pitch': {'x': int(self.keypoints[12][0]), 'y': int(self.keypoints[12][1]), 'z': None, 'pf': 'r_y_hip_default',
                                'cf': 'r_p_hip_default', 'rot_x': True, 'rot_y': False, 'rot_z': False, 'lower_j': 'R_knee', 'parent': 'R_hip_yaw'},

                            'R_hip': {'x': int(self.keypoints[12][0]), 'y': int(self.keypoints[12][1]), 'z': None, 'pf': 'r_p_hip_default',
                                'cf': 'r_hip_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'R_knee', 'parent': 'R_hip_pitch'},

                            'L_hip_yaw': {'x': int(self.keypoints[11][0]), 'y': int(self.keypoints[11][1]), 'z': None, 'pf': 'waist_default',
                                'cf': 'l_y_hip_default', 'rot_x': False, 'rot_y': False, 'rot_z': True, 'lower_j': 'L_knee', 'parent': 'body'},

                            'L_hip_pitch': {'x': int(self.keypoints[11][0]), 'y': int(self.keypoints[11][1]), 'z': None, 'pf': 'l_y_hip_default',
                                'cf': 'l_p_hip_default', 'rot_x': True, 'rot_y': False, 'rot_z': False, 'lower_j': 'L_knee', 'parent': 'L_hip_yaw'},

                            'L_hip': {'x': int(self.keypoints[11][0]), 'y': int(self.keypoints[11][1]), 'z': None, 'pf': 'l_p_hip_default',
                                'cf': 'l_hip_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'L_knee', 'parent': 'L_hip_pitch'},

                            'waist': {'x': int((self.keypoints[11][0]+self.keypoints[12][0])/2), 'y': int((self.keypoints[11][1]+self.keypoints[12][1])/2),
                                'z': None, 'pf': 'body_default', 'cf': 'waist_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'torso', 'parent': 'body'},

                            'R_wrist': {'x': int(self.keypoints[10][0]), 'y': int(self.keypoints[10][1]), 'z': None, 'pf': 'r_elbow_default',
                                'cf': 'r_wrist_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'parent': 'R_elbow'},

                            'L_wrist': {'x': int(self.keypoints[9][0]), 'y': int(self.keypoints[9][1]), 'z': None, 'pf': 'l_elbow_default',
                                'cf': 'l_wrist_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'parent': 'L_elbow'},

                            'R_elbow': {'x': int(self.keypoints[8][0]), 'y': int(self.keypoints[8][1]), 'z': None, 'pf': 'r_shoulder_default',
                                'cf': 'r_elbow_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'R_wrist', 'parent': 'R_shoulder'},

                            'L_elbow': {'x': int(self.keypoints[7][0]), 'y': int(self.keypoints[7][1]), 'z': None, 'pf': 'l_shoulder_default',
                                'cf': 'l_elbow_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'L_wrist', 'parent': 'L_shoulder'},

                            'R_shoulder_yaw': {'x': int(self.keypoints[6][0]), 'y': int(self.keypoints[6][1]), 'z': None, 'pf': 'torso_default',
                                'cf': 'r_y_shoulder_default', 'rot_x': False, 'rot_y': False, 'rot_z': True, 'lower_j': 'R_elbow', 'parent': 'torso'},

                            'R_shoulder_pitch': {'x': int(self.keypoints[6][0]), 'y': int(self.keypoints[6][1]), 'z': None, 'pf': 'r_y_shoulder_default',
                                'cf': 'r_p_shoulder_default', 'rot_x': True, 'rot_y': False, 'rot_z': False, 'lower_j': 'R_elbow', 'parent': 'R_shoulder_yaw'},

                            'R_shoulder': {'x': int(self.keypoints[6][0]), 'y': int(self.keypoints[6][1]), 'z': None, 'pf': 'r_p_shoulder_default',
                                'cf': 'r_shoulder_default', 'rot_x': False, 'rot_y': False, 'rot_z': True, 'lower_j': 'R_elbow', 'parent': 'R_shoulder_pitch'},

                            'L_shoulder_yaw': {'x': int(self.keypoints[5][0]), 'y': int(self.keypoints[5][1]), 'z': None, 'pf': 'torso_default',
                                'cf': 'l_y_shoulder_default', 'rot_x': False, 'rot_y': False, 'rot_z': True, 'lower_j': 'L_elbow', 'parent': 'torso'},

                            'L_shoulder_pitch': {'x': int(self.keypoints[5][0]), 'y': int(self.keypoints[5][1]), 'z': None, 'pf': 'l_y_shoulder_default',
                                'cf': 'l_p_shoulder_default', 'rot_x': True, 'rot_y': False, 'rot_z': False, 'lower_j': 'L_elbow', 'parent': 'L_shoulder_yaw'},

                            'L_shoulder': {'x': int(self.keypoints[5][0]), 'y': int(self.keypoints[5][1]), 'z': None, 'pf': 'l_p_shoulder_default',
                                'cf': 'l_shoulder_default', 'rot_x': False, 'rot_y': False, 'rot_z': True, 'lower_j': 'L_elbow', 'parent': 'L_shoulder_pitch'},

                            'torso': {'x': int((self.keypoints[6][0]+self.keypoints[5][0])/2), 'y': int((self.keypoints[6][1]+self.keypoints[5][1])/2),
                                'z': None, 'pf': 'body_default', 'cf': 'torso_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'parent': 'body'},

                            'R_ear': {'x': int(self.keypoints[4][0]), 'y': int(self.keypoints[4][1]), 'z': None, 'cf': None,
                                'rot_x': False, 'rot_y': False, 'rot_z': False},

                            'L_ear': {'x': int(self.keypoints[3][0]), 'y': int(self.keypoints[3][1]), 'z': None, 'cf': None,
                                'rot_x': False, 'rot_y': False, 'rot_z': False},

                            'R_eye': {'x': int(self.keypoints[2][0]), 'y': int(self.keypoints[2][1]), 'z': None, 'cf': None,
                                'rot_x': False, 'rot_y': False, 'rot_z': False},

                            'L_eye': {'x': int(self.keypoints[1][0]), 'y': int(self.keypoints[1][1]), 'z': None, 'cf': None,
                                'rot_x': False, 'rot_y': False, 'rot_z': False},

                            'head': {'x': int(self.keypoints[0][0]), 'y': int(self.keypoints[0][1]), 'z': None, 'pf': 'p_head_default', 'cf': 'head_default',
                                'rot_x': False, 'rot_y': False, 'rot_z': False, 'parent': 'head_pitch'},

                            'head_roll': {'x': int(self.keypoints[0][0]), 'y': int(self.keypoints[0][1]), 'z': None, 'pf': 'body_default', 'cf': 'r_head_default',
                                'rot_x': False, 'rot_y': False, 'rot_z': False, 'parent': 'torso'},

                            'head_yaw': {'x': int(self.keypoints[0][0]), 'y': int(self.keypoints[0][1]), 'z': None, 'pf': 'r_head_default', 'cf': 'y_head_default',
                                'rot_x': False, 'rot_y': False, 'rot_z': False, 'parent': 'head_roll'},

                            'head_pitch': {'x': int(self.keypoints[0][0]), 'y': int(self.keypoints[0][1]), 'z': None, 'pf':'y_head_default', 'cf': 'p_head_default',
                                'rot_x': False, 'rot_y': False, 'rot_z': False, 'parent': 'head_yaw'},

                            'body': {'x': int((self.keypoints[11][0]+self.keypoints[12][0])/2), 'y': int((self.keypoints[11][1]+self.keypoints[12][1])/2), 'z': None,
                                'pf': 'panda_2/realsense', 'cf': 'body_default', 'rot_x': False, 'rot_y': False, 'rot_z': False}}

                            
                            



                if self.args.circles == True:
                    self.vis_POSE = cv2.circle(self.vis_POSE, (self.body['L_wrist']['x'], self.body['L_wrist']['y']), radius=10, color=(255, 0, 255), thickness=2)
                    self.vis_POSE = cv2.circle(self.vis_POSE, (self.body['R_wrist']['x'], self.body['R_wrist']['y']), radius=10, color=(255, 0, 255), thickness=2)
                    
                    #self.vis_POSE = cv2.circle(self.vis_POSE, (self.maxdepth_loc[1], self.maxdepth_loc[0]), radius=10, color=(0, 255, 0), thickness=2)
                    #self.vis_POSE = cv2.circle(self.vis_POSE, (self.mindepth_loc[1], self.mindepth_loc[0]), radius=10, color=(255, 0, 0), thickness=2)

                """ if self.args.depth == True:
                    self.vis_POSE = cv2.putText(self.vis_POSE,
                    text=self.rounddepth_L,
                    org=(self.body['L_wrist']['x']-60, self.body['L_wrist']['y']),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.5,
                    color=(0, 128, 128),
                    thickness=1)

                    self.vis_POSE = cv2.putText(self.vis_POSE,
                    text=self.rounddepth_R,
                    org=(self.body['R_wrist']['x']+30, self.body['R_wrist']['y']),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.5,
                    color=(0, 128, 128),
                    thickness=1) """
                
                    #pass

                

                #print(f"Kolicina tock: {len(self.keypoints)}")
                #print(f"D: {self.R_wrist} | L: {self.L_wrist}")
            else:
                print(f"{Fore.RED} No pose detected...")

            self.out_POSE = CvBridge().cv2_to_imgmsg(self.vis_POSE, encoding = 'rgb8')
            self.pub_POSE.publish(self.out_POSE)

    def depth_CB(self, input):

        if self.enablePose:
            self.img_DEPTH = CvBridge().imgmsg_to_cv2(input, desired_encoding='16UC1')
            self.img_blur_DEPTH = cv2.GaussianBlur(self.img_DEPTH, (5,5), cv2.BORDER_DEFAULT)
            #self.colourised = cv2.cvtColor(self.img_DEPTH, cv2.COLOR_GRAY2RGB)
            #self.vis_DEPTH = self.vis(self.img_DEPTH, self.pose)
            
            #print(f"{Fore.YELLOW} {self.img_DEPTH}")
            if self.pose != None:
                for key, joint in self.body.items():
                    if joint['y'] >= 480:
                        joint['z'] = self.depth_remap(self.img_blur_DEPTH[479, int(joint['x'])])
                    else:
                        joint['z'] = self.depth_remap(self.img_blur_DEPTH[int(joint['y']), int(joint['x'])])
                    #print(joint['y'])


                print(f"{Fore.CYAN}LEFT:\nDEPTH: {self.body['L_wrist']['z']} | LOCATION: {self.body['L_wrist']['x'], self.body['L_wrist']['y']}")
                print(f"{Fore.MAGENTA}RIGHT:\nDEPTH: {self.body['R_wrist']['z']} | LOCATION: {self.body['R_wrist']['x'], self.body['R_wrist']['y']}")
                #print(f"{Fore.GREEN} Max depth: {np.ndarray.max(self.img_DEPTH)} {Fore.RED} | Min depth: {np.ndarray.min(self.img_DEPTH)}")
                
                self.maxdepth_loc, self.mindepth_loc = np.unravel_index(np.argmax(self.img_blur_DEPTH),self.img_blur_DEPTH.shape), np.unravel_index(np.argmin(self.img_blur_DEPTH), self.img_blur_DEPTH.shape)


                self.rounddepth_L = str(self.body['L_wrist']['z'])[:4]
                self.rounddepth_R = str(self.body['L_wrist']['z'])[:4]

                print(f"{Fore.GREEN} Max depth: {self.maxdepth_loc} {Fore.RED} | Min depth: {self.mindepth_loc}")
                #print(f"{Fore.LIGHTYELLOW_EX} RAW left: {self.img_blur_DEPTH[self.body['L_wrist']['x'], self.body['L_wrist']['y']]} | RAW right: {self.img_blur_DEPTH[self.body['R_wrist']['x'], self.body['R_wrist']['y']]}")
                
                for key, joint in self.body.items():
                    bodyxyz = self.uv_to_XY(self.body['body']['x'], self.body['body']['y'], self.body['body']['z'])

                    self.SendTransform2tf(p=bodyxyz, parent_frame='panda_2/realsense', child_frame=self.body['body']['cf']) 
                    transform = self.GetCameraTrans('world','body_default')
                    rotation = q2r([transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z])
                    tr = np.eye(3)
                    tr = np.transpose(rotation)
                    
                    #TODO: Popravi zanko, da ne bo posiljala translacije
                    if joint['cf'] and key != 'body':
                        if joint['pf'] == 'body_default':
                            posfin = np.eye(3)@tr
                            q_fin = r2q(posfin)
                            #trans_joint_xyz = self.transToParent_xyz(parent=joint['parent'], child=key)
                            trans_joint_xyz = self.uv_to_XY(joint['x'], joint['y'], joint['z'])
                            self.SendTransform2tf(p=trans_joint_xyz, q=q_fin, parent_frame=joint['pf'], child_frame=joint['cf'])
                        else:
                            #and joint['rot_y'] == False and joint['rot_z'] == False
                            if joint['cf'] != None:
                                parentind = joint['parent']
                                distx = joint['x']-self.body[parentind]['x']
                                disty = joint['y']-self.body[parentind]['y']
                                distz = joint['z']-self.body[parentind]['z']
                                pvec = self.uv_to_XY(distx, disty, distz)
                                #trans_joint_xyz = self.transToParent_xyz(parent=joint['parent'], child=key)
                                #trans_joint_xyz = self.uv_to_XY(joint['x'], joint['y'], joint['z'])
                                self.SendTransform2tf(p=pvec, parent_frame=joint['pf'], child_frame=joint['cf'])
                            if joint['rot_y']:
                                trans_joint_q = self.transToParent_PY(child=key)
                                self.SendTransform2tf(p=pvec, q=trans_joint_q, parent_frame=joint['pf'], child_frame=joint['cf'])
                            if joint['rot_z']:
                                
                                trans_joint_q = self.transToParent_PY(child=key)
                                self.SendTransform2tf(p=pvec, q=trans_joint_q, parent_frame=joint['pf'], child_frame=joint['cf'])

                """ body = self.body['body']
                bodyxyz = self.uv_to_XY(body['x'], body['y'], body['z'])

                torso = self.body['torso']
                torsoxyz = self.uv_to_XY(torso['x'], torso['y'], torso['z'])

                shoulderyR = self.body['R_shoulder_yaw']
                shoulderyRxyz = self.uv_to_XY(shoulderyR['x'], shoulderyR['y'], shoulderyR['z'])

                shoulderyL = self.body['L_shoulder_yaw']
                shoulderyLxyz = self.uv_to_XY(shoulderyL['x'], shoulderyL['y'], shoulderyL['z'])

                shoulderpR = self.body['R_shoulder_pitch']
                shoulderpRxyz = self.uv_to_XY(shoulderpR['x'], shoulderpR['y'], shoulderpR['z'])

                shoulderpL = self.body['L_shoulder_pitch']
                shoulderpLxyz = self.uv_to_XY(shoulderpL['x'], shoulderpL['y'], shoulderpL['z'])

                shoulderR = self.body['R_shoulder']
                shoulderRxyz = self.uv_to_XY(shoulderR['x'], shoulderR['y'], shoulderR['z'])
                
                shoulderL = self.body['L_shoulder']
                shoulderLxyz = self.uv_to_XY(shoulderL['x'], shoulderL['y'], shoulderL['z'])

                elbowR = self.body['R_elbow']
                elbowRxyz = self.uv_to_XY(elbowR['x'], elbowR['y'], elbowR['z'])

                elbowL = self.body['L_elbow']
                elbowLxyz = self.uv_to_XY(elbowR['x'], elbowR['y'], elbowR['z'])

                wristR = self.body['R_wrist']
                wristRxyz = self.uv_to_XY(wristR['x'], wristR['y'], wristR['z'])
                
                wristL = self.body['L_wrist']
                wristLxyz = self.uv_to_XY(wristL['x'], wristL['y'], wristL['z'])

                hipyR = self.body['R_hip_yaw']
                hipyRxyz = self.uv_to_XY(hipyR['x'], hipyR['y'], hipyR['z'])

                hipyL = self.body['L_hip_yaw']
                hipyLxyz = self.uv_to_XY(hipyL['x'], hipyL['y'], hipyL['z'])

                hippR = self.body['R_hip_pitch']
                hippRxyz = self.uv_to_XY(hippR['x'], hippR['y'], hippR['z'])

                hippL = self.body['L_hip_pitch']
                hippLxyz = self.uv_to_XY(hippL['x'], hippL['y'], hippL['z'])

                hipR = self.body['R_hip']
                hipRxyz = self.uv_to_XY(hipR['x'], hipR['y'], hipR['z'])
                
                hipL = self.body['L_hip']
                hipLxyz = self.uv_to_XY(hipL['x'], hipL['y'], hipL['z'])

                waist = self.body['waist']
                waistxyz = self.uv_to_XY(waist['x'], waist['y'], waist['z'])



                self.SendTransform2tf(p=bodyxyz, parent_frame='panda_2/realsense', child_frame=body['cf']) 
                transform = self.GetCameraTrans('world','body_default')
                #R_tmp = np.transpose(q2r([1, transform.rotation.x, transform.rotation.y, transform.rotation.z]))
                #print(transform)
                print(f"{Fore.GREEN}EX: {elbowRxyz[0]}|SX: {shoulderRxyz[0]}\nEY: {elbowRxyz[1]}|SY: {shoulderRxyz[1]}\nEZ:{elbowRxyz[2]}|SZ: {shoulderRxyz[2]}")
                rotation = q2r([transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z])
                translation = [transform.translation.x, transform.translation.y, transform.translation.z]
                tr = np.eye(3)
                tr = np.transpose(rotation)
                #tr[0:3, -1] = translation
                rottorso = math.atan(shoulderLxyz[2]-shoulderRxyz[2]/shoulderLxyz[0]-shoulderRxyz[0])+math.pi/2
                pos_torso = np.eye(3)
                
                posfin = pos_torso@tr

                q_fin = r2q(posfin)

                waisttrans = self.transToParent_xyz('body', 'waist')
                shoulderRtransy = self.transToParent_xyz('torso', 'R_shoulder_yaw')
                shoulderRroty=self.transToParent_PY('R_shoulder_yaw')
                shoulderRtransp = self.transToParent_xyz('R_shoulder_yaw', 'R_shoulder_pitch')
                shoulderRrotp=self.transToParent_PY('R_shoulder_pitch')
                shoulderRtrans = self.transToParent_xyz('R_shoulder_pitch', 'R_shoulder')
                
                shoulderLtransy = self.transToParent_xyz('torso', 'L_shoulder_yaw')
                shoulderLroty=self.transToParent_PY('L_shoulder_yaw')
                shoulderLtransp = self.transToParent_xyz('L_shoulder_yaw', 'L_shoulder_pitch')
                shoulderLrotp=self.transToParent_PY('L_shoulder_pitch')
                shoulderLtrans = self.transToParent_xyz('L_shoulder_pitch', 'L_shoulder')
                
                ewlbowRtrans = self.transToParent_xyz('R_shoulder', 'R_elbow')
                ewlbowLtrans = self.transToParent_xyz('L_shoulder', 'L_elbow')

                wristRtrans = self.transToParent_xyz('R_elbow', 'R_shoulder')
                wristLtrans = self.transToParent_xyz('L_elbow', 'L_shoulder')

                hipRtransy = self.transToParent_xyz('waist', 'R_hip_yaw')
                hipRroty=self.transToParent_PY('R_hip_yaw')
                hipRtransp = self.transToParent_xyz('R_hip_yaw', 'R_hip_pitch')
                hipRrotp=self.transToParent_PY('R_hip_pitch')
                hipRtrans = self.transToParent_xyz('R_hip_pitch', 'R_hip')
                
                hipLtransy = self.transToParent_xyz('waist', 'L_hip_yaw')
                hipLroty=self.transToParent_PY('L_hip_yaw')
                hipLtransp = self.transToParent_xyz('L_hip_yaw', 'L_hip_pitch')
                hipLrotp=self.transToParent_PY('L_hip_pitch')
                hipLtrans = self.transToParent_xyz('L_hip_pitch', 'L_hip')


                
                #self.SendTransform2tf(p=bodyxyz, parent_frame='panda_2/realsense', child_frame=body['cf']) 
                #R_tmp = np.eye(3)@rot_z(np.pi/2)
                #q_tmp = r2q(R_tmp)
                self.SendTransform2tf(p=torsoxyz, q=q_fin, parent_frame=torso['pf'], child_frame=torso['cf']) 
                self.SendTransform2tf(p=waistxyz, q=q_fin, parent_frame=waist['pf'], child_frame=waist['cf']) 

                self.SendTransform2tf(p=shoulderRtransy, q=shoulderRroty, parent_frame=shoulderyR['pf'], child_frame=shoulderyR['cf']) 
                self.SendTransform2tf(p=shoulderRtransp, q=shoulderRrotp, parent_frame=shoulderpR['pf'], child_frame=shoulderpR['cf']) 
                self.SendTransform2tf(p=shoulderRtrans, parent_frame=shoulderR['pf'], child_frame=shoulderR['cf']) 

                self.SendTransform2tf(p=shoulderLtransy, q=shoulderLroty, parent_frame=shoulderyL['pf'], child_frame=shoulderyL['cf']) 
                self.SendTransform2tf(p=shoulderLtransp, q=shoulderLrotp, parent_frame=shoulderpL['pf'], child_frame=shoulderpL['cf']) 
                self.SendTransform2tf(p=shoulderLtrans, parent_frame=shoulderL['pf'], child_frame=shoulderL['cf']) 

                self.SendTransform2tf(p=ewlbowRtrans, parent_frame=elbowR['pf'], child_frame=elbowR['cf']) 
                self.SendTransform2tf(p=ewlbowLtrans, parent_frame=elbowL['pf'], child_frame=elbowL['cf']) 

                self.SendTransform2tf(p=wristRtrans, parent_frame=wristR['pf'], child_frame=wristR['cf']) 
                self.SendTransform2tf(p=wristLtrans, parent_frame=wristL['pf'], child_frame=wristL['cf']) 

                self.SendTransform2tf(p=hipRtransy, q=hipRroty, parent_frame=hipyR['pf'], child_frame=hipyR['cf']) 
                self.SendTransform2tf(p=hipRtransp, q=hipRrotp, parent_frame=hippR['pf'], child_frame=hippR['cf']) 
                self.SendTransform2tf(p=hipRtrans, parent_frame=hipR['pf'], child_frame=hipR['cf']) 

                self.SendTransform2tf(p=hipLtransy, q=hipLroty, parent_frame=hipyL['pf'], child_frame=hipyL['cf']) 
                self.SendTransform2tf(p=hipLtransp, q=hipLrotp, parent_frame=hippL['pf'], child_frame=hippL['cf']) 
                self.SendTransform2tf(p=hipLtrans, parent_frame=hipL['pf'], child_frame=hipL['cf'])  """

                

       

                if self.args.circles == True:
                    self.circle_DEPTH = cv2.circle(self.img_blur_DEPTH, (self.body['L_wrist']['x'], self.body['L_wrist']['y']), radius=10, color=(255, 0, 255), thickness=2)
                    self.circle_DEPTH = cv2.circle(self.circle_DEPTH, (self.body['R_wrist']['x'], self.body['R_wrist']['y']), radius=10, color=(255, 0, 255), thickness=2)
                    self.out_DEPTH = CvBridge().cv2_to_imgmsg(self.circle_DEPTH, encoding = '16UC1')

            else:
                self.out_DEPTH = CvBridge().cv2_to_imgmsg(self.img_DEPTH, encoding = '16UC1')

            self.pub_DEPTH.publish(self.out_DEPTH)

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

    def GetCameraTrans(self, from_sys, to_sys):
        trans= self.tfbuffer.lookup_transform(from_sys, to_sys, rospy.Time())
        transform = trans.transform
        return transform
        
    #TODO: Napisi funkcijo, ki bo trensformirala samo rotacije na bazi pravil, ki jih mas na listu

    def calculate_Rot(self, joint_1:list, joint_2:list):
        #print(f"{Fore.RED} J1: {joint_1}|{Fore.BLUE} J2: {joint_2}")
        
        #roll = math.atan((joint_1[0]-joint_2[0])/(joint_1[1]-joint_2[1]))
        #yaw = math.atan((joint_1[2]-joint_2[2])/(joint_1[1]-joint_2[1]))
        #pitch = math.atan((joint_1[0]-joint_2[0])/(joint_1[2]-joint_2[2]))
        rot = [None, None, None]
        rot[0] = math.atan((joint_1[2]-joint_2[2])/(joint_1[1]-joint_2[1]))
        rot[1] = math.atan((joint_1[0]-joint_2[0])/(joint_1[2]-joint_2[2]))
        rot[2] = math.atan((joint_1[0]-joint_2[0])/(joint_1[1]-joint_2[1]))
        
        #rot = rpy2r([0, rotz, rotx])
        #print(f"{Fore.LIGHTRED_EX}angle X: {rot[2]}\nangle Y: {rot[0]}\nangle Z: {rot[1]}\n")

        rotm_x = rot_x(rot[2])
        rotm_y = rot_y(rot[0])
        rotm_z = rot_z(rot[1])

        #roty = math.atan((joint_1[0]-joint_2[0])/(joint_1[1]-joint_2[1]))
        #rotx = math.atan((joint_1[2]-joint_2[2])/(joint_1[1]-joint_2[1]))
        #rotz = math.atan((joint_1[0]-joint_2[0])/(joint_1[2]-joint_2[2]))
        return rotm_x, rotm_y, rotm_z

    def depth_remap(self, depth):
        self.adj_DEPTH = 65536- depth
        self.range_16b = 65536
        self.range_depth = 300 - 3
        self.remapped = (depth * self.range_depth) / self.range_16b 

        #return self.remapped
        return self.remapped/20 + 0.3
    
    def uv_to_XY(self, u,v, z):
        """Convert pixel coordinated (u,v) from realsense camera into real world coordinates X,Y,Z """
        fx = 607.167297
        fy = 608.291809

        x = (u - (326.998790)) / fx
        y = (v - (244.887363)) / fy

        X = (z * x)
        Y = (z * y)
        Z = z
        return [X, Y, Z]

    def transToParent_xyz(self, parent:str, child:str):
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

    def transToParent_PY(self, child:str):
        upperlink = self.body[child]
        lowerj = self.body[child]['lower_j']
        lowerlink = self.body[lowerj]

        rotm_x, rotm_y, rotm_z = self.calculate_Rot([upperlink['x'], upperlink['y'], upperlink['z']], [lowerlink['x'], lowerlink['y'], lowerlink['z']])
        m_res = np.eye(3)
        if upperlink['rot_z'] == True:
            m_res = m_res @ rotm_x

        if upperlink['rot_y'] == True:
            m_res = m_res @ rotm_z
        
        q_res = r2q(m_res)
        return q_res
    
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

    def create_service_client(self):
        try:
            print("waiting for service:" + self.poseNode)
            rospy.wait_for_service(self.poseNode, 2) # 2 seconds
        except rospy.ROSException as e:
            print("Couldn't find to service! " + self.poseNode)
        self.camera_service = rospy.ServiceProxy(self.poseNode, SetBool)

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
    SingleImageAlphaPose(args, cfg)