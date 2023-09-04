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
from std_msgs.msg import Bool, String
from proxmsg.msg import PandaProx
from std_srvs.srv import SetBool 
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import MarkerArray, Marker
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
parser.add_argument('--markers', help='TRUE -> Display markers all the time | FALSE -> Display markers only in calibration',
                    default=False, action='store_true')
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

####################################################################################
class SingleImageAlphaPose():
    def __init__(self, args, cfg):
        self.IMAGE_HEIGHT = None
        self.IMAGE_WIDTH = None
        self.highRes = None
        self.args = args
        self.cfg = cfg
        self.image = None
        self.colorTopic = None
        self.depthTopic = None
        self.camSel = False
        self.tfFrame = 'rs_top'
        self.enablePose = False
        self.enableCamPose = True
        self.camPose = None
        self.corners = None
        self.body = None
        self.poseNode = '/realsense/alphapose/enable'
        self.camPoseNode = '/realsense_top/get_pose'
        self.camSelectPanda = '/realsense/alphapose/selectPanda'
        rospy.Service(self.poseNode, SetBool, self.enablePose_CB)
        rospy.Service(self.camPoseNode, SetBool, self.enableCamPose_CB)
        rospy.Service(self.camSelectPanda, SetBool, self.selectPandaCB)

        self.create_service_client(self.poseNode)
        self.create_service_client(self.camPoseNode)
        self.create_service_client(self.camSelectPanda)
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
        #self.initBodyDict()
        self.initProxDict()
        self.initMarkerDict()
        #init rospy
        #print(f"{Fore.RED}{self.colorTopic}{self.depthTopic}")
        self.initRosPy()

        rospy.spin()
        
####################################################################################

    def pose_CB(self, input):
        if self.enablePose and self.camSel:
            self.img_POSE = CvBridge().imgmsg_to_cv2(input, desired_encoding='rgb8')
            
            self.IMAGE_HEIGHT = self.img_POSE.shape[0]
            self.IMAGE_WIDTH = self.img_POSE.shape[1]
            print(f"{Fore.MAGENTA} COLOUR size = {self.IMAGE_HEIGHT, self.IMAGE_WIDTH}")
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
                
            if self.enableCamPose == True and self.colorTopic == '/realsense_top/color/image_raw':
                self.vis_POSE = self.markerHandler(image=self.vis_POSE)
            self.markerPub()
            print(f"{Fore.LIGHTBLUE_EX} -------{type(self.vis_POSE)}-------")
            self.out_POSE = CvBridge().cv2_to_imgmsg(self.vis_POSE, encoding = 'rgb8')
            self.pub_POSE.publish(self.out_POSE)
            

    def depth_CB(self, input):
        
        if self.enablePose and self.camSel:
            self.img_DEPTH = CvBridge().imgmsg_to_cv2(input, desired_encoding='16UC1')
            print(f"{Fore.MAGENTA} DEPTH size  PRE= {self.img_DEPTH.shape[0], self.img_DEPTH.shape[1]}")
            if self.img_DEPTH.shape[0] != self.IMAGE_HEIGHT:
                self.img_blur_DEPTH = cv2.resize(self.img_DEPTH, dsize=[self.IMAGE_WIDTH, self.IMAGE_HEIGHT])
                self.highRes = True
            else:
                self.img_blur_DEPTH = cv2.GaussianBlur(self.img_DEPTH, (5,5), cv2.BORDER_DEFAULT)
                self.highRes = False
            #print(f"{Fore.GREEN}Depth: {self.img_blur_DEPTH[350, 240]}")
            #self.colourised = cv2.cvtColor(self.img_DEPTH, cv2.COLOR_GRAY2RGB)
            #self.vis_DEPTH = self.vis(self.img_DEPTH, self.pose)
            
            print(f"{Fore.MAGENTA} DEPTH size  POST= {self.img_blur_DEPTH.shape[0], self.img_blur_DEPTH.shape[1]}")
            #print(f"{Fore.YELLOW} {self.img_DEPTH}")

            if self.camPose != None and self.camSel == True:
                self.SendTransform2tf(p=self.camPose,q=self.camRot, parent_frame="/world", child_frame=self.tfFrame)
                #self.SendTransform2tf(p=[0, -1.9, 2.80],q=self.camRot, parent_frame="/world", child_frame=self.tfFrame)
                
                # q=self.camRot,
                        
            if self.pose != None:
                for key, joint in self.body.items():
                    #print(key)
                    if joint['y'] >= self.IMAGE_HEIGHT:
                        joint['z'] = self.img_blur_DEPTH[self.IMAGE_HEIGHT-1, int(joint['x'])]/1000
                        #joint['z'] = self.img_blur_DEPTH[479, int(joint['x'])]
                    elif joint['x'] >= self.IMAGE_WIDTH:
                        joint['z'] = self.img_blur_DEPTH[int(joint['y']), self.IMAGE_WIDTH-1]/1000
                    else:
                        joint['z'] = self.img_blur_DEPTH[int(joint['y']), int(joint['x'])]/1000
                        #joint['z'] = self.img_blur_DEPTH[int(joint['y']), int(joint['x'])]
                    #print('Joint z: ', joint['z'])
                    
                    
                    joint['qx'] = np.roll(joint['qx'], -1)
                    np.put(joint['qx'], 4, joint['x'])
                    
                            
                    joint['qy'] = np.roll(joint['qy'], -1)
                    np.put(joint['qy'], 4, joint['y'])
                    
                            
                    joint['qz'] = np.roll(joint['qz'], -1)
                    np.put(joint['qz'], 4, joint['z'])
                    

                    """ print(f"{Fore.WHITE}------------------------------------------------")
                    print(f"{Fore.CYAN}Sklep: {key}")

                    print(f"{Fore.RED}X: {joint['x']}")
                    #print(f"{Fore.RED}Joint X: {joint['qx']}")

                    print(f"{Fore.GREEN}X: {joint['y']}")
                    #print(f"{Fore.RED}Joint Y: {joint['qy']}")

                    print(f"{Fore.BLUE}X: {joint['z']}")
                    #print(f"{Fore.RED}Joint Z: {joint['qz']}")
                    
                    print(f"{Fore.WHITE}------------------------------------------------") """
                    #print(joint['y'])
                    #print(f"{Fore.LIGHTBLUE_EX}{key}\n  Qx: {joint['qx']}\n  Qy: {joint['qy']}\n  Qz: {joint['qz']}")

                print(f"{Fore.CYAN}LEFT:\nDEPTH: {self.body['L_wrist']['z']} | LOCATION: {self.body['L_wrist']['x'], self.body['L_wrist']['y']}")
                print(f"{Fore.MAGENTA}RIGHT:\nDEPTH: {self.body['R_wrist']['z']} | LOCATION: {self.body['R_wrist']['x'], self.body['R_wrist']['y']}")

                self.maxdepth_loc, self.mindepth_loc = np.unravel_index(np.argmax(self.img_blur_DEPTH),self.img_blur_DEPTH.shape), np.unravel_index(np.argmin(self.img_blur_DEPTH), self.img_blur_DEPTH.shape)


                self.rounddepth_L = str(self.body['L_wrist']['z'])[:4]
                self.rounddepth_R = str(self.body['L_wrist']['z'])[:4]

                print(f"{Fore.GREEN} Max depth: {self.maxdepth_loc} {Fore.RED} | Min depth: {self.mindepth_loc}")
                #print(f"{Fore.LIGHTYELLOW_EX} RAW left: {self.img_blur_DEPTH[self.body['L_wrist']['x'], self.body['L_wrist']['y']]} | RAW right: {self.img_blur_DEPTH[self.body['R_wrist']['x'], self.body['R_wrist']['y']]}")
                
                for key, joint in self.body.items():
                    jointx = self.GetMoveAvg(joint['qx'])        
                    jointy = self.GetMoveAvg(joint['qy'])
                    jointz = self.GetMoveAvg(joint['qz'])
                   

                    #jointxyz = self.uv_to_XY(jointx, jointy, jointz)
                    jointxyz = self.uv_to_XY(jointx, jointy, jointz)

                    

                    if joint['cf'] != None:
                        self.SendTransform2tf(p=jointxyz, parent_frame=self.tfFrame, child_frame=(joint['cf']+'/rs'))
                        if key == 'body':
                            transToWorld = self.GetTrans('world',joint['cf']+'/rs')
                            transJoint = [transToWorld.translation.x, transToWorld.translation.y, transToWorld.translation.z]
                            #print(f"{Fore.RED} Rot: {transToWorld.rotation}")
                            #rotJoint = [transToWorld.rotation.w, transToWorld.rotation.x, transToWorld.rotation.y, transToWorld.rotation.z]
                            rot = rot_z(phi=-90, unit='deg')
                            
                            rotq = r2q(rot)
                            
                            #self.SendTransform2tf(p=transJoint, parent_frame=joint['pf'], child_frame=joint['cf'])
                        else: 
                            #print(f"{Fore.RED}Append to: {key}\n{joint}")
                            #jointxyz = self.uv_to_XY(joint['x'], joint['y'], joint['z'])
                            
                            
                            #transToWorld = self.GetTrans('world',joint['cf']+'/rs')
                    
                            worldPos = transToWorld.translation
                            #print(f"{Fore.LIGHTBLACK_EX}Joint: {key} | Abs position: {worldPos}")
                            
                            joint['worldx'] = -worldPos.x
                            joint['worldy'] = -worldPos.y
                            joint['worldz'] = -worldPos.z
                            #print(key, joint['worldz'])

                            jointRot = self.getRot(joint=key)
                            if joint['transj'] != None:
                                transJoint = joint['transj']
                            else:
                                transJoint = [0,0,0]
                            #self.SendTransform2tf(p=transJoint, q=jointRot, parent_frame=joint['pf'], child_frame=joint['cf'])
                            
                            #print(key,'World Z: ', joint['worldz'])
                if self.args.circles == True:
                    
                    self.circle_DEPTH = cv2.circle(self.img_blur_DEPTH, (self.body['L_wrist']['x'], self.body['L_wrist']['y']), radius=10, color=(255, 0, 255), thickness=2)
                    self.circle_DEPTH = cv2.circle(self.circle_DEPTH, (self.body['R_wrist']['x'], self.body['R_wrist']['y']), radius=10, color=(255, 0, 255), thickness=2)
                    self.out_DEPTH = CvBridge().cv2_to_imgmsg(self.circle_DEPTH, encoding = '16UC1')

            else:
                self.out_DEPTH = CvBridge().cv2_to_imgmsg(self.img_DEPTH, encoding = '16UC1')

            
                
            self.pub_DEPTH.publish(self.out_DEPTH)
            if self.colorTopic == '/realsense_top/color/image_raw':
                self.visMarker()
            self.getProximity()


            
    def arucoInit(self,dict:str):
        """
        This function initialises the ArUco dictionaries and prepares the one
        that will be used program-wide

        Args
        ----
            dict(str) : Specifies the dictionary we want to use
        """
        
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

    def initRosPy(self):
        """
        This function initialises all of the publishers and subsribers
        required by the program.

        Args
        ----
            color_topic(str) : ROS topic publishing color images, used by AlphaPose

            depth_topic(str) : ROS topic publishing depth images, used to determine 3D location of joints
        """
        rospy.init_node("vision", anonymous = True)

        self.pub_TRANS_POSE = tf2_ros.TransformBroadcaster()
        self.transmsg = geometry_msgs.msg.TransformStamped()
        self.tfbuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfbuffer)
        self.camInfo('/realsense_top/color')
        
        self.maxDEPTH = rospy.get_param("/realsense_top/aligned_depth_to_color/image_raw/compressedDepth/depth_max") # Za kasnejse mapiranje globine
        while True:
            if self.colorTopic != None:
                break
        
        self.sub_POSE = rospy.Subscriber(self.colorTopic, Image, self.pose_CB)
        self.sub_DEPTH = rospy.Subscriber(self.depthTopic, Image, self.depth_CB)

        self.pub_POSE = rospy.Publisher("/alphapose_pose", Image, queue_size=1)
        self.pub_DEPTH = rospy.Publisher("/alphapose_depth", Image, queue_size=1)
        self.pub_MARKER = rospy.Publisher("/reconcell/markers", MarkerArray, queue_size=1)
        self.pub_PANDA_PROX_1 = rospy.Publisher('/alphapose/prox/panda_1', PandaProx, queue_size= 1)
        self.pub_PANDA_PROX_2 = rospy.Publisher('/alphapose/prox/panda_2', PandaProx, queue_size= 1)

    def initMarkerDict(self):
        per = 0.0235

        self.markerDict= {
            0:[
                [-0.8, 0.4, 0.825],
                [-0.8, 0.3, 0.825],
                [-0.9, 0.3, 0.825],
                [-0.9, 0.4, 0.825]
            ],
            1:[
                [-0.3, 0.4, 0.825],
                [-0.3, 0.3, 0.825],
                [-0.4, 0.3, 0.825],
                [-0.4, 0.4, 0.825]
            ],
            2:[
                [-0.2, 0.3, 0.825],
                [-0.2, 0.2, 0.825],
                [-0.3, 0.2, 0.825],
                [-0.3, 0.3, 0.825]
            ],
            3:[
                [-0.2, 0.05, 0.825],
                [-0.2, -0.05, 0.825],
                [-0.3, -0.05, 0.825],
                [-0.3, 0.05, 0.825]
            ],
            4:[
                [-0.2, -0.2, 0.825],
                [-0.2, -0.3, 0.825],
                [-0.3, -0.3, 0.825],
                [-0.3, -0.2, 0.825]
            ],
            5:[
                [0.05, -0.2, 0.825],
                [0.05, -0.3, 0.825],
                [-0.05, -0.3, 0.825],
                [-0.05, -0.2, 0.825]
            ],
            6:[
                [0.3, -0.2, 0.825],
                [0.3, -0.3, 0.825],
                [0.2, -0.3, 0.825],
                [0.2, -0.2, 0.825]
            ]
        }


        """
        This funbction initialises the body dictionary which stores the locations of all the joints in the human body
        and the information required to publish them, calculate joint angles and smooth the motion
        """

        self.body ={'R_ankle': {'x': None, 'y': None, 'z': None, 'pf': 'r_knee_default', 
                        'cf': 'r_ankle_default','rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'R_knee',
                        'transj': [0, 0, -0.44], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_ankle': {'x': None, 'y': None, 'z': None, 'pf': 'l_knee_default', 
                        'cf': 'l_ankle_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'L_knee',
                        'transj': [0, 0, -0.44], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_knee': {'x': None, 'y': None, 'z': None, 'pf': 'r_hip_default',
                        'cf': 'r_knee_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'R_ankle', 'parent': 'R_hip', 'neg': False,
                        'transj': [0, 0, -0.33], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_knee': {'x': None, 'y': None, 'z': None, 'pf': 'l_hip_default',
                        'cf': 'l_knee_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'L_ankle', 'parent': 'L_hip', 'neg': False,
                        'transj': [0, 0, -0.33], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5), 
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_hip_yaw': {'x': None, 'y': None, 'z': None, 'pf': 'waist_default',
                        'cf': 'r_y_hip_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': 'R_knee', 'parent': 'body', 'neg': True,
                        'transj': [0, -0.132, 0], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_hip_pitch': {'x': None, 'y': None, 'z': None, 'pf': 'r_y_hip_default',
                        'cf': 'r_p_hip_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': 'R_knee', 'parent': 'R_hip_yaw', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_hip': {'x': None, 'y': None, 'z': None, 'pf': 'r_p_hip_default',
                        'cf': 'r_hip_default', 'rot_x': True, 'rot_y': False, 'rot_z': False, 'lower_j': 'R_knee', 'parent': 'R_hip_pitch', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_hip_yaw': {'x': None, 'y': None, 'z': None, 'pf': 'waist_default',
                        'cf': 'l_y_hip_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': 'L_knee', 'parent': 'body', 'neg': True, 
                        'transj': [0, 0.132, 0], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_hip_pitch': {'x': None, 'y': None, 'z': None, 'pf': 'l_y_hip_default',
                        'cf': 'l_p_hip_default', 'rot_x': True, 'rot_y': False, 'rot_z': False, 'lower_j': 'L_knee', 'parent': 'L_hip_yaw', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_hip': {'x': None, 'y': None, 'z': None, 'pf': 'l_p_hip_default',
                        'cf': 'l_hip_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'L_knee', 'parent': 'L_hip_pitch', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_wrist': {'x': None, 'y': None, 'z': None, 'pf': 'r_elbow_default',
                        'cf': 'r_wrist_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'R_elbow', 
                        'transj': [0, 0, -0.22], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_wrist': {'x': None, 'y': None, 'z': None, 'pf': 'l_elbow_default',
                        'cf': 'l_wrist_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'L_elbow', 
                        'transj': [0, 0, -0.22], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_elbow': {'x': None, 'y': None, 'z': None, 'pf': 'r_shoulder_default',
                        'cf': 'r_elbow_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'R_wrist', 'parent': 'R_shoulder', 'neg': False, 
                        'transj': [0, 0, -0.352], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_elbow': {'x': None, 'y': None, 'z': None, 'pf': 'l_shoulder_default',
                        'cf': 'l_elbow_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'L_wrist', 'parent': 'L_shoulder', 'neg': False, 
                        'transj': [0, 0, -0.352], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_shoulder_yaw': {'x': None, 'y': None, 'z': None, 'pf': 'torso_default',
                        'cf': 'r_y_shoulder_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': 'R_elbow', 'parent': 'torso', 'neg': False, 
                        'transj' : [0, -0.176, 0], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_shoulder_pitch': {'x': None, 'y': None, 'z': None, 'pf': 'r_y_shoulder_default',
                        'cf': 'r_p_shoulder_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'R_elbow', 'parent': 'R_shoulder_yaw', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_shoulder': {'x': None, 'y': None, 'z': None, 'pf': 'r_p_shoulder_default',
                        'cf': 'r_shoulder_default', 'rot_x': True, 'rot_y': False, 'rot_z': False, 'lower_j': 'R_elbow', 'parent': 'R_shoulder_pitch', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_shoulder_yaw': {'x': None, 'y': None, 'z': None, 'pf': 'torso_default',
                        'cf': 'l_y_shoulder_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': 'L_elbow', 'parent': 'torso', 'neg': True, 
                        'transj' : [0, 0.176, 0], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_shoulder_pitch': {'x': None, 'y': None, 'z': None, 'pf': 'l_y_shoulder_default',
                        'cf': 'l_p_shoulder_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'L_elbow', 'parent': 'L_shoulder_yaw', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_shoulder': {'x': None, 'y': None, 'z': None, 'pf': 'l_p_shoulder_default',
                        'cf': 'l_shoulder_default', 'rot_x': True, 'rot_y': False, 'rot_z': False, 'lower_j': 'L_elbow', 'parent': 'L_shoulder_pitch', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_ear': {'x': None, 'y': None, 'z': None, 'cf': None,
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'transj' : None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_ear': {'x': None, 'y': None, 'z': None, 'cf': None,
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'transj' : None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'R_eye': {'x': None, 'y': None, 'z': None, 'cf': None,
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'transj' : None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'L_eye': {'x': None, 'y': None, 'z': None, 'cf': None,
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'transj' : None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'head': {'x': None, 'y': None, 'z': None, 'pf': 'p_head_default', 'cf': 'head_default',
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'head_pitch', 
                        'transj' : [0.055, 0, 0.11], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'head_roll': {'x': None, 'y': None, 'z': None, 'pf': 'torso_default', 'cf': 'r_head_default',
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'torso', 
                        'transj' : [0, 0, 0.055], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'head_yaw': {'x': None, 'y': None, 'z': None, 'pf': 'r_head_default', 'cf': 'y_head_default',
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'head_roll', 
                        'transj' : None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'head_pitch': {'x': None, 'y': None, 'z': None, 'pf':'y_head_default', 'cf': 'p_head_default',
                        'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'head_yaw', 
                        'transj' : None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'body': {'x': None, 'y': None, 'z': None,
                        'pf': 'world', 'cf': 'body_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 
                        'transj' : None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},

                    'torso': {'x': None, 'y': None,
                        'z': None, 'pf': 'waist_default', 'cf': 'torso_default', 'rot_x': False, 'rot_y': False, 'rot_z': False, 'lower_j': None, 'parent': 'waist', 'neg': True, 
                        'transj': [0, 0, 0.605], 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None},
                    
                    'waist': {'x': None, 'y': None,'z': None, 'pf': 'body_default', 
                        'cf': 'waist_default', 'rot_x': False, 'rot_y': True, 'rot_z': False, 'lower_j': 'torso', 'parent': 'body', 'neg': False, 
                        'transj': None, 'qx': np.ndarray(5), 'qy': np.ndarray(5), 'qz': np.ndarray(5),
                        'worldx': None, 'worldy': None, 'worldz': None}}

    def initProxDict(self):
        # indices have 'k' in front of them to avoid string slicing
        self.pandaPose1 = {
            0: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}},
            1: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}},
            2: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}}, 
            3: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}}, 
            4: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}}, 
            5: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}},
            6: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}}, 
            7: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}},
            8: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}},
        } 
        
        self.pandaPose2 = {
            0: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}}, 
            1: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}},
            2: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}}, 
            3: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}},
            4: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}}, 
            5: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}},
            6: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}}, 
            7: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}},
            8: {'POS': None,
                'PROX': {'head': None, 'L_shoulder': None, 'R_shoulder': None, 'L_elbow': None, 'R_elbow': None, 'L_wrist': None, 'R_wrist': None}},
        } 

    def markerHandler(self, image:np.ndarray):
        """
        This function detects ArUco markers from an image, draws them
        onto the original image and stores their coordinates in a dictionary.

        Aditionally the function calls  ```getCameraPose()```

        It can be configured to output the drawn-on image

        Args
        ----
            image(np.ndarray) : Input image
        """
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
            marker = []
            ids = np.ndarray.flatten(ids)
            if len(corners) == 7:
            #if len(corners) == 8:
                print(f"{Fore.BLUE} CORNERS: {len(corners)} | IDS: {ids}")
                self.corners = np.asfarray(corners).reshape(28,2)
                #self.corners = np.asfarray(corners).reshape(32,2)

            # A dictionary storing the corners of each detected marker as an array of 4
            self.cornerDict = {0: None,
                                    1: None,
                                    2: None,
                                    3: None,
                                    4: None,
                                    5: None,
                                    6: None}
            
            print(f"{Fore.GREEN} TEST")
            for (id, corner) in zip(ids,corners):
                print(f"{Fore.MAGENTA}Corner: {corner} | Type: {type(corner)}")
                corners = corner.reshape((4, 2))
                print(f"{Fore.YELLOW}Corners: {corners} | Type: {type(corners)}")
                
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = [int(topRight[0]), int(topRight[1])]
                bottomRight = [int(bottomRight[0]), int(bottomRight[1])]
                bottomLeft = [int(bottomLeft[0]), int(bottomLeft[1])]
                topLeft = [int(topLeft[0]), int(topLeft[1])]
                self.cornerDict[id] = [topRight, bottomRight, bottomLeft, topLeft]
 
                # draw the bounding box of the ArUCo detection
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                cv2.circle(image, topRight, 4, (255, 0, 0), -1)
                cv2.putText(image, str(id),
                        (topLeft[0], topLeft[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2
                        )
                
                """ for (id, corner) in zip(ids,rejected):
                    print(f"{Fore.MAGENTA}Corner: {corner} | Type: {type(corner)}")
                    corners = corner.reshape((4, 2))
                    print(f"{Fore.YELLOW}Corners: {corners} | Type: {type(corners)}")
                    
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = [int(topRight[0]), int(topRight[1])]
                    bottomRight = [int(bottomRight[0]), int(bottomRight[1])]
                    bottomLeft = [int(bottomLeft[0]), int(bottomLeft[1])]
                    topLeft = [int(topLeft[0]), int(topLeft[1])]
                    self.cornerDict[id] = [topRight, bottomRight, bottomLeft, topLeft]
                    #print(topLeft)
                    # draw the bounding box of the ArUCo detection
                    cv2.line(image, topLeft, topRight, (255, 0, 0), 2)
                    cv2.line(image, topRight, bottomRight, (255, 0, 0), 2)
                    cv2.line(image, bottomRight, bottomLeft, (255, 0, 0), 2)
                    cv2.line(image, bottomLeft, topLeft, (255, 0, 0), 2)
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                    cv2.circle(image, topRight, 4, (255, 0, 0), -1)
                    cv2.putText(image, str(id),
                            (topLeft[0], topLeft[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2
                            )
 """
            print(f"{Fore.CYAN}{self.cornerDict}")
            #print(self.cornerdict)

            self.camPose, self.camRot = self.getCamPose()
            
            #print(f"{Fore.GREEN}ROTmat: {self.camRot}")
            self.enableCamPose = False
            """ else:
                print(f"{Fore.MAGENTA}Not enough visible markers!")
                pass """
            return image
        
        else:
            return image
    
    def getProximity(self):
        for link, loc in self.pandaPose1.items():
            self.pandaPose1[link]['POS'] = self.GetTrans('world', 'panda_1/panda_1_link'+str(link)).translation
            self.pandaPose2[link]['POS'] = self.GetTrans('world', 'panda_2/panda_2_link'+str(link)).translation
            for joint, dist in self.pandaPose1[link]['PROX'].items():
                if self.body[joint]['worldx'] != None:
                    self.pandaPose1[link]['PROX'][joint] = math.sqrt((self.pandaPose1[link]['POS'].x+self.body[joint]['worldx'])**2+(self.pandaPose1[link]['POS'].y+self.body[joint]['worldy'])**2+(self.pandaPose1[link]['POS'].z+self.body[joint]['worldz'])**2)
                    #print(f"{Fore.MAGENTA}{joint}\n   {Fore.RED}X:{self.body[joint]['worldx']}\n   {Fore.GREEN}Y:{self.body[joint]['worldy']}\n   {Fore.BLUE}Z:{self.body[joint]['worldz']}")
                    self.pandaPose2[link]['PROX'][joint] = math.sqrt((self.pandaPose2[link]['POS'].x+self.body[joint]['worldx'])**2+(self.pandaPose2[link]['POS'].y+self.body[joint]['worldy'])**2+(self.pandaPose2[link]['POS'].z+self.body[joint]['worldz'])**2)
            #print(f"{Fore.LIGHTCYAN_EX}Pos 1: {self.pandaPose1[link]['POS']}{type(self.pandaPose1[link]['PROX']['head'])}\nPos 2: {self.pandaPose2[link]['POS']}")
            #print(f"{Fore.RED}{link}\n{Fore.LIGHTGREEN_EX}Prox 1: {self.pandaPose1[link]['PROX']}\Prox 2: {self.pandaPose2[link]['PROX']}")
  
        self.proxPub("Panda_1", self.pandaPose1)
        self.proxPub("Panda_2", self.pandaPose2)
        #self.pub_PANDA_PROX_2.publish(P2ProxMsg)
                


               
    
    def visMarker(self):
        """
        This function publishes msarker locations from the camera's perspective, it allows us to see if the camera pose is set properly
        """
        for key, coords in self.cornerDict.items():
            #print(f"{Fore.LIGHTMAGENTA_EX} Markerdict {key}: {Fore.GREEN}{coords}")
            if self.cornerDict[key] != None:
                topRc = coords[0]
                botLc = coords[2]
                #print(f"{Fore.LIGHTBLUE_EX} Top R: {topR}")
                #print(f"{Fore.LIGHTBLUE_EX} Top R: {botL}")
                cx = (botLc[0]+topRc[0])/2
                cy = (botLc[1]+topRc[1])/2
                cz = self.img_blur_DEPTH[int(cy), int(cx)]/1000
                visMarkerloc = self.uv_to_XY(cx, cy, cz)
                self.SendTransform2tf(p=visMarkerloc, parent_frame='rs_top', child_frame=('markerVis/'+str(key)))
                
 

    def markerPub(self):
        """
        This function is a static marker array publisher and a marker TF publisher
        for us to visualise ArUco marker locations in RViz
        """
        markerarray = MarkerArray()
        
        for key, item in self.markerDict.items():
            topR = item[0]
            botL = item[2]
            cx = (botL[0]+topR[0])/2
            cy = (botL[1]+topR[1])/2

            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = rospy.Time.now()
            marker.id = key
            marker.type = 1
            marker.action = 0
            marker.pose.position.x = cx
            marker.pose.position.y = cy
            marker.pose.position.z = 0.825
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.01
            marker.frame_locked = False
            #prit(f"{Fore.GREEN}Marker: {key} {Fore.BLUE}|{Fore.RED} Location: {cx, cy, item[0][2]}")
            markerarray.markers.append(marker) 
            #print(f"{Fore.LIGHTWHITE_EX}Item: {item[3]}")
            self.SendTransform2tf(p=[marker.pose.position.x, marker.pose.position.y, marker.pose.position.z], parent_frame='world', child_frame=('marker/'+str(key)))
        self.pub_MARKER.publish(markerarray)

    def proxPub(self, panda_name:str, prox_dict:dict):

        proxmsg = PandaProx()
        proxmsg.panda_name.data = panda_name
        
        proxmsg.joint_0.position.x = prox_dict[5]['POS'].x
        proxmsg.joint_0.position.y = prox_dict[5]['POS'].y
        proxmsg.joint_0.position.z = prox_dict[5]['POS'].z
        proxmsg.joint_0.proximity.head.dist = prox_dict[5]['PROX']['head']
        proxmsg.joint_0.proximity.L_shoulder.dist = prox_dict[5]['PROX']['L_shoulder']
        proxmsg.joint_0.proximity.R_shoulder.dist = prox_dict[5]['PROX']['R_shoulder']
        proxmsg.joint_0.proximity.head.dist = prox_dict[5]['PROX']['head']
        proxmsg.joint_0.proximity.L_elbow.dist = prox_dict[5]['PROX']['L_elbow']
        proxmsg.joint_0.proximity.R_elbow.dist = prox_dict[5]['PROX']['R_elbow']
        proxmsg.joint_0.proximity.L_wrist.dist = prox_dict[5]['PROX']['L_wrist']
        proxmsg.joint_0.proximity.R_wrist.dist = prox_dict[5]['PROX']['R_wrist']

        proxmsg.joint_1.position.x = prox_dict[1]['POS'].x
        proxmsg.joint_1.position.y = prox_dict[1]['POS'].y
        proxmsg.joint_1.position.z = prox_dict[1]['POS'].z
        proxmsg.joint_1.proximity.head.dist = prox_dict[1]['PROX']['head']
        proxmsg.joint_1.proximity.L_shoulder.dist = prox_dict[1]['PROX']['L_shoulder']
        proxmsg.joint_1.proximity.R_shoulder.dist = prox_dict[1]['PROX']['R_shoulder']
        proxmsg.joint_1.proximity.head.dist = prox_dict[1]['PROX']['head']
        proxmsg.joint_1.proximity.L_elbow.dist = prox_dict[1]['PROX']['L_elbow']
        proxmsg.joint_1.proximity.R_elbow.dist = prox_dict[1]['PROX']['R_elbow']
        proxmsg.joint_1.proximity.L_wrist.dist = prox_dict[1]['PROX']['L_wrist']
        proxmsg.joint_1.proximity.R_wrist.dist = prox_dict[1]['PROX']['R_wrist']

        proxmsg.joint_2.position.x = prox_dict[2]['POS'].x
        proxmsg.joint_2.position.y = prox_dict[2]['POS'].y
        proxmsg.joint_2.position.z = prox_dict[2]['POS'].z
        proxmsg.joint_2.proximity.head.dist = prox_dict[2]['PROX']['head']
        proxmsg.joint_2.proximity.L_shoulder.dist = prox_dict[2]['PROX']['L_shoulder']
        proxmsg.joint_2.proximity.R_shoulder.dist = prox_dict[2]['PROX']['R_shoulder']
        proxmsg.joint_2.proximity.head.dist = prox_dict[2]['PROX']['head']
        proxmsg.joint_2.proximity.L_elbow.dist = prox_dict[2]['PROX']['L_elbow']
        proxmsg.joint_2.proximity.R_elbow.dist = prox_dict[2]['PROX']['R_elbow']
        proxmsg.joint_2.proximity.L_wrist.dist = prox_dict[2]['PROX']['L_wrist']
        proxmsg.joint_2.proximity.R_wrist.dist = prox_dict[2]['PROX']['R_wrist']

        proxmsg.joint_3.position.x = prox_dict[3]['POS'].x
        proxmsg.joint_3.position.y = prox_dict[3]['POS'].y
        proxmsg.joint_3.position.z = prox_dict[3]['POS'].z
        proxmsg.joint_3.proximity.head.dist = prox_dict[3]['PROX']['head']
        proxmsg.joint_3.proximity.L_shoulder.dist = prox_dict[3]['PROX']['L_shoulder']
        proxmsg.joint_3.proximity.R_shoulder.dist = prox_dict[3]['PROX']['R_shoulder']
        proxmsg.joint_3.proximity.head.dist = prox_dict[3]['PROX']['head']
        proxmsg.joint_3.proximity.L_elbow.dist = prox_dict[3]['PROX']['L_elbow']
        proxmsg.joint_3.proximity.R_elbow.dist = prox_dict[3]['PROX']['R_elbow']
        proxmsg.joint_3.proximity.L_wrist.dist = prox_dict[3]['PROX']['L_wrist']
        proxmsg.joint_3.proximity.R_wrist.dist = prox_dict[3]['PROX']['R_wrist']

        proxmsg.joint_4.position.x = prox_dict[4]['POS'].x
        proxmsg.joint_4.position.y = prox_dict[4]['POS'].y
        proxmsg.joint_4.position.z = prox_dict[4]['POS'].z
        proxmsg.joint_4.proximity.head.dist = prox_dict[4]['PROX']['head']
        proxmsg.joint_4.proximity.L_shoulder.dist = prox_dict[4]['PROX']['L_shoulder']
        proxmsg.joint_4.proximity.R_shoulder.dist = prox_dict[4]['PROX']['R_shoulder']
        proxmsg.joint_4.proximity.head.dist = prox_dict[4]['PROX']['head']
        proxmsg.joint_4.proximity.L_elbow.dist = prox_dict[4]['PROX']['L_elbow']
        proxmsg.joint_4.proximity.R_elbow.dist = prox_dict[4]['PROX']['R_elbow']
        proxmsg.joint_4.proximity.L_wrist.dist = prox_dict[4]['PROX']['L_wrist']
        proxmsg.joint_4.proximity.R_wrist.dist = prox_dict[4]['PROX']['R_wrist']

        proxmsg.joint_5.position.x = prox_dict[5]['POS'].x
        proxmsg.joint_5.position.y = prox_dict[5]['POS'].y
        proxmsg.joint_5.position.z = prox_dict[5]['POS'].z
        proxmsg.joint_5.proximity.head.dist = prox_dict[5]['PROX']['head']
        proxmsg.joint_5.proximity.L_shoulder.dist = prox_dict[5]['PROX']['L_shoulder']
        proxmsg.joint_5.proximity.R_shoulder.dist = prox_dict[5]['PROX']['R_shoulder']
        proxmsg.joint_5.proximity.head.dist = prox_dict[5]['PROX']['head']
        proxmsg.joint_5.proximity.L_elbow.dist = prox_dict[5]['PROX']['L_elbow']
        proxmsg.joint_5.proximity.R_elbow.dist = prox_dict[5]['PROX']['R_elbow']
        proxmsg.joint_5.proximity.L_wrist.dist = prox_dict[5]['PROX']['L_wrist']
        proxmsg.joint_5.proximity.R_wrist.dist = prox_dict[5]['PROX']['R_wrist']

        proxmsg.joint_6.position.x = prox_dict[6]['POS'].x
        proxmsg.joint_6.position.y = prox_dict[6]['POS'].y
        proxmsg.joint_6.position.z = prox_dict[6]['POS'].z
        proxmsg.joint_6.proximity.head.dist = prox_dict[6]['PROX']['head']
        proxmsg.joint_6.proximity.L_shoulder.dist = prox_dict[6]['PROX']['L_shoulder']
        proxmsg.joint_6.proximity.R_shoulder.dist = prox_dict[6]['PROX']['R_shoulder']
        proxmsg.joint_6.proximity.head.dist = prox_dict[6]['PROX']['head']
        proxmsg.joint_6.proximity.L_elbow.dist = prox_dict[6]['PROX']['L_elbow']
        proxmsg.joint_6.proximity.R_elbow.dist = prox_dict[6]['PROX']['R_elbow']
        proxmsg.joint_6.proximity.L_wrist.dist = prox_dict[6]['PROX']['L_wrist']
        proxmsg.joint_6.proximity.R_wrist.dist = prox_dict[6]['PROX']['R_wrist']

        proxmsg.joint_7.position.x = prox_dict[7]['POS'].x
        proxmsg.joint_7.position.y = prox_dict[7]['POS'].y
        proxmsg.joint_7.position.z = prox_dict[7]['POS'].z
        proxmsg.joint_7.proximity.head.dist = prox_dict[7]['PROX']['head']
        proxmsg.joint_7.proximity.L_shoulder.dist = prox_dict[7]['PROX']['L_shoulder']
        proxmsg.joint_7.proximity.R_shoulder.dist = prox_dict[7]['PROX']['R_shoulder']
        proxmsg.joint_7.proximity.head.dist = prox_dict[7]['PROX']['head']
        proxmsg.joint_7.proximity.L_elbow.dist = prox_dict[7]['PROX']['L_elbow']
        proxmsg.joint_7.proximity.R_elbow.dist = prox_dict[7]['PROX']['R_elbow']
        proxmsg.joint_7.proximity.L_wrist.dist = prox_dict[7]['PROX']['L_wrist']
        proxmsg.joint_7.proximity.R_wrist.dist = prox_dict[7]['PROX']['R_wrist']
        
        proxmsg.joint_8.position.x = prox_dict[8]['POS'].x
        proxmsg.joint_8.position.y = prox_dict[8]['POS'].y
        proxmsg.joint_8.position.z = prox_dict[8]['POS'].z
        proxmsg.joint_8.proximity.head.dist = prox_dict[8]['PROX']['head']
        proxmsg.joint_8.proximity.L_shoulder.dist = prox_dict[8]['PROX']['L_shoulder']
        proxmsg.joint_8.proximity.R_shoulder.dist = prox_dict[8]['PROX']['R_shoulder']
        proxmsg.joint_8.proximity.head.dist = prox_dict[8]['PROX']['head']
        proxmsg.joint_8.proximity.L_elbow.dist = prox_dict[8]['PROX']['L_elbow']
        proxmsg.joint_8.proximity.R_elbow.dist = prox_dict[8]['PROX']['R_elbow']
        proxmsg.joint_8.proximity.L_wrist.dist = prox_dict[8]['PROX']['L_wrist']
        proxmsg.joint_8.proximity.R_wrist.dist = prox_dict[8]['PROX']['R_wrist']

        if panda_name == 'Panda_1':
            self.pub_PANDA_PROX_1.publish(proxmsg)
        elif panda_name == 'Panda_2':
            self.pub_PANDA_PROX_2.publish(proxmsg)
        else:
            print(f"{Fore.RED}INVALID NAME")
       
    def camInfo(self, camera_topic:str):
        """
        This function pulls camera parameters from the "/camera_info" topic
        - Camera matrix 
            - focal lengths: fx, fy
            - optical centres: cx, cy
        - Distortion coefficients

        Args
        ----
            camera_topic (str) : Specify from which camera we should pull the parameters
        """
        caminfo = rospy.wait_for_message(camera_topic+'/camera_info', CameraInfo, timeout=20)
        self.camera_fx = caminfo.K[0]
        self.camera_cx = caminfo.K[2]
        self.camera_fy = caminfo.K[4]
        self.camera_cy = caminfo.K[5]

        self.cameramatrix = np.array([[self.camera_fx, 0.0, self.camera_cx],
                                    [0.0, self.camera_fy, self.camera_cy],
                                    [0.0, 0.0, 1.0]],dtype=np.float32)
        
        self.camMatLowRes = np.array([[625.11944,0.0, 322.41944],
                            [0.0, 623.68514, 243.40506],
                            [0.0,0.0,1.0]], dtype=np.float32)
        
        self.camMatHighRes = np.array([[909.4385,0.0,  641.420126],
                            [0.0, 909.540780, 357.579186],
                            [0.0,0.0,1.0]], dtype=np.float32)
        
        self.camMat = np.array([[865.717580, 0.00000, 646.303204],
                            [0.0, 865.878228, 362.008238],
                            [0.0,0.0,1.0]], dtype=np.float32)
        self.fx = self.camMat[0, 0]
        self.cx = self.camMat[0, 2]
        self.fy = self.camMat[1, 1]
        self.cy = self.camMat[1, 2]
        

        #self.camMat = np.array([[496.915339, -1.383087, 635.775274],
        #                    [0.0, 486.192335, 355.610420],
        #                    [0.0,0.0,1.0]], dtype=np.float32)

        self.distCoefs = np.array([0.059732, -0.104126, -0.003368, -0.000256, 0.000000], dtype=np.float32)

        self.distCoefsLowRes = np.array([0.140602, -0.332482, -0.000699, -0.004239, 0.000000], dtype=np.float32)

        #self.distCoefsHighRes = np.array([0.114931, -0.183645 , 0.015991, -0.012492, 0.000000], dtype=np.float32)
        
        self.distCoefsHighRes = np.array([0, 0 , 0, 0, 0.000000], dtype=np.float32)
    
    def getCamPose(self):
        """ 
        This function converts image and actual locations
        of ArUco markers from dictionaries to np.arrays

        It only takes into account the values of markers,
        that were detected, therefore avoiding any false results
        and jitter that may come from spotty marker detection.

        After the conversion it calculates the camera position.

        Returns
        -------
            (list):  Translation vector from the world origin to the camera

            (list): Rotation quaternions of the camera with respect to the world origin
                [w, Rot_X, Rot_Y, Rot_Z]

        """
        cornerList = []
        markerList = []

        for marker, corners in self.cornerDict.items():
            if self.cornerDict[marker] != None:
                for subind, tup in enumerate(self.cornerDict[marker]):
                    cornerList.append(list(self.cornerDict[marker][subind]))
                    print(f"{Fore.LIGHTMAGENTA_EX}MARKER: {marker} | SUBIND: {subind}")
                    print(f"{Fore.LIGHTWHITE_EX}CORNERDICT: {self.cornerDict}")
                    print(f"{Fore.LIGHTCYAN_EX} MARKERDICT: {self.markerDict}")
                    markerList.append(self.markerDict[marker][subind])

        markerArray = np.asfarray(markerList, dtype=np.float32)
        cornerArray = np.asfarray(cornerList, dtype=np.float32)

        """if self.highRes == True:
            cmat = self.camMatHighRes
            dcoef = self.distCoefsHighRes
        else:
            cmat = self.camMatLowRes
            dcoef = self.distCoefsLowRes """
        cmat = self.camMat
        dcoef = self.distCoefs

        flag = cv2.SOLVEPNP_ITERATIVE 
        #flag = cv2.SOLVEPNP_IPPE
        print(f"{Fore.GREEN}Marker array: {markerArray}| Length: {Fore.LIGHTGREEN_EX}{markerArray.shape} | Type: {type(markerArray[0][0])}")
        print(f"{Fore.RED}Corner array: {cornerArray}| Length: {Fore.LIGHTRED_EX}{cornerArray.shape} | Type: {type(cornerArray[0][0])}")
        retval, self.rvec, self.tvec = cv2.solvePnP(markerArray, cornerArray, cmat, dcoef, flags=flag)
        print(f"{Fore.RED}################################################\n DIAGNOSTIC:\n {Fore.BLUE} TVEC: {self.tvec}\n {Fore.GREEN}RVEC:{self.rvec}\n################################################")

        rotm = np.zeros((3,3)) 
        cv2.Rodrigues(self.rvec, rotm)
        #rotm = rot_x(-135, unit='deg')
        rotm = np.transpose(rotm)
        rotq = r2q(rotm)
        print('RotQ: ',rotq, 'RotM: ',rotm )
        
        #Publish temporary pose so compensation can happen
        camPose = [-self.tvec[0], -self.tvec[1], self.tvec[2]]
        self.SendTransform2tf(p=camPose,q=rotq, parent_frame="/world", child_frame="/rs_top")
        correctedPose = self.CamPoseCorrection(initial_pose=camPose)
        #correctedPose = [0, -1.6, 2.8]
        #correctedPose = camPose

        return correctedPose, rotq

    def CamPoseCorrection(self, initial_pose:list)->list:
        """
        This function autocorrects the camera position based on the deviation from real world marker 
        positions which are hardcoded. It shifts the camera position in all 3 axes by the deviation ammount
        for each specific axis.

        Args
        ----
            initial_pose(list) : Coordinates of the camera, calculated using the solve pnp function

        Returns
        -------
            list : Corrected camera pose array [x+devX, y+devY, z+devZ]
        """
        errorx = []
        errory = []
        errorz = []
        #print(f"{Fore.MAGENTA}Coords: {self.cornerDict}")
        for key, coords in self.cornerDict.items():
            if coords != None:
                print(f"{Fore.BLUE}Key: {key}| Coords: {self.cornerDict[key]}")
                # First publish the uncompensated marker positions as the camera sees
                topRc = coords[0]
                botLc = coords[2]
                cx = (botLc[0]+topRc[0])/2
                cy = (botLc[1]+topRc[1])/2
                cz = self.img_blur_DEPTH[int(cy), int(cx)]/1000
                visMarkerloc = self.uv_to_XY(cx, cy, cz)
                self.SendTransform2tf(p=visMarkerloc, parent_frame='rs_top', child_frame=('markerVis/'+str(key)))

                # Calculate center of corresponding marker in real world coordinates
                topRw = self.markerDict[key][0]
                botLw = self.markerDict[key][2]
                wx = (botLw[0]+topRw[0])/2
                wy = (botLw[1]+topRw[1])/2
                wz = botLw[2]

                # Get real world coordinates from cameras view of the markers
                worldloc = self.tfbuffer.lookup_transform('world','markerVis/'+str(key), time=rospy.Time())
                wcx, wcy, wcz = worldloc.transform.translation.x, worldloc.transform.translation.y, worldloc.transform.translation.z

                errorx.append(wx-wcx)
                errory.append(wy-wcy)
                errorz.append(wz-wcz)
        # Get mean error for each axis
        mx = np.mean(errorx)
        my = np.mean(errory)
        mz = np.mean(errorz)
        print(f"{Fore.RED}e X: {mx} \n{Fore.GREEN}e Y: {my} \n{Fore.BLUE}e Z: {mz}")
        
        return [initial_pose[0]+mx, initial_pose[1]+my, initial_pose[2]]
        


    def SendTransform2tf(self, p:list=[0,0,0],q:list=[1,0,0,0], parent_frame:str= "panda_2/realsense",child_frame:str="Human_Pose"):
        """
        This functions publishes a point to a TF topic, the point can be seen in RViz

        Args
        ----
            p(list) : Translation vector [x, y, z]

            q(list) : Quaternion rotation vector [w, Rot_X, Rot_Y, Rot_Z]

            parent_frame(str) : The point with respect to which we specify the translation vector

            child_frame(str) : The name of the resulting TF point
        """
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

    def GetTrans(self, from_sys:str, to_sys:str):
        """
        This functon looks up the transform from any specified frame to any other specified frame

        Args
        ----
            from_sys(str) : Origin frame

            to_sys(str) : Destination frame

        Returns
        ------
            transformation : Transformation matrix from specified frame to specified frame
        """
        trans= self.tfbuffer.lookup_transform(from_sys, to_sys, rospy.Time())
        transform = trans.transform
        return transform


    def GetMoveAvg(self, axis:np.ndarray):
        
        return np.mean(axis)


    def getRot(self, joint:str) -> np.ndarray:
        """
        This function calculates joint angles for the URDF human model from
        the calculated/measured locations of hte keypoints

        Args
        ----
            joint(str) : Specifies for which joint the angle is being calculated

        Returns
        -------
            outRot(np,ndarray) : Rotation matrix of the specified joint
        """
        curJoint = self.body[joint]
        
        if curJoint['lower_j'] != None:
            lowJoint = self.body[curJoint['lower_j']]
            #print(f"{Fore.MAGENTA}{joint} | Z: {curJoint['worldz']}")
            #print(f"{Fore.LIGHTMAGENTA_EX}{curJoint['lower_j']} | Z: {lowJoint['worldz']}")
            
            """ curPos = [curJoint['x'], curJoint['y'], curJoint['z']]
            lowPos = [lowJoint['x'], lowJoint['y'], lowJoint['z']]

            curPos = [self.GetMoveAvg(curJoint['qx']), self.GetMoveAvg(curJoint['qy']), self.GetMoveAvg(curJoint['qz'])]
            lowPos = [self.GetMoveAvg(lowJoint['qx']), self.GetMoveAvg(lowJoint['qy']), self.GetMoveAvg(lowJoint['qz'])] """

            curPos = [curJoint['worldx'], curJoint['worldy'], curJoint['worldz']]
            lowPos = [lowJoint['worldx'], lowJoint['worldy'], lowJoint['worldz']]
            

            if curJoint['rot_x']:
                dx = curPos[0] - lowPos[0]
                dy = curPos[1] - lowPos[1]
                rot = math.atan(dx/dy)
                if curJoint['neg']:
                    rot = -rot
                outRot = r2q(rot_x(rot))
            
            elif curJoint['rot_y']:
                #print('Cur: ', curPos)
                #print('Low: ', lowPos)
                dz = curPos[2] - lowPos[2]
                dy = curPos[1] - lowPos[1]
                rot = math.atan(dz/dy)
                if curJoint['neg']:
                    rot = -rot
                outRot = r2q(rot_y(rot))
            
            elif curJoint['rot_z']:
                dx = curPos[0] - lowPos[0]
                dz = curPos[2] - lowPos[2]
                rot = math.atan(dx/dz)
                if curJoint['neg']:
                    rot = -rot
                outRot = r2q(rot_z(rot))
            else:
                outRot = r2q(np.eye(3))
        else:
            outRot = r2q(np.eye(3))

        return outRot

    def uv_to_XY(self, u:int,v:int, z:int) -> list:
        """
        Convert pixel coordinated (u,v) from realsense camera
        into real world coordinates X,Y,Z 

        Args
        ----
            u(int) : Horizontal coordinate

            v(int) : Vertical coordinate

            z(int) : Depth coordinate

        Returns
        -------
            worldPos(list) : Real world position (in respect to camera)
        """
        
        #x = (u - (496.91)) / 635.7753
        x = (u - (self.cx)) / self.fx

        #y = (v - (489.182)) / 355.61024
        y = (v - (self.cy)) / self.fy

        X = (z * x)
        Y = (z * y)
        Z = z

        worldPos = [X, Y, Z]
        return worldPos
    
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

    def selectPandaCB(self, req):
        if self.colorTopic == None:
            panda = req.data
            if panda:
                print("Camera: realsense (Panda 2) selected...")
                self.colorTopic = '/realsense/color/image_raw'
                self.depthTopic = '/realsense/aligned_depth_to_color/image_raw'
                self.tfFrame = 'panda_2/realsense'
                msg = self.camSelectPanda + " Camera location: Panda 2"
            else:
                print("Camera: realsense_top (DEFAULT) selected...")
                self.colorTopic = '/realsense_top/color/image_raw'
                self.depthTopic = '/realsense_top/aligned_depth_to_color/image_raw'
                self.tfFrame = 'rs_top'
                msg = self.camSelectPanda + " Camera location: Top"
            self.camSel = True
            self.initRosPy()
            #self.sub_POSE = rospy.Subscriber(self.colorTopic, Image, self.pose_CB)
            #self.sub_DEPTH = rospy.Subscriber(self.depthTopic, Image, self.depth_CB)
            return True, msg
        else:
            pass

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