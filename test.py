from __future__ import division
import time
import torch
import numpy as np
from torch.autograd import Variable
import models.networks
from options.test_options import TestOptions 
import sys
from data.data_loader import *
from models.models import create_model
import random
from tensorboardX import SummaryWriter
from util import DATA_PATH
from pathlib import Path
import os.path
from os import path

EVAL_BATCH_SIZE = 2
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

root = '/'


if opt.dataset == 'interiornet':
    eval_list_path = root + '/phoenix/S6/wx97/interiornet/test_interiornet_normal_list.txt'
    eval_num_threads = 3
    test_data_loader = CreateInteriorNetryDataLoader(opt, eval_list_path, 
                                                    False, EVAL_BATCH_SIZE, 
                                                    eval_num_threads)
    test_dataset = test_data_loader.load_data()
    test_data_size = len(test_data_loader)
    print('========================= InteriorNet Test #images = %d ========='%test_data_size)


elif opt.dataset == 'scannet':
    eval_list_path = os.path.join(DATA_PATH, 'test_scannet_normal_list.txt')
    eval_num_threads = 1
    test_data_loader = CreateScanNetDataLoader(opt, eval_list_path, 
                                                    False, EVAL_BATCH_SIZE, 
                                                    eval_num_threads)
    test_dataset = test_data_loader.load_data()
    test_data_size = len(test_data_loader)
    print('========================= ScanNet eval #images = %d ========='%test_data_size)


else:
    print('INPUT DATASET DOES NOT EXIST!!!')
    sys.exit()

model = create_model(opt, _isTrain=False)
model.switch_to_train()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
global_step = 0


def test_numerical(model, dataset, global_step):
    rot_e_list = []
    roll_e_list = []
    pitch_e_list = []
    time_list = []

    count = 0.0

    model.switch_to_eval()

    count = 0

    for i, data in enumerate(dataset):
        stacked_img = data[0]
        targets = data[1]
        
        image_path = os.path.join(DATA_PATH, targets["img_path"][0])
        if(path.exists(image_path)):

            start_ = time.perf_counter_ns()
            rotation_error, roll_error, pitch_error = model.test_angle_error(stacked_img, targets)
            end_ = time.perf_counter_ns()
            total_time = (end_ - start_)*1e-6

            pred_roll, pred_pitch, gt_roll, gt_pitch, est_up_n, gt_up_n = model.test_roll_pitch(stacked_img, targets)
            
            print("pred_roll: ", pred_roll*180/3.1416)
            print("pred_pitch: ", pred_pitch*180/3.1416)
            print("gt_roll: ", gt_roll*180/3.1416)
            print("gt_pitch: ", gt_pitch*180/3.1416)

            # save predicted roll and pitch
            output_path_pred = image_path.split("/rgb/", 1)[0] + "/pose_pred/"
            output_path_gt = image_path.split("/rgb/", 1)[0] + "/pose_gt/"
            output_path_grav_pred = image_path.split("/rgb/", 1)[0] + "/gravity_pred/"
            output_path_grav_gt = image_path.split("/rgb/", 1)[0] + "/gravity_gt/"
            file_nr = image_path.split("/rgb/", 1)[1].split(".png", 1)[0]
            Path(output_path_pred).mkdir(parents=True, exist_ok=True)
            Path(output_path_gt).mkdir(parents=True, exist_ok=True)
            Path(output_path_grav_pred).mkdir(parents=True, exist_ok=True)
            Path(output_path_grav_gt).mkdir(parents=True, exist_ok=True)

            textfile_pred = open(output_path_pred + file_nr + ".txt", "w")
            textfile_pred.write(str(pred_roll) + " " + str(pred_pitch))
            textfile_pred.close()

            textfile_gt = open(output_path_gt + file_nr + ".txt", "w")
            textfile_gt.write(str(gt_roll) + " " + str(gt_pitch))
            textfile_gt.close()

            textfile_grav_pred = open(output_path_grav_pred + file_nr + ".txt", "w")
            textfile_grav_pred.write(str(est_up_n[0]) + " " + str(est_up_n[1]) + " " + str(est_up_n[2]))
            textfile_grav_pred.close()

            textfile_grav_gt = open(output_path_grav_gt + file_nr + ".txt", "w")
            textfile_grav_gt.write(str(gt_up_n[0]) + " " + str(gt_up_n[1]) + " " + str(gt_up_n[2]))
            textfile_grav_gt.close()

            rot_e_list = rot_e_list + rotation_error    
            roll_e_list = roll_e_list + roll_error
            pitch_e_list = pitch_e_list + pitch_error
            
            if i>1:
                time_list.append(total_time)

            rot_e_arr = np.array(rot_e_list)
            roll_e_arr = np.array(roll_e_list)
            pitch_e_arr = np.array(pitch_e_list)
            time_arr = np.array(time_list)


    rot_e_arr = np.array(rot_e_list)
    roll_e_arr = np.array(roll_e_list)
    pitch_e_arr = np.array(pitch_e_list)
    time_arr = np.array(time_list)

    mean_rot_e = np.mean(rot_e_arr)
    median_rot_e = np.median(rot_e_arr)
    std_rot_e = np.std(rot_e_arr)

    mean_roll_e = np.mean(roll_e_arr)
    median_roll_e = np.median(roll_e_arr)
    std_roll_e = np.std(roll_e_arr)

    mean_pitch_e = np.mean(pitch_e_arr)
    median_pitch_e = np.median(pitch_e_arr)
    std_pitch_e = np.std(pitch_e_arr)

    mean_time = np.mean(time_arr[:-1])
    median_time = np.median(time_arr[:-1])
    std_time = np.std(time_arr[:-1])
    
    print('======================= FINAL STATISCIS ==========================')
    print('mean_rot_e ', mean_rot_e)
    print('median_rot_e ', median_rot_e)
    print('std_rot_e ', std_rot_e)

    print('mean_roll_e ', mean_roll_e)
    print('median_roll_e ', median_roll_e)
    print('std_roll_e ', std_roll_e)

    print('mean_pitch_e ', mean_pitch_e)
    print('median_pitch_e ', median_pitch_e)    
    print('std_pitch_e ', std_pitch_e)

    print('mean_time ', mean_time)
    print('median_time ', median_time)    
    print('std_time ', std_time)


test_numerical(model, test_dataset, global_step)