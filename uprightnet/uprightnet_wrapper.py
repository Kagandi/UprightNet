from __future__ import division
import torch
from options.test_options import TestOptions 
from uprightnet.data.data_loader import *
from uprightnet.models.models import create_model


class UprightNetWrapper(object):
    def __init__(self):
        opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
        self.model = create_model(opt, _isTrain=False)
        self.model.switch_to_eval()

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        eval_list_path = '/cluster/home/timoscho/data/test_scannet_normal_list.txt'
        eval_num_threads = 1
        EVAL_BATCH_SIZE = 8
        test_data_loader = CreateScanNetDataLoader(opt, eval_list_path, 
                                                        False, EVAL_BATCH_SIZE, 
                                                        eval_num_threads)
        self.test_dataset = test_data_loader.load_data()
        test_data_size = len(test_data_loader)
        print('========================= ScanNet eval #images = %d ========='%test_data_size)

    def get_gravity_vector(self):

        for i, data in enumerate(self.test_dataset):
            stacked_img = data[0]
            targets = data[1]

            pred_roll, pred_pitch, gt_roll, gt_pitch = self.model.test_roll_pitch(stacked_img, targets)

            print("pred_roll: ", pred_roll*180/3.1416)
            print("pred_pitch: ", pred_pitch*180/3.1416)
            print("gt_roll: ", gt_roll*180/3.1416)
            print("gt_pitch: ", gt_pitch*180/3.1416)
            
            print(stacked_img.size())

            break

# todo: remove
wrapper = UprightNetWrapper()
wrapper.get_gravity_vector()
