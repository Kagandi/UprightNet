# UprightNet
PyTorch implementation of paper "UprightNet: Geometry-Aware Camera Orientation Estimation from Single Images", ICCV 2019
[[Paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xian_UprightNet_Geometry-Aware_Camera_Orientation_Estimation_From_Single_Images_ICCV_2019_paper.pdf) 

## Dependency
The code is tested with Python3, Pytorch >= 1.0 and CUDA >= 10.0, the dependencies includes 
* tensorboardX (only for training, not needed for inference)
* matplotlib
* opencv
* scikit-image
* scipy

## Installation
```console
  git clone https://github.com/Kagandi/UprightNet.git
  cd UprightNet
  pip install . 
```
or
```console
pip install git+https://github.com/Kagandi/UprightNet.git@main
```

## How to create UprightNet gravity estimaton data:

- Download the pretrained weights from [here](https://drive.google.com/file/d/15ZIFwPHP9W50YnsM4JPQGrlcvOeM3fM4/view?usp=sharing): https://drive.google.com/file/d/15ZIFwPHP9W50YnsM4JPQGrlcvOeM3fM4/view?usp=sharing

- Copy the weights to the folder UprightNet/checkpoints/test_local/

If your computer does not have an gpu, check out the branch "no-gpu".


To create the data folders, do the following steps:

1. Download the pretrained weights as described in the section above.

2. Download and extract from https://drive.google.com/drive/folders/1V2KIsXIZ-2-5kGDaErTIpRNnBV2zhVjG?usp=sharing:
	  sample_data.zip (for UprightNet preprocessed ScanNet data)

3. Copy the sample_data folder into the root directry of the UprightNet repo (same level as folder checkpoints)

4. Adapt DATA_PATH in util/config.py to be the absolute path of the directory where test_scannet_normal_list.txt is located

5. To add the predicted gravity vector to the scene folders, run
	python3 test.py --mode ResNet --dataset scannet
	
	Each scene folder in sample_data/data should now contain 4 new folders: pose_pred, pose_gt, gravity_pred, gravity_gt

6. To get the data structure needed for the pose prediction, adapt in 3d-vision/visn/utils/create_data_folder.py 
  - the variable "our_data_path" to be the path where the scene data is located (should be sth like .../sample_data/data)
  - the variable "visn_data_path" to be the folder where the new scene folders should be created
  - Run "python3 create_data_folder.py"

## Dataset
* Download pre-processed InteriorNet and ScanNet, as well as their corresponding training/validation/testing txt files from [link](https://drive.google.com/drive/folders/1WdNAESqDYcUPQyXAW6PvlcdQIYlOEXIw?usp=sharing)
* Modify the paths in train.py, test.py and txt files to match the dataset path in your machine.

## Coordinate system
Our upright and local coordinate systems are defined as follows (corresponding to the normal images in the pre-processed datasets):
* Z upward, Y right, X backward , equivalent to
* Roll negative -> image rotate counterclockwise, Pitch positive -> camera rotate up


## Training

* To train the network on the InteriorNet, run 
```bash
	python3 train.py --mode ResNet --dataset interiornet --w_grad 0.25 --w_pose 2.0
```

* To train the network on the ScanNet, run 
```bash
	python3 train.py --mode ResNet --dataset scannet --w_grad 0.25 --w_pose 0.5
```

## Evaluation: 
* Download checkpoints.zip from [link](https://drive.google.com/drive/folders/1WdNAESqDYcUPQyXAW6PvlcdQIYlOEXIw?usp=sharing), unzip it and make sure checkpoints folder is in the root directory of codebase.

* To evaluate InteriorNet pretrained network on the InteriorNet testset, run
```bash
	python test.py --mode ResNet --dataset interiornet
```

* To evaluate ScanNet pretrained network on the ScanNet testset, run 
```bash
	python test.py --mode ResNet --dataset scannet
```


## Error handling:
* ModuleNotFoundError: No module named 'tensorboardX'
```bash
conda install -c conda-forge tensorboardx
```
or with pip:
```
pip install tensorboardx==2.1
```
