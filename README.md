# UprightNet
PyTorch implementation of paper "UprightNet: Geometry-Aware Camera Orientation Estimation from Single Images", ICCV 2019
[[Paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xian_UprightNet_Geometry-Aware_Camera_Orientation_Estimation_From_Single_Images_ICCV_2019_paper.pdf) 

## Dependency
The code is tested with Python3, Pytorch >= 1.0 and CUDA >= 10.0, the dependencies includes 
* tensorboardX
* matplotlib
* opencv
* scikit-image
* scipy

## How to create UprightNet gravity estimaton data:

1. Download and extract from https://drive.google.com/drive/folders/1WdNAESqDYcUPQyXAW6PvlcdQIYlOEXIw:
	a) ScanNet.zip (for UprightNet preprocessed ScanNet data)
	b) checkpoints.zip (pretrained weights)
	c) test_scannet_normal_list.txt (List with ScanNet test data)
	
2. Adapt the paths of test_scannet_normal_list.txt to the path of ScanNet data (e.g. with find&replace)

3. Copy checkpoints folder to root directory of UprightNet repo

4. Adapt DATA_PATH in util/config.py to be the directory where test_scannet_normal_list.txt is located

5. To add the predicted gravity vector to the data folders, run
	python3 test.py --mode ResNet --dataset scannet
	
	Each scene folder should now contain 4 new folders: pose_pred, pose_gt, gravity_pred, gravity_gt

	If your computer does not have an gpu, check out the branch "no-gpu".


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
