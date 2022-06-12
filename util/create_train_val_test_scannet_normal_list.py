import os
from config import DATA_PATH

scenes = {'train': ["scene0144_01", "scene0169_01", "scene0207_00", "scene0221_01", 
                    "scene0328_00", "scene0334_02", "scene0353_02", "scene0357_00",
                    "scene0426_00", "scene0435_03", "scene0490_00", "scene0527_00",
                    "scene0574_01", "scene0578_00", "scene0583_02", "scene0598_01",
                    "scene0616_01", "scene0629_02", "scene0645_00", "scene0647_00",
                    "scene0658_00", "scene0663_02", "scene0664_02", "scene0671_00",
                    "scene0686_02", "scene0695_00", "scene0696_00", "scene0697_01" ],
          'validation': ["scene0251_00", "scene0378_01", "scene0553_00", "scene0599_02",
                         "scene0651_01", "scene0678_00", "scene0701_00" ],
          'test': ["scene0278_01", "scene0406_02", "scene0565_00", "scene0608_02",
                   "scene0653_01", "scene0685_01", "scene0704_01" ]}

for stage in scenes:
    with open(os.path.join(DATA_PATH, f'{stage}_scannet_normal_list.txt'), 'w+') as f:
        for scene in scenes[stage]:
            normal_pair_path = os.path.join(DATA_PATH, scene, 'normal_pair')
            for file in os.listdir(normal_pair_path):
                path = os.path.join(normal_pair_path, file)
                f.write(path)
                f.write('\n')
