import os
from config import DATA_PATH

scenes = {'train': ['scene0000_00'],
          'validation': ['scene0357_00'],
          'test': []}

for stage in scenes:
    with open(os.path.join(DATA_PATH, f'{stage}_scannet_normal_list.txt'), 'w') as f:
        for scene in scenes[stage]:
            normal_pair_path = os.path.join(DATA_PATH, scene, 'normal_pair')
            for file in os.listdir(normal_pair_path):
                path = os.path.join(normal_pair_path, file)
                f.write(path)
                f.write('\n')
