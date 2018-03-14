import os
from glob import glob
from tqdm import tqdm

for _, _, im in os.walk('/home/abc/pzw/data/class_19/images'):
    img_names = im
c = 0
img_dir = sorted(glob('data/data0211/*'))
for i in tqdm(img_dir):
    if i.split('/')[-1] in img_names:
        c += 1
print c
