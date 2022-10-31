# this is modified based on the code released by https://github.com/knazeri/edge-connect
# Nazeri, K., Ng, E., Joseph, T., Qureshi, F. Z., & Ebrahimi, M. (2019). Edgeconnect: Generative image inpainting with adversarial edge learning. arXiv preprint arXiv:1901.00212.


import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset')
parser.add_argument('--output', type=str, help='path to the file list')
args = parser.parse_args()

imgdir='F:/dunhuang-paintings/917dataset/val/images'
savedir='F:/dunhuang-paintings/flist/val/images.flist'

ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}

images = []
for root, dirs, files in os.walk(imgdir):#args.path args.output
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1].upper() in ext:
            images.append(os.path.join(root, file))

images = sorted(images)
np.savetxt(savedir, images, fmt='%s')
