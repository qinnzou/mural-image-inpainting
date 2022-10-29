# Line Drawing Guided Progressive Inpainting of Mural Damages - MuralNet

This is the source code of 'Line Drawing Guided Progressive Inpainting of Mural Damages'. We provide the codes, the partitioning of the datasets, and the pretrained models on MuralDataset. For convienence of introduction, the proposed network is named as MuralNet.

-Mural image inpainting, -Mural damage repair, -Line drawing guided image inpainting


# Set up
##Requirments
- Python 3.7
- PyTorch 1.6
- We run the codes on NVIDIA GPU RTX2080ti wiht CUDA 10.2 and cuDNN 7.6.5

## Preparation
### Data Preparation
We collected a mural dataset DhMural1714 to train and test the MuralNet. You can download them and put them into "./dataset/".
### Pretrained Models
Our model is pretrained on DhMural1714, the pretrained model can be obtained at `./checkpoints/InpaintingModel_gen.pth` and `./checkpoints/InpaintingModel_dis.pth`


## Download:
### DhMural1714 Dataset

CRKWH100 dataset: https://1drv.ms 
CRKWH100 GT: https://1drv.ms

You can also download the dataset from  
Link: https://pan.baidu.com/s/13dkh4zX3C4_z2W3Kigg6uQ 
passcodes: 0vlr


### Pretrained Models
You can also download the pretrained models from  
link: https://pan.baidu.com/s/1EKvTAHyOaMbL9s7aqdZEfw 
passcodes: vg5v


## Training

MuralNet is trained on two stages: 1) training the coarse network, 2) training the whole model. 

To train the model, modify the training parameters in `checkpoints/config.yml`, you can refer to [EdgeConnect](https://github.com/knazeri/edge-connect) for details.

Run the code for training:
```bash
python train.py
```

## Test
We provide several example images in `checkpoints/test` for testing, just run the code:
```bash
python test.py
```
Directory of testing images can be modified in `main.py`, the network requires the input images, the corresponding line drawings, and the masks for inpainting.

## Evaluate
To evaluate the performance, run the code:
```bash
python eval_mix.py
```

## Acknowledgment
Our implementation is mainly based on [EdgeConnect](https://github.com/knazeri/edge-connect). Thanks for the authors of the EdgeConnect paper.
```bash
@inproceedings{nazeri2019edgeconnect,
  title={EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning},
  author={Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  journal={arXiv preprint},
  year={2019},
}
```
