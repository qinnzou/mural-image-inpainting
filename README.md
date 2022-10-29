# Line Drawing Guided Progressive Inpainting of Mural Damages - MuralNet

This is the source code for 'Line Drawing Guided Progressive Inpainting of Mural Damages'. We provide the code, the dataset, and the pretrained models. For convienence of introduction, we name the proposed network as MuralNet.

## Abstract:
Mural image inpainting refers to repairing the damage or missing areas in a mural image to restore the visual appearance. Most existing image-inpainting methods tend to take a target image as the only input and directly repair the damage to generate a visually plausible result. These methods obtain high performance in restoration or completion of some specific objects, e.g., human face, fabric texture, and printed texts, etc., however, are not suitable for repairing murals with varied subjects, especially for murals with large damaged areas. Moreover, due to the discrete colors in paints, mural inpainting may suffer from apparent color bias as compared to natural image inpainting. To this end, in this paper, we propose a line drawing guided progressive mural inpainting method. It divides the inpainting process into two steps: structure reconstruction and color correction, executed by a structure reconstruction network (SRN) and a color correction network (CCN), respectively. In the structure reconstruction, line drawings are used by SRN as a guarantee for large-scale content authenticity and structural stability. In the color correction, CCN operates a local color adjustment for missing pixels which reduces the negative effects of color bias and edge jumping. The proposed approach is evaluated against the current state-of-the-art image inpainting methods. Qualitative and quantitative results demonstrate the superiority of the proposed method in mural image inpainting. 

-Mural image inpainting, -Mural damage repair, -Line drawing guided image inpainting

## Mural Inpainting with and without line drawings
![image](https://github.com/qinnzou/mural-image-inpainting/blob/main/other/intro2.jpg)

## Network Architecture
![image](https://github.com/qinnzou/mural-image-inpainting/blob/main/other/net.jpg)


## Set up:
### Requirments
- Python 3.7
- PyTorch 1.6
- We run the codes on NVIDIA GPU RTX2080ti wiht CUDA 10.2 and cuDNN 7.6.5

## Preparation:
### Data Preparation
We collected a mural dataset DhMural1714 to train and test the MuralNet. You can download them and put them into "./dataset/".
### Pretrained Models
Our model is pretrained on DhMural1714, the pretrained model can be obtained at `./checkpoints/InpaintingModel_gen.pth` and `./checkpoints/InpaintingModel_dis.pth`


## Download:
### DhMural1714 Dataset

dataset: https://1drv.ms 

You can also download the dataset from  
Link: https://pan.baidu.com/s/13dkh4zX3C4_z2W3Kigg6uQ 
passcodes: 0vlr


### Pretrained Models
You can also download the pretrained models from  
link: https://pan.baidu.com/s/1EKvTAHyOaMbL9s7aqdZEfw 
passcodes: vg5v


## Training:

MuralNet is trained on two stages: 1) training the coarse network, 2) training the whole model. 

To train the model, modify the training parameters in `checkpoints/config.yml`.

Run the code for training:
```bash
python train.py
```

## Test:
We provide several example images in `checkpoints/test` for testing, just run the code:
```bash
python test.py
```
Directory of testing images can be modified in `main.py`, the network requires the input images, the corresponding line drawings, and the masks for inpainting.

## Evaluation:
To evaluate the performance, run the code:
```bash
python eval_mix.py
```

## Acknowledgment:
Our implementation is mainly based on [EdgeConnect](https://github.com/knazeri/edge-connect). We thank the authors of the EdgeConnect paper.
```bash
@inproceedings{nazeri2019edgeconnect,
  title={EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning},
  author={Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  journal={arXiv preprint},
  year={2019},
}
```
