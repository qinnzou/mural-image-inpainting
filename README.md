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
We built a mural dataset for training and testing MuralNet. You can download them and put them into "./dataset/".
### Pretrained Models
Our model is pretrained on MuralDataset, the pretrained model can be obtained at `./checkpoints/InpaintingModel_gen.pth` and `./checkpoints/InpaintingModel_dis.pth`
## Training

MuralNet is trained in two stages: 1) training the coarse network, 2) training the whole model. 

To train the model, modify the training parameters in `checkpoints/config.yml`, you can refer to [edge-connect](https://github.com/knazeri/edge-connect) for specific format.

Run the code for training:
```bash
python train.py
```

## Test
We provide several example images in `checkpoints/test` for testing, just run the code:
```bash
python test.py
```
Directory of testing images can be modified in `main.py`, the network requires images, corresponding line drawings and mask for inpainting.

## Evaluate
To evaluate the performance, run the code:
```bash
python metrics/eval_mix.py
```

## Acknowledgment
Our implementation is mainly based on [edge-connect](https://github.com/knazeri/edge-connect). Thanks for the authors.
```bash
@inproceedings{nazeri2019edgeconnect,
  title={EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning},
  author={Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  journal={arXiv preprint},
  year={2019},
}
```
