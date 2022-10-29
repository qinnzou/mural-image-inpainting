import os
import cv2
from skimage.measure import compare_ssim
import torch
import numpy as np
import lpips
from torch.autograd import Variable

class calpackage(object):
    def __init__(self,mode="all"):
        self.mode = mode
    def call(self, img1, img2):
        #lpips part
        loss_fn_alex = lpips.LPIPS(net='alex',verbose=False) # best forward scores
        if torch.cuda.is_available():
            loss_fn_alex = loss_fn_alex.cuda()
            img1 = img1.cuda()
            img2 = img2.cuda()
        lpips_value_alex = loss_fn_alex(img1, img2,normalize=True)

        return lpips_value_alex


def get_metics(img,referimg,mask=None,size=512):

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    referimg = cv2.resize(referimg, (size, size), interpolation=cv2.INTER_AREA)


    (score, diff) = compare_ssim(referimg, img, win_size=21, full=True, multichannel=True)
    ssim = score

    img = np.array(img, dtype=np.float64).transpose((2, 0, 1))
    referimg = np.array(referimg, dtype=np.float64).transpose((2, 0, 1))

    img = torch.from_numpy(img)
    referimg = torch.from_numpy(referimg)

    img1 = Variable(img/ 255).unsqueeze(0)  # image should be RGB, IMPORTANT: normalized to [-1,1]
    img2 = Variable(referimg/ 255).unsqueeze(0)  # image should be RGB, IMPORTANT: normalized to [-1,1]
    caltool = calpackage()
    LPIPS_value_alexnetbase = caltool.call(img1.type(torch.FloatTensor),img2.type(torch.FloatTensor))
    LPIPS_value_alex = LPIPS_value_alexnetbase.detach().cpu().numpy().mean()

    loss2 = torch.mean((img / 255. - referimg / 255.) ** (2))

    out_dict = {}
    if mask is not None:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
        proportion = np.array(mask, dtype=np.float64).mean() / 255
        ssim = ssim - (1-ssim)*(1-proportion)/proportion
        loss2 = loss2/proportion
        LPIPS_value_alex = LPIPS_value_alex/proportion
        psnr = 10 * torch.log(1 / loss2) / torch.log(torch.tensor(10.0))
    else:
        psnr = 10 * torch.log(1 / loss2) / torch.log(torch.tensor(10.0))

    out_dict["ssim"] = ssim
    out_dict["l2"] = loss2
    out_dict["psnr"] = psnr
    out_dict["LPIPS_value_alex"] = LPIPS_value_alex
    return out_dict

# test_methods = ["ours","deepfill","edgeconnect","RFR"]
test_methods = ["ours"]
test_root = '/public/home/lucy/mural/quantitative/'
referpath = "/public/home/lucy/mural/quantitative/eval/input"
referfiles = os.listdir(referpath)
maskpath = "/public/home/lucy/mural/quantitative/eval/input"
maskfiles = os.listdir(maskpath)

out_frame = {"ssim":0,"l2":0,"psnr":0,"LPIPS_value_alex":0}
pred_pathlists= {}
out_dicts={}
run_dicts={}
for method in test_methods:
    path=test_root+method
    pred_pathlists[method]=path
    out_dicts[method]=out_frame.copy()
    run_dicts[method]={}

show = False

for i, filename in enumerate(referfiles):
    referimg=cv2.imread(os.path.join(referpath,filename))
    mask=cv2.imread(os.path.join(maskpath,filename))
    for method in test_methods:
        img = cv2.imread(os.path.join(pred_pathlists[method], filename))
        values=get_metics(img,referimg,mask,size=256)
        for key in values.keys():
            out_dicts[method][key]+=values[key]
            run_dicts[method][key]=values[key]

    if show:
        for method in test_methods:
            print("iter:{:-3} method:{:12} ".format(i+1,method),end="")
            for key in run_dicts[method].keys():
                print(key + ":{:.4f}  ".format(run_dicts[method][key]),end="")
            print(" \t")

    print(" \t")


for method in test_methods:

    ssim=out_dicts[method]["ssim"]/(len(referfiles))
    l2=(out_dicts[method]["l2"]/(len(referfiles)))
    psnr=(out_dicts[method]["psnr"]/(len(referfiles)))
    LPIPS_value_alex=out_dicts[method]["LPIPS_value_alex"]/len(referfiles)

    print("total:{} method:{:12} l2_loss:{:.4f} psnr:{:.4f} ssim:{:.4f} LPIPS_alex:{:.4f}".format(len(referfiles), method, l2, psnr, ssim, LPIPS_value_alex))


