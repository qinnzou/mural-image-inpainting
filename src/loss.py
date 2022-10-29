import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import torch.nn.functional as F

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
            # self.criterion = nn.CrossEntropyLoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels =  (self.real_label if is_real else self.fake_label).expand_as(outputs) #todo
            # print('labels:')
            # print(labels.data)
            # print('outputs: ')
            # print(outputs.data)
            # outputs[outputs < 0.0] = 0.0
            # outputs[outputs > 1.0] = 1.0
            # outputs=outputs.long()
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss

class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out

class HistogramLoss(nn.Module):
    def __init__(self):
        super(HistogramLoss, self).__init__()
        self.add_module('vgg', VGG19())

    def hist_match(self,source,template):
        shape = source.shape
        source = torch.flatten(source,1,3)
        template = torch.flatten(template,1,3)
        hist_bins = 255

        max_value=torch.max(torch.cat((torch.max(source, 1).values,torch.max(template,1).values),dim=0),0)
        min_value=torch.min(torch.cat((torch.min(source, 1).values,torch.min(template,1).values),dim=0),0)

        hist_delta = (max_value.values - min_value.values) / hist_bins
        hist_range = torch.range(min_value.values.squeeze(0).data, max_value.values.squeeze(0).data, step=hist_delta.data).to("cuda")
        hist_range = torch.add(hist_range, torch.div(hist_delta, 2))
        s_hist=torch.histc(source,hist_bins,min_value.values.squeeze(0).data,max_value.values.squeeze(0).data)
        t_hist=torch.histc(template,hist_bins,min_value.values.squeeze(0).data,max_value.values.squeeze(0).data)

        s_quantiles = torch.cumsum(s_hist,0)
        s_last_element = torch.sub(len(s_quantiles), 1).to("cuda")
        s_quantiles = torch.div(s_quantiles, s_quantiles.gather(0, s_last_element))#维度待定

        t_quantiles = torch.cumsum(t_hist,0)
        t_last_element = torch.sub(len(t_quantiles), 1).to("cuda")
        t_quantiles = torch.div(t_quantiles, t_quantiles.gather(0, t_last_element))


        nearest_indices = torch.tensor(list(map(lambda x: torch.argmin(torch.abs(torch.sub(t_quantiles, x))),
                                    s_quantiles)),dtype=torch.int64,device="cuda")

        s_bin_index = (torch.div(source.squeeze(0), hist_delta.squeeze(0)))

        s_bin_index = torch.clamp(s_bin_index, 0, 254).long()
        matched_to_t = hist_range.gather(0, nearest_indices.gather(0, s_bin_index)).int()
        return torch.reshape(matched_to_t, shape)

    def __call__(self, source, target):
        # vgg=self.vgg.cuda()
        x_vgg, y_vgg = self.vgg(source)['relu4_1'], self.vgg(target)['relu4_1']
        loss1=0
        loss2=0

        for i in range(source.shape[0]):
            histogram1 = self.hist_match(source[i].unsqueeze(0), target[i].unsqueeze(0))
            d1=source[i].unsqueeze(0) - histogram1
            loss1 += torch.sum(torch.mul(d1, d1).float().mean())

            histogram2=self.hist_match(x_vgg[i].unsqueeze(0), y_vgg[i].unsqueeze(0))
            d2=x_vgg[i].unsqueeze(0) - histogram2
            loss2+=torch.sum(torch.mul(d2, d2).float().mean())


        # histogram = self.hist_match(x_vgg['relu1_1'], y_vgg['relu1_1'])
        # loss1 = torch.sum(torch.mul(x_vgg['relu1_1']-histogram,x_vgg['relu1_1']-histogram).float().mean())
        #
        # histogram = self.hist_match(x_vgg['relu4_1'], y_vgg['relu4_1'])
        # loss2 = torch.sum(torch.mul(x_vgg['relu4_1']-histogram,x_vgg['relu4_1']-histogram).float().mean())
        #
        loss=(loss2/source.shape[0]+loss1/source.shape[0])/2
        return loss
