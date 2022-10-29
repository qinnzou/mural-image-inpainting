import os
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import InpaintGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, HistogramLoss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            # if torch.cuda.is_available():
            #     data = torch.load(self.gen_weights_path)
            # else:
            #
            data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)



            import collections  # todo change dataloader
            dicts = collections.OrderedDict()
            for k, value in data['generator'].items():
                if "module" in k:  # 去除命名中的module
                    k = k.split(".")[1:]
                    k = ".".join(k)
                dicts[k] = value

            #如果是多gpu 就要保留module. 单gpu就要去掉
            if(len(self.config.GPU) > 1):
                self.generator.load_state_dict(data['generator'])
            else:
                self.generator.load_state_dict(dicts)

            # self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            if(len(self.config.GPU) > 1):
                self.discriminator.load_state_dict(data['discriminator'])
            else:
                dicts = collections.OrderedDict()
                for k, value in data['discriminator'].items():
                    if "module" in k:  # 去除命名中的module
                        k = k.split(".")[1:]
                        k = ".".join(k)
                    dicts[k] = value
                self.discriminator.load_state_dict(dicts)

    def save(self):
        print('\nsaving %s...\n' % self.name)

        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)
        self.GPU = config.GPU
        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        # refineDiscriminator=Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        histogram_loss=HistogramLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)


        if len(config.GPU) > 1:
            l1_loss = nn.DataParallel(l1_loss, config.GPU)
            perceptual_loss = nn.DataParallel(perceptual_loss , config.GPU)
            style_loss = nn.DataParallel(style_loss, config.GPU)
            histogram_loss = nn.DataParallel(histogram_loss, config.GPU)
            adversarial_loss = nn.DataParallel(adversarial_loss , config.GPU)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
        # self.add_module('refinediscriminator', refineDiscriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('histogram_loss', histogram_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks):
        self.iteration += 1
        coarseonly=False

        if self.iteration<self.config.COARSE_ITERS:
            coarseonly=True

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs1, outputs2= self(images, edges, masks,coarseOnly=coarseonly)
        outputs2_merged = (outputs2 * masks) + (images * (1 - masks))

        msk=masks[:,0,:,:]

        # rate=(msk.shape[1]*msk.shape[2])/torch.sum(msk,dim=(1,2))
        # rate=torch.sum(rate)/masks.shape[1]

        gen_loss = 0
        dis_loss = 0

        if(coarseonly):
            # discriminator loss
            dis_input_real = images
            dis_input_fake = outputs1.detach()
            dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
            dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]

            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)

            dis_loss += (dis_real_loss + dis_fake_loss) / 2

            # generator adversarial loss
            gen_fake, _ = self.discriminator(outputs1)  # in: [rgb(3)]
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_loss += gen_loss

            # generator l1 loss
            gen_l1_loss = self.l1_loss(outputs1, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
            gen_loss += gen_l1_loss

            # generator perceptual loss
            gen_content_loss = self.perceptual_loss(outputs1, images)* self.config.CONTENT_LOSS_WEIGHT
            gen_loss += gen_content_loss

            # generator style loss
            gen_style_loss = self.style_loss(outputs1, images )* self.config.STYLE_LOSS_WEIGHT
            gen_loss += gen_style_loss

            # create logs
            if len(self.GPU) > 1:
                dis_loss = torch.mean(dis_loss)
                gen_gan_loss = torch.mean(gen_gan_loss)
                gen_l1_loss = torch.mean(gen_l1_loss)
                gen_content_loss = torch.mean(gen_content_loss)
                gen_style_loss = torch.mean(gen_style_loss)
                gen_loss = torch.mean(gen_loss)

            logs = [
                ("l_d2", dis_loss.item()),
                ("l_g2", gen_gan_loss.item()),
                ("l_l1", gen_l1_loss.item()),
                ("l_per", gen_content_loss.item()),
                ("l_sty", gen_style_loss.item()),
            ]
            # generator histgramloss
            # gen_hist_loss = self.histogram_loss(outputs1*255, images*255)* self.config.HIST_LOSS_WEIGHT
            # gen_loss += gen_hist_loss
            # gen_hist_loss=0

        else:
            msk = masks[:, 0, :, :]

            # rate=(msk.shape[1]*msk.shape[2])/torch.sum(msk,dim=(1,2))
            # rate=torch.sum(rate)/masks.shape[0]

            # discriminator loss
            dis_input_real = images
            dis_input_fake1 = outputs1.detach()
            dis_input_fake2 = outputs2_merged.detach()
            dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
            dis_fake1, _ = self.discriminator(dis_input_fake1)  # in: [rgb(3)]
            dis_fake2, _ = self.discriminator(dis_input_fake2)  # in: [rgb(3)]

            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss1 = self.adversarial_loss(dis_fake1, False, True)
            dis_fake_loss2 = self.adversarial_loss(dis_fake2, False, True)

            dis_fake_loss = (dis_fake_loss1 + dis_fake_loss2) / 2
            dis_loss += (dis_real_loss + dis_fake_loss) / 2

            # generator adversarial loss
            gen_input_fake1 = outputs1
            gen_input_fake2 = outputs2_merged

            gen_fake1, _ = self.discriminator(gen_input_fake1)  # in: [rgb(3)]
            gen_fake2, _ = self.discriminator(gen_input_fake2)  # in: [rgb(3)]

            gen_gan_loss = self.adversarial_loss(gen_fake1, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_loss += gen_gan_loss

            # generator l1 loss
            gen_l1_loss_1 = self.l1_loss(outputs1, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
            gen_l1_loss_2 = self.l1_loss(outputs2_merged, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)

            # gen_l1_loss_2 = self.l1_loss(outputs2_merged, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)

            gen_l1_loss = (gen_l1_loss_1 + gen_l1_loss_2) / 2
            gen_loss += gen_l1_loss

            # generator perceptual loss
            gen_content_loss_1 = self.perceptual_loss(outputs1, images)
            gen_content_loss_2 = self.perceptual_loss(outputs2_merged, images)
            gen_content_loss = (gen_content_loss_1 + gen_content_loss_2) / 2

            gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            gen_loss += gen_content_loss

            # generator style loss
            gen_style_loss_1 = self.style_loss(outputs1, images)
            gen_style_loss_2 = self.style_loss(outputs2_merged * masks, images * masks)
            gen_style_loss = (gen_style_loss_1 + gen_style_loss_2) / 2
            gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
            gen_loss += gen_style_loss

            # generator histgramloss
            gen_hist_loss_1 = self.histogram_loss(outputs1*255, images*255)
            gen_hist_loss_2 = self.histogram_loss(outputs2 * masks*255, images * masks*255)
            gen_hist_loss = (gen_hist_loss_1 + gen_hist_loss_2) / 2
            gen_hist_loss = gen_hist_loss * self.config.HIST_LOSS_WEIGHT
            gen_loss += gen_hist_loss

            # create logs
            if len(self.GPU) > 1:
                dis_loss = torch.mean(dis_loss)
                gen_gan_loss = torch.mean(gen_gan_loss)
                gen_l1_loss = torch.mean(gen_l1_loss)
                gen_content_loss = torch.mean(gen_content_loss)
                gen_style_loss = torch.mean(gen_style_loss)
                gen_hist_loss = torch.mean(gen_hist_loss)
                gen_loss = torch.mean(gen_loss)

            logs = [
                ("l_d2", dis_loss.item()),
                ("l_g2", gen_gan_loss.item()),
                ("l_l1", gen_l1_loss.item()),
                ("l_per", gen_content_loss.item()),
                ("l_sty", gen_style_loss.item()),
                ("l_hist", gen_hist_loss.item()),
            ]


        return outputs2, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks,returnInput=False,coarseOnly=False):#孔洞是1
        images_masked = (images * (1 - masks).float())
        inputs = torch.cat((images_masked, edges), dim=1)
        if returnInput:
            outputs1,outputs2,inputs2 = self.generator(inputs,masks,returnInput2=returnInput,coarseOnly=coarseOnly)                                 # in: [rgb(3) + edge(1)]
            return outputs1, outputs2, inputs2
        else:
            outputs1,outputs2 = self.generator(inputs,masks,returnInput2=returnInput,coarseOnly=coarseOnly)                                 # in: [rgb(3) + edge(1)]
            return outputs1,outputs2

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        gen_loss.backward()
        self.dis_optimizer.step()
        self.gen_optimizer.step()



