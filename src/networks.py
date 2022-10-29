import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class PSA_s(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_s, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 4

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        # self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        # kaiming_init(self.conv_q_right, mode='fan_in')
        # kaiming_init(self.conv_v_right, mode='fan_in')
        # kaiming_init(self.conv_q_left, mode='fan_in')
        # kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))

        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        out = self.spatial_pool(x)

        # [N, C, H, W]
        out = self.channel_pool(out)

        # [N, C, H, W]
        # out = context_spatial + context_channel

        return out


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out, attention
        else:
            return out

# class InpaintFineGenerator(BaseNetwork):
#     def __init__(self, residual_blocks=8, init_weights=True):
#         super(InpaintFineGenerator, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_channels=5, out_channels=64, kernel_size=7, padding=0),
#             nn.InstanceNorm2d(64, track_running_stats=False),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(128, track_running_stats=False),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(256, track_running_stats=False),
#             nn.ReLU()
#         )
#
#         # self.refine_attn = Self_Attn(256, 'relu', with_attn=False)
#
#         blocks = []
#         for _ in range(residual_blocks):
#             block = ResnetBlock(256, 2)
#             blocks.append(block)
#         self.middle = nn.Sequential(
#             *blocks,
#         )
#
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(128, track_running_stats=False),
#             nn.ReLU(),
#
#             nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(64, track_running_stats=False),
#             nn.ReLU(),
#
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
#         )
#
#         if init_weights:
#             self.init_weights()
#
#     def forward(self, x):
#         x = self.encoder(x)
#         # x=self.refine_attn(x)
#         x = self.middle(x)
#         x = self.decoder(x)
#         x = (torch.tanh(x) + 1) / 2
#         mask_view1 = x[0, 0].cpu().detach().numpy()#
#
#         return x
#

class InpaintCoarseNet(nn.Module):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintCoarseNet, self).__init__()

        self.res_blocks = residual_blocks

        self.pad1 = nn.ReflectionPad2d(3)
        self.pConv1_1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=7, padding=0)
        self.pnorm1_1 = nn.InstanceNorm2d(32, track_running_stats=False)
        self.pact1_1 = nn.ReLU()

        self.pConv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.pnorm2_1 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.pact2_1 = nn.ReLU()

        self.pConv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.pnorm3_1 = nn.InstanceNorm2d(128, track_running_stats=False)
        self.pact3_1 = nn.ReLU()

        self.pConv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.pnorm4_1 = nn.InstanceNorm2d(256, track_running_stats=False)
        self.pact4_1 = nn.ReLU()

        self.block1 = ResnetBlock(256, 2)
        self.block2 = ResnetBlock(256, 2)
        self.block3 = ResnetBlock(256, 2)
        self.block4 = ResnetBlock(256, 2)

        self.conv0_5 = nn.Conv2d(in_channels=256 , out_channels=128, kernel_size=3, padding=1)  # +256
        self.norm0_5 = nn.InstanceNorm2d(128, track_running_stats=False)
        self.act0_5 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(in_channels=128+128 , out_channels=64, kernel_size=3, padding=1)  # +256
        self.norm1_1 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.act1_1 = nn.ReLU()

        self.conv2_1 = nn.Conv2d(in_channels=64+64, out_channels=32, kernel_size=3, padding=1)  # +256
        self.norm2_1 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.act2_1 = nn.ReLU()

        self.conv3_1 = nn.Conv2d(in_channels=32+32, out_channels=16, kernel_size=3, padding=1)  # +128
        self.norm3_1 = nn.InstanceNorm2d(32, track_running_stats=False)
        self.act3_1 = nn.ReLU()

        self.pad2 = nn.ReflectionPad2d(3)
        self.conv3_2 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=7, padding=0)


    def forward(self, x):
        x = self.pad1(x)
        x = self.pConv1_1(x)
        x = self.pnorm1_1(x)
        x = self.act1_1(x)
        x1 = x

        x = self.pConv2_1(x)
        x = self.pnorm2_1(x)
        x = self.pact2_1(x)
        x2 = x

        x = self.pConv3_1(x)
        x = self.pnorm3_1(x)
        x = self.pact3_1(x)
        x3 = x

        x = self.pConv4_1(x)
        x = self.pnorm4_1(x)
        x = self.pact4_1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.conv0_5(x)
        x = self.norm0_5(x)
        x = self.act0_5(x)

        x = F.interpolate(x, size=[x.shape[2] * 2, x.shape[3] * 2], mode='bilinear', align_corners=True)
        x = torch.cat((x, x3), dim=1)

        x = self.conv1_1(x)
        x = self.norm1_1(x)
        x = self.act1_1(x)

        x = F.interpolate(x, size=[x.shape[2] * 2, x.shape[3] * 2], mode='bilinear', align_corners=True)
        x = torch.cat((x, x2), dim=1)

        x = self.conv2_1(x)
        x = self.norm2_1(x)
        x = self.act2_1(x)

        x = F.interpolate(x, size=[x.shape[2] * 2, x.shape[3] * 2], mode='bilinear', align_corners=True)
        x = torch.cat((x, x1), dim=1)

        x = self.conv3_1(x)
        x = self.norm3_1(x)
        x = self.act3_1(x)

        # x = F.interpolate(x, size=[x.shape[2] * 2, x.shape[3] * 2], mode='bilinear', align_corners=True)

        x = self.pad2(x)
        x = self.conv3_2(x)

        x = (torch.tanh(x) + 1) / 2
        x=torch.clamp(x,0,1)

        return x

class InpaintRefineNet(nn.Module):
    def __init__(self, residual_blocks=4, init_weights=True):
        super(InpaintRefineNet, self).__init__()

        self.res_blocks = residual_blocks

        self.pad1 = nn.ReflectionPad2d(3)
        self.pConv1_1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=7, padding=0)
        self.pnorm1_1 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.pact1_1 = nn.ReLU()

        self.pConv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.pnorm2_1 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.pact2_1 = nn.ReLU()

        self.pConv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.pnorm3_1 = nn.InstanceNorm2d(128, track_running_stats=False)
        self.pact3_1 = nn.ReLU()

        self.pConv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.pnorm4_1 = nn.InstanceNorm2d(256, track_running_stats=False)
        self.pact4_1 = nn.ReLU()

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.refine_attn = Self_Attn(256, 'relu', with_attn=False)
        # self.refine_attn = PSA_s(256,256)

        self.middle = nn.Sequential(*blocks)

        self.conv0_5 = nn.Conv2d(in_channels=256 , out_channels=128, kernel_size=3, padding=1)  # +256
        self.norm0_5 = nn.InstanceNorm2d(128, track_running_stats=False)
        self.act0_5 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(in_channels=128+128 , out_channels=64, kernel_size=3, padding=1)  # +256
        self.norm1_1 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.act1_1 = nn.ReLU()

        self.conv2_1 = nn.Conv2d(in_channels=64+64 , out_channels=32, kernel_size=3, padding=1)  # +256
        self.norm2_1 = nn.InstanceNorm2d(32, track_running_stats=False)
        self.act2_1 = nn.ReLU()

        self.conv3_1 = nn.Conv2d(in_channels=32+32 , out_channels=16, kernel_size=3, padding=1)  # +128
        self.norm3_1 = nn.InstanceNorm2d(16, track_running_stats=False)
        self.act3_1 = nn.ReLU()

        self.pad2 = nn.ReflectionPad2d(3)
        self.conv3_2 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=7)

    def forward(self, x):
        x0=x[:,:-1,:,:]

        x = self.pad1(x)
        x = self.pConv1_1(x)
        x = self.pnorm1_1(x)
        x = self.act1_1(x)
        x1 = x

        x = self.pConv2_1(x)
        x = self.pnorm2_1(x)
        x = self.pact2_1(x)
        x2 = x

        x = self.pConv3_1(x)
        x = self.pnorm3_1(x)
        x = self.pact3_1(x)
        x3 = x

        x = self.pConv4_1(x)
        x = self.pnorm4_1(x)
        x = self.pact4_1(x)

        x=self.refine_attn(x)
        x = self.middle(x)

        x = self.conv0_5(x)
        x = self.norm0_5(x)
        x = self.act0_5(x)

        x = F.interpolate(x, size=[x.shape[2] * 2, x.shape[3] * 2], mode='bilinear', align_corners=True)
        x = torch.cat((x, x3), dim=1)

        x = self.conv1_1(x)
        x = self.norm1_1(x)
        x = self.act1_1(x)

        x = F.interpolate(x, size=[x.shape[2] * 2, x.shape[3] * 2], mode='bilinear', align_corners=True)
        x = torch.cat((x, x2), dim=1)

        x = self.conv2_1(x)
        x = self.norm2_1(x)
        x = self.act2_1(x)

        x = F.interpolate(x, size=[x.shape[2] * 2, x.shape[3] * 2], mode='bilinear', align_corners=True)
        x = torch.cat((x, x1), dim=1)

        x = self.conv3_1(x)
        x = self.norm3_1(x)
        x = self.act3_1(x)

        x = self.pad2(x)
        x = self.conv3_2(x)
        x = (torch.tanh(x) + 1) / 2

        x=torch.clamp(x,0,1)
        x = x + x0
        x=torch.clamp(x,0,1)
        return x

class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=4, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.coarsenet=InpaintCoarseNet(residual_blocks=residual_blocks,init_weights=True)
        self.refinenet=InpaintRefineNet(residual_blocks=residual_blocks,init_weights=True)

        if init_weights:
            self.init_weights()

    def forward(self, x,masks,returnInput2=False,coarseOnly=False):
        x0=x[:,:-1,:,:]

        # x = F.interpolate(x, size=[(int)(x.shape[2] / 2), (int)(x.shape[3] / 2)], mode='bilinear', align_corners=True)

        x1 = self.coarsenet(x)
        newinput_merged = (x1 * masks) + (x0 * (1 - masks))

        channel1mask=masks[:,0,:,:]
        channel1mask=torch.unsqueeze(channel1mask,1)
        newinput_merged_withmask=torch.cat((newinput_merged, channel1mask), dim=1)

        if(coarseOnly):
            x2=x1
            # with torch.no_grad():
            #     x2 = self.refinenet(newinput_merged_withmask)
        else:
            x2=self.refinenet(newinput_merged_withmask)

        if returnInput2:
            return x1,x2,newinput_merged
        else:
            return x1,x2

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
