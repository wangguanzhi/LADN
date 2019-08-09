import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler


####################################################################
#------------------------- Discriminators --------------------------
####################################################################

class Dis_pair(nn.Module):
    def __init__(self, input_dim_a=3, input_dim_b=3, dis_n_layer=5, norm='None', sn=True):
        super(Dis_pair, self).__init__()
        ch = 64
        self.model = self._make_net(ch, input_dim_a+input_dim_b, dis_n_layer, norm, sn)

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
        tch = ch
        for i in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
            tch *= 2
        model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)]
        tch *= 2
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]
        else:
            model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]
        return nn.Sequential(*model)

    def forward(self, image_a, image_b):
        out = torch.cat((image_a, image_b), 1)
        out = self.model(out)
        out = out.view(-1)
        outs = []
        outs.append(out)
        return outs

class Dis_content(nn.Module):
    def __init__(self, ndf=256):
        super(Dis_content, self).__init__()
        model = []
        model += [LeakyReLUConv2d(ndf, ndf, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(ndf, ndf, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(ndf, ndf, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(ndf, ndf, kernel_size=4, stride=1, padding=0)]
        model += [nn.Conv2d(ndf, 1, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1)
        outs = []
        outs.append(out)
        return outs


class MultiScaleDis(nn.Module):
    def __init__(self, input_dim, n_scale=3, n_layer=4, norm='None', sn=False):
        super(MultiScaleDis, self).__init__()
        ch = 64
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.Diss = nn.ModuleList()
        for _ in range(n_scale):
            self.Diss.append(self._make_net(ch, input_dim, n_layer, norm, sn))

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, 4, 2, 1, norm, sn)]
        tch = ch
        for _ in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, 4, 2, 1, norm, sn)]
            tch *= 2
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(tch, 1, 1, 1, 0))]
        else:
            model += [nn.Conv2d(tch, 1, 1, 1, 0)]
        return nn.Sequential(*model)

    def forward(self, x):
        outs = []
        for Dis in self.Diss:
            outs.append(Dis(x))
            x = self.downsample(x)
        return outs


class Dis(nn.Module):
    def __init__(self, input_dim, norm='None', sn=False):
        super(Dis, self).__init__()
        ch = 64
        n_layer = 5
        self.model = self._make_net(ch, input_dim, n_layer, norm, sn)

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
        tch = ch
        for i in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
            tch *= 2
        model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)]
        tch *= 2
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]
        else:
            model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]
        return nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1)
        outs = []
        outs.append(out)
        return outs

####################################################################
#---------------------------- Encoders -----------------------------
####################################################################

class E_content(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, ngf = 64):
        super(E_content, self).__init__()
        self.conv1_e1_A = nn.Conv2d(in_channels=input_dim_a, out_channels=ngf, kernel_size=5, stride=2, padding=2, bias=False)
        self.relu_e1_A = nn.LeakyReLU(0.2, True)
        self.conv2_e1_A = nn.Conv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn_e1_A = nn.BatchNorm2d(ngf*2)

        self.conv1_e1_B = nn.Conv2d(in_channels=input_dim_b, out_channels=ngf, kernel_size=5, stride=2, padding=2, bias=False)
        self.relu_e1_B = nn.LeakyReLU(0.2, True)
        self.conv2_e1_B = nn.Conv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn_e1_B = nn.BatchNorm2d(ngf*2)


    def forward(self, xa, xb):
        outputA = self.forward_a(xa)
        outputB = self.forward_b(xb)
        return outputA, outputB

    def forward_a(self, xa):
        # xa (3, 216, 216)
        e1_1_A = self.conv1_e1_A(xa)
        # e1_1_A (ngf, 108, 108)
        e1_2_A = self.bn_e1_A(self.conv2_e1_A(self.relu_e1_A(e1_1_A)))
        # e1_2_A (ngf*2, 54, 54)
        return e1_1_A, e1_2_A

    def forward_b(self, xb):
        # xb (3, 216, 216)
        e1_1_B = self.conv1_e1_B(xb)
        # e1_1_B (ngf, 108, 108)
        e1_2_B = self.bn_e1_B(self.conv2_e1_B(self.relu_e1_B(e1_1_B)))
        # e1_2_B (ngf*2, 54, 54)
        return e1_1_B, e1_2_B

class E_attr(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, ngf=64):
        super(E_attr, self).__init__()

        self.conv1_e2_A = nn.Conv2d(in_channels=input_dim_a, out_channels=ngf, kernel_size=5, stride=2, padding=2, bias=False)
        self.relu_e2_A = nn.LeakyReLU(0.2, True)
        self.conv2_e2_A = nn.Conv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn_e2_A = nn.BatchNorm2d(ngf*2)

        self.conv1_e2_B = nn.Conv2d(in_channels=input_dim_b, out_channels=ngf, kernel_size=5, stride=2, padding=2, bias=False)
        self.relu_e2_B = nn.LeakyReLU(0.2, True)
        self.conv2_e2_B = nn.Conv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn_e2_B = nn.BatchNorm2d(ngf*2)


    def forward(self, xa, xb):
        outputA = self.forward_a(xa)
        outputB = self.forward_b(xb)
        return outputA, outputB

    def forward_a(self, xa):
        # xa (3, 216, 216)
        e2_1_A = self.conv1_e2_A(xa)
        # e1_1_A (ngf, 108, 108)
        e2_2_A = self.bn_e2_A(self.conv2_e2_A(self.relu_e2_A(e2_1_A)))
        # e1_2_A (ngf*2, 54, 54)

        return e2_2_A

    def forward_b(self, xb):
        # xb (3, 216, 216)
        e2_1_B = self.conv1_e2_B(xb)
        # e1_1_B (ngf, 108, 108)
        e2_2_B = self.bn_e2_B(self.conv2_e2_B(self.relu_e2_B(e2_1_B)))
        # e1_2_B (ngf*2, 54, 54)

        return e2_2_B

####################################################################
#--------------------------- Generators ----------------------------
####################################################################

class G(nn.Module):
    def __init__(self, output_dim_a, output_dim_b, num_residule_block=5, ngf=64):
        super(G, self).__init__()

        self.relu1_m_A = nn.LeakyReLU(0.2, True)
        self.conv_m_A = nn.Conv2d(in_channels=ngf*2*2, out_channels=ngf*4, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn_m_A = nn.BatchNorm2d(ngf*4)
        self.relu2_m_A = nn.ReLU(True)
        self.residual_layer_A = self.make_layer(Residule_Block, num_residule_block, ngf*4)
        self.relu1_d_A = nn.ReLU(True)
        self.conv1_d_A = nn.Conv2d(ngf*4*2, ngf*2, kernel_size=3, stride=1, padding=1)
        self.bn1_d_A = nn.BatchNorm2d(ngf*2)
        self.relu2_d_A = nn.ReLU(True)
        self.up2_A = functools.partial(nn.functional.interpolate, scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2_d_A = nn.Conv2d(ngf*2*2, ngf, kernel_size=3, stride=1, padding=1)
        self.bn2_d_A = nn.BatchNorm2d(ngf)
        self.relu3_d_A = nn.ReLU(True)
        self.up3_A = functools.partial(nn.functional.interpolate, scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3_d_A = nn.Conv2d(ngf*1*2, ngf, kernel_size=3, stride=1, padding=1)
        self.bn3_d_A = nn.BatchNorm2d(ngf)
        self.relu4_d_A = nn.ReLU(True)
        self.conv4_d_A = nn.Conv2d(in_channels=ngf, out_channels=output_dim_a, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh_A = nn.Tanh()


        self.relu1_m_B = nn.LeakyReLU(0.2, True)
        self.conv_m_B = nn.Conv2d(in_channels=ngf*2*2, out_channels=ngf*4, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn_m_B = nn.BatchNorm2d(ngf*4)
        self.relu2_m_B = nn.ReLU(True)
        self.residual_layer_B = self.make_layer(Residule_Block, num_residule_block, ngf*4)
        self.relu1_d_B = nn.ReLU(True)
        self.conv1_d_B = nn.Conv2d(ngf*4*2, ngf*2, kernel_size=3, stride=1, padding=1)
        self.bn1_d_B = nn.BatchNorm2d(ngf*2)
        self.relu2_d_B = nn.ReLU(True)
        self.up2_B = functools.partial(nn.functional.interpolate, scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2_d_B = nn.Conv2d(ngf*2*2, ngf, kernel_size=3, stride=1, padding=1)
        self.bn2_d_B = nn.BatchNorm2d(ngf)
        self.relu3_d_B = nn.ReLU(True)
        self.up3_B = functools.partial(nn.functional.interpolate, scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3_d_B = nn.Conv2d(ngf*1*2, ngf, kernel_size=3, stride=1, padding=1)
        self.bn3_d_B = nn.BatchNorm2d(ngf)
        self.relu4_d_B = nn.ReLU(True)
        self.conv4_d_B = nn.Conv2d(in_channels=ngf, out_channels=output_dim_b, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh_B = nn.Tanh()

    def make_layer(self, block, num_residule_block, num_channel):
        layers = []
        for _ in range(num_residule_block):
            layers.append(block(num_channel))
        return nn.Sequential(*layers)

    def forward_a(self, e1_1_B, e1_2_B, e2_2_A):
        m_1_A = torch.cat([e1_2_B, e2_2_A], 1)
        # m_1_A (ngf*2*2, 54, 54)
        m_2_A = self.bn_m_A(self.conv_m_A(self.relu1_m_A(m_1_A)))
        # m_2_A (ngf*4, 54, 54)
        res_layer_A = self.residual_layer_A(self.relu2_m_A(m_2_A))
        # res_layer_A (ngf*4, 54, 54)

        d_0_A = torch.cat([res_layer_A, m_2_A], 1)
        # d_0_A (ngf*4*2, 54, 54)
        d_1_A = self.bn1_d_A(self.conv1_d_A(self.relu1_d_A(d_0_A)))
        # d_1_A (ngf*2, 54, 54)
        d_1_A = torch.cat([d_1_A, e1_2_B], 1)

        d_2_A = self.bn2_d_A(self.conv2_d_A(self.up2_A(self.relu2_d_A(d_1_A))))
        # d_2_A (ngf, 108, 108)
        d_2_A = torch.cat([d_2_A, e1_1_B], 1)

        d_3_A = self.bn3_d_A(self.conv3_d_A(self.up3_A(self.relu3_d_A(d_2_A))))
        # d_3_A (ngf, 216, 216)

        out_A = self.tanh_A(self.conv4_d_A(self.relu4_d_A(d_3_A)))
        # out_A (3, 216, 216)
        return out_A

    def forward_b(self, e1_1_A, e1_2_A, e2_2_B):
        m_1_B = torch.cat([e1_2_A, e2_2_B], 1)
        # m_1_B (ngf*2*2, 54, 54)
        m_2_B = self.bn_m_B(self.conv_m_B(self.relu1_m_B(m_1_B)))
        # m_2_B (ngf*4, 54, 54)
        res_layer_B = self.residual_layer_B(self.relu2_m_B(m_2_B))
        # res_layer_B (ngf*4, 54, 54)

        d_0_B = torch.cat([res_layer_B, m_2_B], 1)
        # d_0_B (ngf*4*2, 54, 54)
        d_1_B = self.bn1_d_B(self.conv1_d_B(self.relu1_d_B(d_0_B)))
        # d_1_B (ngf*2, 54, 54)
        d_1_B = torch.cat([d_1_B, e1_2_A], 1)
        # d_1_B (ngf*2*2, 54, 54)
        d_2_B = self.bn2_d_B(self.conv2_d_B(self.up2_B(self.relu2_d_B(d_1_B))))
        # d_2_B (ngf, 108, 108)
        d_2_B = torch.cat([d_2_B, e1_1_A], 1)
        # d_2_B (ngf*1*2, 108, 108)
        d_3_B = self.bn3_d_B(self.conv3_d_B(self.up3_B(self.relu3_d_B(d_2_B))))
        # d_3_B (ngf, 216, 216)
        out_B = self.tanh_B(self.conv4_d_B(self.relu4_d_B(d_3_B)))
        # out_B (3, 216, 216)
        return out_B

####################################################################
#------------------------- Basic Functions -------------------------
####################################################################

def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    elif opts.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opts.lr_policy)
    return scheduler

def init_weights(net, init_type, gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fainplanes')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, gpu, init_type='normal', gain=0.02):
    assert(torch.cuda.is_available())
    net.to(gpu)
    init_weights(net, init_type, gain)
    return net

####################################################################
#-------------------------- Basic Blocks --------------------------
####################################################################

# conv + (spectral) + (instance) + leakyrelu
class LeakyReLUConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(outplanes, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class Residule_Block(nn.Module):
    def __init__(self, nc):
        super(Residule_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nc)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nc)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(x)
        out = self.bn2(out)
        return out + x
