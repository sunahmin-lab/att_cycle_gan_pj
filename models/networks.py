import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_scheduler(optimizer, opt):
    """
    Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    return scheduler


def get_norm_layer(norm_type='instance'):
    """
    Return a normalization layer
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


###################################################################################
## Your Implementation is from Here ##


def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'attention_basic':
        net = AttentionGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # you can modify the input variables and types
    elif netG == 'basic':
        net = BaselineGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # you can modify the input variables and types
    elif netG == 'makeup':
        net = MakeUpGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # you can modify the input variables and types (PA Step4)
    elif netG == 'basic_':
        net = BaselineGenerator_(input_nc, output_nc, ngf, norm_layer=norm_layer,
                              use_dropout=use_dropout)  # you can modify the input variables and types (PA Step4)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'attention_basic':  # self attention based discriminator
        net = AttentionDiscriminator(input_nc, ndf, n_layers=2, norm_layer=norm_layer)
    elif netD == 'basic':  # baseline discriminator
        net = BaselineDiscriminator(input_nc, ndf, n_layers=2, norm_layer=norm_layer)
    elif netD == 'makeup':
        net = MakeUpDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)  # you can modify the input variables and types (PA Step4)
    elif netD == 'basic_':
        net = BaselineDiscriminator_(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        """
        super(GANLoss, self).__init__()
        ## Your Implementation Here ##
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss() ## Your Implementation Here ##
        elif gan_mode == 'vanilla':
            self.loss = nn.BCELoss() ## Your Implementation Here ##
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        ## Your Implementation Here ##
        if target_is_real:
            target = torch.full_like(prediction, self.target_real_label)
        else:
            target = torch.full_like(prediction, self.target_fake_label)

        loss = self.loss(prediction, target)

        return loss

class MakeUpGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
        """Construct a generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(MakeUpGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        input_1 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, bias=True),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        input_2 = [nn.ReflectionPad2d(3),
                   nn.Conv2d(input_nc, ngf, kernel_size=7, bias=True),
                   norm_layer(ngf),
                   nn.ReLU(True)]

        model = []

        input_channel = ngf
        for _ in range(2):
            output_channel = input_channel * 2
            model += [nn.Conv2d(input_channel, output_channel,kernel_size=3, stride=2, padding=1),
                      norm_layer(output_channel),
                      nn.ReLU(True)]
            input_channel = output_channel

        for _ in range(6):
            model += [ResnetBlock(input_channel, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for _ in range(2):
            output_channel = input_channel // 2
            model += [nn.ConvTranspose2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(output_channel),
                      nn.ReLU(True)]
            input_channel = output_channel


        output_1 = []
        output_2 = []
        output_1 += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, 7),
                  nn.Tanh()]
        output_2 += [nn.ReflectionPad2d(3),
                   nn.Conv2d(input_nc, ngf, kernel_size=7, bias=True),
                   norm_layer(ngf),
                   nn.ReLU(True)]

        self.input_1 = nn.Sequential(*input_1)
        self.input_2 = nn.Sequential(*input_2)
        self.model = nn.Sequential(*model)
        self.output_1 = nn.Sequential(*output_1)
        self.output_2 = nn.Sequential(*output_2)

    def forward(self, input):
        """Standard forward"""
        output = self.input_1(input)
        output2 = self.input_2(input)
        output = torch.cat((output, output2), dim=1)
        output = self.model(output)

        return output


class BaselineGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
        """Construct a generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(BaselineGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, bias=True),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        input_channel = ngf
        for _ in range(2):
            output_channel = input_channel * 2
            model += [nn.Conv2d(input_channel, output_channel,kernel_size=3, stride=2, padding=1),
                      norm_layer(output_channel),
                      nn.ReLU(True)]
            input_channel = output_channel

        for _ in range(6):
            model += [ResnetBlock(input_channel, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for _ in range(2):
            output_channel = input_channel // 2
            model += [nn.ConvTranspose2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(output_channel),
                      nn.ReLU(True)]
            input_channel = output_channel

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class BaselineGenerator_(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
        """Construct a generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(BaselineGenerator_, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, bias=True),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        input_channel = ngf
        for _ in range(2):
            output_channel = input_channel * 2
            model += [nn.Conv2d(input_channel, output_channel,kernel_size=3, stride=2, padding=1),
                      norm_layer(output_channel),
                      nn.ReLU(True)]
            input_channel = output_channel

        for _ in range(9):
            model += [ResnetBlock(input_channel, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for _ in range(2):
            output_channel = input_channel // 2
            model += [nn.ConvTranspose2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(output_channel),
                      nn.ReLU(True)]
            input_channel = output_channel

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class AttentionGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
        """
        Construct generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(AttentionGenerator, self).__init__()

        self.att_out_ch = 0

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, bias=True),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        input_channel = ngf
        for _ in range(2):
            output_channel = input_channel * 2
            model += [nn.Conv2d(input_channel, output_channel, kernel_size=3,
                                                       stride=2, padding=1),
                      norm_layer(output_channel),
                      nn.ReLU(True)]

            input_channel = output_channel

        model += [Self_Attn(input_channel, 'relu')]

        for _ in range(9):
            model += [
                ResnetBlock(input_channel, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        for _ in range(2):
            output_channel = input_channel // 2
            model += [
                nn.ConvTranspose2d(input_channel, output_channel, kernel_size=3, stride=2,
                                                          padding=1, output_padding=1, bias=use_bias),
                norm_layer(output_channel),
                nn.ReLU(True)]
            input_channel = output_channel

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""

        return self.model(input)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        """
        in_dim   -- input feature's channel dim
        activation    -- activation function type
        """
        super(Self_Attn, self).__init__()
        self.input_channel = in_dim
        self.activation = activation
        self.k = 8
        self.key = nn.Conv2d(self.input_channel, self.input_channel // self.k, kernel_size=1)
        self.query = nn.Conv2d(self.input_channel, self.input_channel // self.k, kernel_size=1)
        self.value = nn.Conv2d(self.input_channel, self.input_channel, kernel_size=1)
        self.conv_1x1 = nn.Conv2d(self.input_channel, self.input_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channel, width, height = x.size()
        k = self.key(x)
        k = k.view(batch_size, channel//8, -1)
        q = self.query(x)
        q = q.view(batch_size, channel//8, -1) #(N, input_ch/k, h*W)
        s = torch.bmm(k.transpose(1,2), q) #(N, h*W, h*W)
        attention_map = self.softmax(s) #(2,128.128)
        v = self.value(x) #(N, input_ch, h, W)
        v = v.view(batch_size, channel, -1) #(N, input_ch, h*W)
        out = torch.bmm(v, attention_map.transpose(1,2)) #(N, h*W, input_ch)
        out = out.view(batch_size, channel, width, height)
        attention_fmap = self.conv_1x1(out)
        o = x + self.gamma * attention_fmap

        return o

class MakeUpDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(MakeUpDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        model += [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=use_bias),
                  norm_layer(ndf),
                  nn.LeakyReLU(0.2, True)]

        input_channel = ndf
        for _ in range(n_layers):
            output_channel = min(input_channel * 2, 512)
            model += [nn.Conv2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1, bias=use_bias),
                      norm_layer(output_channel),
                      nn.LeakyReLU(0.2, True)]
            input_channel = output_channel

        output_channel = min(input_channel*2, 512)
        model += [nn.Conv2d(input_channel, output_channel, kernel_size=4, padding=1, bias=use_bias),
                  norm_layer(output_channel),
                  nn.LeakyReLU(0.2, True),
                  nn.Conv2d(output_channel, 1, kernel_size=4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""

        return self.model(input)


class BaselineDiscriminator_(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(BaselineDiscriminator_, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        model += [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=use_bias),
                  norm_layer(ndf),
                  nn.LeakyReLU(0.2, True)]

        input_channel = ndf
        for _ in range(n_layers):
            output_channel = min(input_channel * 2, 512)
            model += [nn.Conv2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1, bias=use_bias),
                      norm_layer(output_channel),
                      nn.LeakyReLU(0.2, True)]
            input_channel = output_channel

        output_channel = min(input_channel*2, 512)
        model += [nn.Conv2d(input_channel, output_channel, kernel_size=4, padding=1, bias=use_bias),
                  norm_layer(output_channel),
                  nn.LeakyReLU(0.2, True),
                  nn.Conv2d(output_channel, 1, kernel_size=4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""

        return self.model(input)

class BaselineDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(BaselineDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        model += [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=use_bias),
                  norm_layer(ndf),
                  nn.LeakyReLU(0.2, True)]

        input_channel = ndf
        for _ in range(n_layers):
            output_channel = min(input_channel * 2, 512)
            model += [nn.Conv2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1, bias=use_bias),
                      norm_layer(output_channel),
                      nn.LeakyReLU(0.2, True)]
            input_channel = output_channel

        output_channel = min(input_channel*2, 512)
        model += [nn.Conv2d(input_channel, output_channel, kernel_size=4, padding=1, bias=use_bias),
                  norm_layer(output_channel),
                  nn.LeakyReLU(0.2, True),
                  nn.Conv2d(output_channel, 1, kernel_size=4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""

        return self.model(input)


class AttentionDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=2, norm_layer=nn.BatchNorm2d):
        """Construct discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(AttentionDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        model += [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=use_bias),
                  norm_layer(ndf),
                  nn.LeakyReLU(0.2, True)]

        input_channel = ndf

        for _ in range(n_layers):
            output_channel = min(input_channel * 2, 512)
            model += [nn.Conv2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1, bias=use_bias),
                      norm_layer(output_channel),
                      nn.LeakyReLU(0.2, True)]
            input_channel = output_channel
        output_channel = min(input_channel * 2, 512)

        model += [Self_Attn(input_channel, 'relu')]

        model += [nn.Conv2d(input_channel, output_channel, kernel_size=4, padding=1, bias=use_bias),
                  norm_layer(output_channel),
                  nn.LeakyReLU(0.2, True),
                  nn.Conv2d(output_channel, 1, kernel_size=4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""

        return self.model(input)
