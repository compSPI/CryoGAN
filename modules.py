import numpy
import torch
import torch.nn as nn
from ml_modules import FCBlock, CNNEncoder, UNET
from pytorch3d.transforms import  Rotate, rotation_6d_to_matrix, matrix_to_rotation_6d, matrix_to_quaternion
import numpy as np
from src.transforms import primal_to_fourier_2D, fourier_to_primal_2D




def downsample_fourier_crop(proj, size):

    sidelen=proj.shape[-1]
    center=sidelen//2
    l_min=center-size//2
    l_max=center+size//2+size%2
    
    proj_ft=primal_to_fourier_2D(proj)
    proj_ft_crop=proj_ft[:,:,l_min:l_max,l_min:l_max]
    proj_crop_down=fourier_to_primal_2D(proj_ft_crop)
    
    factor=(float(sidelen)/size)**2
    return proj_crop_down.real/factor


def weights_init(m, args):
    if isinstance(m, nn.Conv2d):

        if m.weight is not None:
            torch.nn.init.kaiming_normal_(m.weight, a=args.leak_value)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)

    if isinstance(m, nn.Linear):
        if m.weight is not None:
            torch.nn.init.kaiming_normal_(m.weight, a=args.leak_value)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)


class Discriminator(torch.nn.Module):
    def __init__(self, args):
        ''' 6: simple conv network with max pooling'''
        super(Discriminator, self).__init__()
        self.Fourier = args.FourierDiscriminator
        K = args.num_channel_Discriminator  # num channels
        N = args.num_N_Discriminator  # penultimate features
        numConvs = args.num_layer_Discriminator

        # first one halves the number of numbers, then multiplies by K
        # interval convolutions, each halves the number of values (because channels double)

        self.convs = nn.ModuleList(
            [torch.nn.Sequential(
                torch.nn.Conv2d(2 ** (i) * K ** (i > 0) + 2 * self.Fourier * (i == 0), 2 ** (i + 1) * K, kernel_size=3,
                                stride=1, padding=1),
                torch.nn.MaxPool2d(kernel_size=2),
                torch.nn.LeakyReLU(args.leak_value))
                for i in range(numConvs)]
        )

        # todo: have to think about how to handle this
        size = args.ProjectionSize

        # flatten down to N numbers, then 1 number
        # size=K * size**2 * 2**numConvs / 4**numConvs

        input = torch.zeros(1, 1 + self.Fourier * 2, int(size), int(size))
        with torch.no_grad():
            for conv in self.convs:
                input = conv(input)

        self.fully = torch.nn.Sequential(
            torch.nn.Linear(np.prod(input.size()), N),
            torch.nn.LeakyReLU(args.leak_value)
            # torch.nn.ReLU()
        )

        self.linear = torch.nn.Linear(N, 1)
        self.args = args
        self.normalizer=torch.nn.InstanceNorm2d(num_features=1, momentum=0.0)
       
    def forward(self, input_image):
   
        if self.args.normalize_dis_input:
            output=self.normalizer(input_image)
        else:
            output=input_image
            
        for conv in self.convs:
            output = conv(output)
        self.cnn_output=output
        
        output_linear= self.fully(self.cnn_output.flatten(1))
   
        self.output = self.linear(output_linear)

        return output

