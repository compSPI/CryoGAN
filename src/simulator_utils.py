"""contains simulator and its components classes."""

import os
import mrcfile
import torch
from transforms import fourier_to_primal_2D, primal_to_fourier_2D
from ctf_utils import CTF
from noise_utils import Noise
from projector_utils import Projector
from shift_utils import Shift


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


def init_cube(sidelen):
    """Create a volume with cube.

    Parameters
    ----------
    sidelen: int
        size of the volume

    Returns
    -------
    volume: torch.Tensor
        Tensor (sidelen,sidelen,sidelen) with a cube
        of size (sidelen//2,sidelen//2,sidelen//2)
    """
    L = sidelen // 2
    length = sidelen // 8
    volume = torch.zeros([sidelen] * 3)
    volume[
        L - length : L + length, L - length : L + length, L - length : L + length
    ] = 1
    return volume



"""Module to generate data using using liner forward model."""


class LinearSimulator(torch.nn.Module):
    """Class to generate data using liner forward model.
    Parameters
    ----------
    config: class
        Class containing parameters of the simulator
    """

    def __init__(self, config, initial_volume=None):
        super(LinearSimulator, self).__init__()

        self.config = config
        self.projector = Projector(config)  # class for tomographic projector
        #self.init_volume()  # changes the volume inside the projector
        if self.config.ctf:
            self.ctf = CTF(config)  # class for ctf
        if self.config.shift:
            self.shift = Shift(config)  # class for shifts
        self.noise = Noise(config)  # class for noise
        self.proj_scalar= torch.nn.Parameter(torch.Tensor([0]))

       

    def forward(self, rotmat, ctf_params, shift_params, noise_params):
        """Create cryoEM measurements using input parameters.
        Parameters
        ----------
        rot_params: dict of type str to {tensor}
            Dictionary of rotation parameters for a projection chunk
        ctf_params: dict of type str to {tensor}
            Dictionary of Contrast Transfer Function (CTF) parameters
             for a projection chunk
        shift_params: dict of type str to {tensor}
            Dictionary of shift parameters for a projection chunk
        Returns
        -------
        projection.real : tensor
            Tensor ([chunks,1,sidelen,sidelen]) contains cryoEM measurement
        """
        proj={}
        proj.update({"rotmat": rotmat,
              "ctf_params": ctf_params,
              "shift_params": shift_params,
              "noise_params": noise_params})
        
        projection_tomo = self.projector(rotmat)
        
        if ctf_params is not None or shift_params is not None:
            f_projection_tomo = primal_to_fourier_2D(projection_tomo)
            f_projection_ctf = self.ctf(f_projection_tomo, ctf_params)
            f_projection_shift = self.shift(f_projection_ctf, shift_params)
            projection_clean = fourier_to_primal_2D(f_projection_shift).real  
            projection_ctf=fourier_to_primal_2D(f_projection_ctf).real
            proj.update({"ctf": projection_ctf})
            
        else:
            projection_clean=projection_tomo
            
        projection_clean=torch.exp(self.proj_scalar[0]) *projection_clean
        
        projection = self.noise(projection_clean, noise_params)
        
        proj.update({"tomo":projection_tomo, 
                     "clean":projection_clean,
                     "proj":projection
                    })
        return proj

    def init_volume(self):
        """Initialize the volume inside the projector.
        Initializes the mrcfile whose path is given in config.input_volume_path.
        If the path is not given or doesn't exist then the volume
        is initialized with a cube.
        """
        if (
            self.config.input_volume_path == ""
            or os.path.exists(os.path.join(os.getcwd(), self.config.input_volume_path))
            is False
        ):

            print(
                "No input volume specified or the path doesn't exist. "
                "Using cube as the default volume."
            )
            volume = init_cube(self.config.side_len)
        else:
            with mrcfile.open(self.config.input_volume_path, "r") as m:
                volume = torch.from_numpy(m.data.copy()).to(self.projector.vol.device)

        self.projector.vol = volume