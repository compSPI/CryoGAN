"""Module to corrupt the projection with noise."""
import torch


class Noise(torch.nn.Module):
    """Class to corrupt the projection with noise.
    Written by J.N. Martel and H. Gupta.
    Parameters
    ----------
    config: class
        contains parameters of the noise distribution
    """

    def __init__(self, config):

        super(Noise, self).__init__()
        self.scalar = torch.nn.Parameter(torch.Tensor([0]))
        print("noise module sigma kept constant to 1")
    def forward(self, proj, noise_params ):
        """Add noise to projections.
        Currently, only supports additive white gaussian noise.
        Parameters
        ----------
        proj: torch.Tensor
            input projection of shape (batch_size,1,side_len,side_len)
        Returns
        -------
        out: torch.Tensor
            noisy projection of shape (batch_size,1,side_len,side_len)
        """
        
        noise_sigma = 1#torch.exp(self.scalar[0]) 
        
        if  noise_params is not None:
            if "noise" in noise_params:
                out = proj + noise_sigma * noise_params["noise"]
        else:
             out = proj + noise_sigma * torch.randn_like(proj)
        return out