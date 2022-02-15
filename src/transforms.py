import torch
def primal_to_fourier_2D(r):
    r = torch.fft.ifftshift(r, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.fftn(r, s=(r.shape[-2], r.shape[-1]), dim=(-2, -1)), dim=(-2, -1))

def fourier_to_primal_2D(f):
    f = torch.fft.ifftshift(f, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.ifftn(f, s=(f.shape[-2], f.shape[-1]), dim=(-2, -1)), dim=(-2, -1))

