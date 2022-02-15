import sys
import os
import torch
import torch.fft
import mrcfile
import starfile
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pytorch3d.transforms import rotation_6d_to_matrix, random_rotations, quaternion_to_matrix, matrix_to_quaternion, euler_angles_to_matrix
from simulator_utils import LinearSimulator
import matplotlib.pyplot as plt


def filter_rotation(rotmat):
    v=matrix_to_quaternion(rotmat)
    v[:,1]=v[:,1].abs()
    v[:,2]=v[:,2].abs()
    v_abs=v
    r_abs=quaternion_to_matrix(v_abs)
    return r_abs


def dataloader_from_dataset(dataset, config):

    out_dataloader = DataLoader(dataset,
                                shuffle=True, batch_size=config.batch_size,
                                pin_memory=True, num_workers=config.train_num_workers, drop_last=True)
    return out_dataloader



def dataloader( config):
    if config.simulated:
        data_loader=SimulatedDataLoader(config)
        noise_loader=SimulatedDataLoader(config, fake_params=True)
    else:
        dataset=RelionDataLoader(config)
        
        data_loader = dataloader_from_dataset(dataset, config)
        noise_loader = dataloader_from_dataset(noise_dataset, config)
    return data_loader, noise_loader


def remove_none_from_dict(dictionary):
    output = {k: v for k, v in dictionary.items() if v is not None}
    return output

def rotmat_generator(num, config):
        if config.protein== "betagal" :
            rotmat=filter_rotation(random_rotations(num))
        else:
            rotmat=random_rotations(num)
        return rotmat
    

class SimulatedDataLoader(Dataset):
    def __init__(self, config, fake_params=False):
        self.config=config
        self.fake_params=fake_params
        
        if not self.fake_params:
            self.sim=LinearSimulator(config).to(self.config.device)
            self.sim.projector.vol.requires_grad=False
            with torch.no_grad():
                self.sim.projector.vol[:,:,:]=self.init_gt_generator(self.config)[:,:,:]
            self.snr_specifier()
        
        self.normalizer=torch.nn.InstanceNorm2d(num_features=1, momentum=0.0)
        print("dataloader uses gaussian assumption for snr calculation\n")
        print("clean proj value scaled to change snr in gt\n")
        self.counter=0
        
    def __len__(self):
        return 41000//self.config.batch_size
    
    def get_samps_sim(self):
        ctf_params=None
        shift_params=None
        rotmat=rotmat_generator(self.config.batch_size, self.config).to(self.config.device)
        noise_params=None
        if not self.fake_params:
            with torch.no_grad():
                output_dict=self.sim(rotmat, ctf_params, shift_params, noise_params)
        else:
            output_dict={"noise_params": noise_params,
                       "rotmat": rotmat,
                       "shift_params": None,
                       "ctf_params": None

                      }
        return output_dict
        
    def __next__(self):
        self.counter+=1
#         if self.counter==self.__len__():
#             raise StopIteration
            
        output_dict=self.get_samps_sim()
        if self.config.normalize_gt:
            output_dict["proj"]=self.normalizer(output_dict["proj"])
        
        return output_dict
    def __iter__(self):
        return self
        
        
    def snr_specifier(self):
        save_mode=self.config.normalize_gt
        self.config.normalize_gt=False

        output_dict=self.get_samps_sim() 
        signal_energy=output_dict["clean"].pow(2).flatten(1).sum(1).sqrt()
        noise_energy=torch.randn_like(output_dict["clean"]).pow(2).flatten(1).sum(1).sqrt()
        mean_energy_ratio=(signal_energy/noise_energy).mean()
            
        sigma_val=mean_energy_ratio*(10**(-self.config.snr_val/20.0))
        inv_sigma_val=1/sigma_val
        with torch.no_grad():
            self.sim.proj_scalar[0]=torch.log(inv_sigma_val)
        self.config.normalize_gt=save_mode

    
 
        
    def init_gt_generator(self,config):
            L = config.side_len
        
            print("Protein is "+config.protein )
            if config.protein == "betagal" :
                with mrcfile.open("/sdf/home/h/hgupta/ondemand/CryoPoseGAN/figs/GroundTruth_Betagal-256.mrc") as m:
                    vol = torch.Tensor(m.data.copy()) / 10
                    

            elif config.protein == "ribo": 
                with mrcfile.open("/sdf/home/h/hgupta/ondemand/CryoPoseGAN/figs-old/GroundTruth_Ribo-64.mrc") as m:
                    vol = torch.Tensor(m.data.copy()) / 1000
                   
    
            elif config.protein == "cube":
                vol = torch.Tensor(init_cube(L)) / 50
                
            if config.side_len != vol.shape[-1]:
                        vol = torch.nn.AvgPool3d(kernel_size=vol.shape[-1]//config.side_len, stride=vol.shape[-1]//config.side_len)(vol[None, None, :, :, :])
                
            return vol
        
        
        
    
    def __getitem__(self, idx):
         pass
#         ctf_params=None
#         shift_params=None
#         rotmat=rotmat_generator(self.config.batch_size, self.config).to(self.config.device)
#         noise_params=None
#         with torch.no_grad():
#             output=self.sim(rotmat, ctf_params, shift_params, noise_params)
        
#         if self.config.normalize_gt:
#             output["proj"]=self.normalizer(output["proj"])
            
#         output.update({"idx":idx})
        
#         output=remove_none_from_dict(output)
        
        
#         filtered_output={}
#         for k, v in output.items():
#             if isinstance(v, torch.Tensor):
#                 filtered_output.update({k: v.squeeze(0)})
#             else:
#                 filtered_output.update({k:v})
       
#         return filtered_output

        

    

class RelionDataLoader(Dataset):
    def __init__(self, config, relion_invert_hand=False):
        self.relion_path = config.relion_path
        self.relion_star_file = config.relion_star_file
        self.df = starfile.open(os.path.join(self.relion_path, self.relion_star_file))

        self.vol_sidelen = self.df['optics']['rlnImageSize'][0]
        self.invert_hand = relion_invert_hand
        self.num_projs = len(self.df['particles'])

    def get_df_optics_params(self, config):
        config.side_len= self.df['optics']['rlnImageSize'][0]
        config.kv=self.df['optics']['rlnVoltage'][0]
        confg.pixel_size=  self.df['optics']['rlnImagePixelSize'][0]
        config.cs=self.df['optics']['rlnSphericalAberration'][0]
        config.amplitude_contrast=self.df['optics']['rlnAmplitudeContrast'][0]
        config.ctf_Size=config.side_len
        return config

    def __len__(self):
        return self.num_projs

    def __getitem__(self, idx):
        particle = self.df['particles'].iloc[idx]
        try:
            # Load particle image from mrcs file
            imgnamedf = particle['rlnImageName'].split('@')
            mrc_path = os.path.join(self.relion_path, imgnamedf[1])
            pidx = int(imgnamedf[0]) - 1
            with mrcfile.mmap(mrc_path, mode='r', permissive=True) as mrc:
                proj_np = mrc.data[pidx].copy()
                proj = torch.from_numpy(proj_np).float()
            proj = proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt)
            # --> (C,H,W)
        except Exception:
            print(f"WARNING: Particle image {particle['rlnImageName']} invalid!\nSetting to zeros.")
            proj = torch.zeros(self.vol_sidelen, self.vol_sidelen)
            proj = proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt)

        # Generate CTF from relion paramaters
        defocus_u = torch.from_numpy(np.array(particle['rlnDefocusU'] / 1e4, ndmin=2)).float()
        defocus_v = torch.from_numpy(np.array(particle['rlnDefocusV'] / 1e4, ndmin=2)).float()
        angleAstigmatism = torch.from_numpy(np.radians(np.array(particle['rlnDefocusAngle'], ndmin=2))).float()

        # Read relion "GT" orientations
        relion_euler_np = np.radians(np.stack([-particle['rlnAnglePsi'],                                     # convert Relion to our convention
                                               particle['rlnAngleTilt'] * (-1 if self.invert_hand else 1),   # convert Relion to our convention + invert hand
                                               -particle['rlnAngleRot']]))                                   # convert Relion to our convention
        rotmat = euler_angles_to_matrix(torch.from_numpy(relion_euler_np[np.newaxis,...]), convention='ZYZ')
        rotmat = torch.squeeze(rotmat).float()

        # Read relion "GT" shifts
        shiftX = torch.from_numpy(np.array(particle['rlnOriginXAngst']))
        shiftY = torch.from_numpy(np.array(particle['rlnOriginYAngst']))

        data_dict = {'proj': proj,
                   # The eventual groundtruth rotation
                   "rotmat": rotmat,
                   "ctf_params":{'defocus_u':defocus_u,
                                'defocus_v': defocus_v,
                                'defocus_angle': angleAstigmatism},
                    "shift_params":{'shift_x': shiftX,
                                       'shift_y': shiftY},
        
                   # The sample idx for any autodecoder type of model
                   'idx': torch.tensor(idx, dtype=torch.long)}  # this is the dict passed to the model


        return data_dict

    
    
    
    
    
    

        
