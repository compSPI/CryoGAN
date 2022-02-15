from cryogan import CryoGAN
from saveimage_utils import save_fig_double
import os
import mrcfile
import shutil
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from utils import mean_snr_calculator, dict2cuda
import pytorch3d
from pytorch3d.transforms import so3_relative_angle
from writer_utils import writer_image_add_dict
from src.dataio import dataloader



class CryoganWrapper():
    def __init__(self, config):
        super(CryoganWrapper, self).__init__()

        self.config = config

        self.cryogan = CryoGAN(config)
        self.cryogan.init_encoder()
        self.cryogan.init_gen()
        self.cryogan.init_dis()
        
        self.cryogan.to(self.config.device)
        self.gt_loader, self.noise_loader=dataloader( config)
        self.init_scheduler(self.cryogan)
        self.init_path()
        

    def run(self):
        total_epochs=400
        per_epoch_iteration=len(self.gt_loader)//(self.config.dis_iterations+1)
        max_iter=total_epochs*per_epoch_iteration
   
        iteration=-1
        for epoch in range(total_epochs):
            iter_loader=zip(self.gt_loader,self.noise_loader )
            
            for _ in range(per_epoch_iteration):
                iteration+=1
                    
                for dis_iter in range(self.config.dis_iterations + 1):
                    gt_data, fake_params=next(iter_loader)
                    gt_data=dict2cuda(gt_data)
                    fake_params=dict2cuda(fake_params)
                    train_all = dis_iter == self.config.dis_iterations
                    loss_dict, fake_data, self.writer = self.cryogan.train(gt_data, fake_params, max_iter, iteration,
                                                                                     self.writer, train_all)
                if self.config.cryogan:
                    self.scheduler_dis.step()
                self.scheduler_gen.step()
                self.scheduler_encoder.step()
                

                if iteration % 500 == 0: 
                     self.plot_images(gt_data, fake_data, iteration)
                     if self.config.cryogan:
                         wass_loss=loss_dict["loss_wass"]
                         print(f"iter: {iteration} loss_wass: {wass_loss}")
                     else:
                        print(f"iter: {iteration}")
            
        self.writer.close()

    def writer_add(self, loss_dict, iteration):
        for keys in loss_dict:
            self.writer.add_scalar("loss/" + keys, loss_dict[keys], iteration)

    def plot_images(self, gt_data, fake_data, rec_data, iteration):

       
        
        self.writer = writer_image_add_dict(self.writer, gt_data, fake_data, rec_data, iteration) 
        
        
        if fake_data is not None:
            save_fig_double(gt_data["proj"].cpu().data, fake_data["proj"].detach().cpu().data,
                        self.OUTPUT_PATH, "Proj", iteration=str(iteration).zfill(6),
                        Title1='Real', Title2='Fake_' + str(iteration),
                        doCurrent=True, sameColorbar=False)
        
            save_fig_double(gt_data["clean"].cpu().data, fake_data["clean"].detach().cpu().data,
                        self.OUTPUT_PATH, "Proj_clean", iteration=str(iteration).zfill(6),
                        Title1='Real_clean', Title2='fake_clean' + str(iteration),
                        doCurrent=True, sameColorbar=False)
        
        if rec_data is not None:
             
            with torch.no_grad():
                loss_rec_clean=(rec_data["clean"]-gt_data["clean"]).pow(2).sum()/self.config.batch_size
            self.writer.add_scalar("loss/loss_rec_clean", loss_rec_clean, iteration)

            
            save_fig_double(gt_data["clean"].cpu().data, rec_data["clean"].detach().cpu().data,
                        self.OUTPUT_PATH, "Proj_rec", iteration=str(iteration).zfill(6),
                        Title1='Real_clean', Title2='rec_clean' + str(iteration),
                        doCurrent=True, sameColorbar=False)
        
        volume_path=self.OUTPUT_PATH + '/'+str(iteration).zfill(6)+"_volume.mrc"
        with mrcfile.new(volume_path, overwrite=True) as m:
            m.set_data(self.cryogan.gen.projector.vol.detach().cpu().numpy())
            
        curr_volume_path=self.OUTPUT_PATH + '/current_volume.mrc'
        shutil.copy(volume_path, curr_volume_path)
            
        torch.save(self.cryogan.encoder, self.OUTPUT_PATH + "/Encoder.pt")
        torch.save(self.cryogan.gen.noise.scalar, self.OUTPUT_PATH + "/scalar.pt")


    
    def init_path(self):
        for path in ["/logs/", "/figs/"]:
            OUTPUT_PATH = os.getcwd() + path
            if os.path.exists(OUTPUT_PATH) == False:    os.mkdir(OUTPUT_PATH)
            OUTPUT_PATH = OUTPUT_PATH + self.config.exp_name
            if os.path.exists(OUTPUT_PATH) == False:    os.mkdir(OUTPUT_PATH)
            if "logs" in path:
                self.writer = SummaryWriter(log_dir=OUTPUT_PATH)

        self.OUTPUT_PATH = OUTPUT_PATH
        shutil.copy(self.config.config_path, self.OUTPUT_PATH)
  




    def init_scheduler(self, cryoposegan):

        self.scheduler_dis = torch.optim.lr_scheduler.StepLR(self.cryogan.dis_optim,
                                                             step_size=self.config.scheduler_step_size * self.config.dis_iterations,
                                                             gamma=self.config.scheduler_gamma)
        self.scheduler_gen = torch.optim.lr_scheduler.StepLR(self.cryogan.gen_optim,
                                                             step_size=self.config.scheduler_step_size,
                                                             gamma=self.config.scheduler_gamma)
        self.scheduler_encoder = torch.optim.lr_scheduler.StepLR(self.cryogan.encoder_optim,
                                                                 step_size=self.config.scheduler_step_size,
                                                                 gamma=self.config.scheduler_gamma)
        
#     def init_with_gt(self):
#         if self.config.init_with_gt:
#             with torch.no_grad():
#                 self.cryoposegan.gen.projector.vol[:, :, :] = self.GT.vol[:, :, :]
       
