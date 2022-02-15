import torch
import torch.fft
from torch import nn
from modules import  Discriminator, weights_init
from writer_utils import  writer_update_weight, writer_scalar_add_dict
from loss_utils import calculate_loss_dis, calculate_loss_gen, dict_to_loss, dict_to_loss_dis
from utils import get_samps_simulator
from src.simulator_utils import LinearSimulator

class CryoGAN(nn.Module):
    def __init__(self, config):
        super(CryoGAN, self).__init__()
        self.config = config

    def init_dis(self):

        self.dis = Discriminator(self.config)
        self.dis.apply(lambda m: weights_init(m, self.config))
        self.dis_optim = torch.optim.Adam(self.dis.parameters(),
                                          lr=self.config.dis_lr,
                                          betas=(self.config.dis_beta_1, self.config.dis_beta_2),
                                          eps=self.config.dis_eps,
                                          weight_decay=self.config.dis_weight_decay)

    def init_gen(self):
        self.gen = LinearSimulator(self.config)
        

        self.gen_optim = torch.optim.Adam(self.gen.parameters(),
                                          lr=self.config.gen_lr,
                                          betas=(self.config.gen_beta_1, self.config.gen_beta_2),
                                          eps=self.config.gen_eps,
                                          weight_decay=self.config.gen_weight_decay)


    def train(self, real_data, params, max_iter, iteration, writer, train_all=True):
        
        config=self.config
        rec_data=None
        loss_dict={}
        loss_dis_dict={}
        loss_unsup_dict={}
        loss_supervised_dict={}
        

        fake_data=get_samps_simulator(self.gen, params, grad=train_all)

        loss_dis_dict = calculate_loss_dis(self.dis, real_data, fake_data, self.config)
        loss_dis=dict_to_loss_dis(loss_dis_dict, self.config)
        loss_dis.backward(retain_graph=True)
        self.train_dis()
        self.zero_grad()
        loss_dict.update( ** loss_dis_dict)
            
        if train_all:

            loss_gen_dict=calculate_loss_gen(self.dis, self.gen,real_data, fake_data, self.config )
            weight_dict_gen={"weight_loss_gen": 1
                            }
            loss_gen=dict_to_loss(loss_gen_dict, weight_dict_gen)

        
            loss_gen.backward()
            if iteration%10==0:
                writer=writer_update_weight(self.gen, writer, iteration)
           
        
            self.train_gen()  
            self.train_enc()
            self.zero_grad()
            
            loss_dict={**loss_dis_dict, **loss_gen_dict}
            writer=writer_scalar_add_dict(writer, loss_dict, iteration, prefix="loss/")
            writer=writer_scalar_add_dict(writer, weight_dict_gen, iteration, prefix="coefficients/")
            writer.add_scalar("sigma_snr", torch.exp(-self.gen.proj_scalar.data[0]), iteration)

        return loss_dict, fake_data, writer


    def train_dis(self):
        
        if self.config.dis_clip_grad == True:
            torch.nn.utils.clip_grad_norm_(self.dis.parameters(), max_norm=self.config.dis_clip_norm_value)

        self.dis_optim.step()



    def train_gen(self):
    
            if self.config.gen_clip_grad == True:
                torch.nn.utils.clip_grad_norm_(self.gen.projector.parameters(), max_norm=self.config.gen_clip_norm_value)
                #torch.nn.utils.clip_grad_norm_(self.gen.noise.parameters(), max_norm=self.config.scalar_clip_norm_value)


            self.gen_optim.step()
            self.constraint()

    def zero_grad(self):
        self.dis_optim.zero_grad()
        self.gen_optim.zero_grad()


    def constraint(self):

            if self.config.positivity:
                with torch.no_grad():
                    self.gen.projector.vol.clamp_(min=0)



