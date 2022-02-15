import torch
from utils import get_samps_simulator
import pytorch3d
from pytorch3d.transforms import so3_relative_angle, matrix_to_rotation_6d
import numpy as np
def calculate_loss_dis(dis, real_data, fake_data, config):
    fake_samps=fake_data["proj"]
    real_samps=real_data["proj"]
    fake_out = dis(fake_samps)
    real_out = dis(real_samps)
    
    val_dis_fake= torch.mean(fake_out)
    val_dis_real=torch.mean(real_out)
    
    loss_wass=val_dis_real-val_dis_fake

    gp = config.lambdapenalty * stable_gradient_penalty_cryogan(dis, real_samps, fake_samps)
    loss_dis=-loss_wass+gp
        
    loss_dict = {"loss_dis":loss_dis,
            "loss_wass": loss_wass.item(),
            "loss_gp": gp.item()}

    return loss_dict

    
def calculate_loss_gen( dis, fake_data, config):
    loss_dict={}
    

    fake_samps=fake_data["proj"]
    fake_out = dis(fake_samps)
    val_dis_fake= torch.mean(fake_out)
    loss_gen_fake=-val_dis_fake


    loss_dict.update({"loss_gen":loss_gen_fake})
    #=======================

    return loss_dict
    

    
def dict_to_loss_dis(loss_dict, weight_dict):
    return loss_dict["loss_dis"]

def dict_to_loss(loss_dict, weight_dict):
    loss=0
    weight_total=0
    for keys in weight_dict:
        
        if weight_dict[keys]<0:
            raise AssertionError(keys+" val is negative")
     
            
        if weight_dict[keys]>0:
            loss+=weight_dict[keys]*loss_dict[keys[7:]]
        weight_total+=weight_dict[keys]
    loss=loss/weight_total
    return loss
    
    
    

def stable_gradient_penalty_cryogan(dis, real_samps, fake_samps):
    """
    private helper for calculating the gradient penalty
    :param real_samps: real samples
    :param fake_samps: fake samples
    :param reg_lambda: regularisation lambda
    :return: tensor (gradient penalty)
    """
    batch_size = real_samps.shape[0]

    # generate random epsilon
    epsilon = torch.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

    # create the merge of both real and fake samples
    merged = epsilon * real_samps + ((1 - epsilon) * fake_samps)
    merged.requires_grad_(True)

    # forward pass
    op = dis(merged)

    # perform backward pass from op to merged for obtaining the gradients
    gradients = torch.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=torch.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    # Return gradient penalty
    #print("wrong gradient penalty computations")
    return ((gradients_norm  - 1) ** 2).mean()

