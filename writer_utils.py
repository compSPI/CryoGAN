import torchvision


def writer_scalar_add_dict(writer, dict_weights, iteration, prefix=None):
    
    for keys in dict_weights:
            if prefix is not None:
                name=prefix+keys
            writer.add_scalar(name, dict_weights[keys], iteration)
    return writer
            
            
def writer_image_add_dict(writer, real_data, fake_data, rec_data, iteration):
    if real_data is not None:
        grid = torchvision.utils.make_grid(real_data["proj"][:16].data.cpu())
        writer.add_image("images/current_real",grid, iteration)
        
        
        grid = torchvision.utils.make_grid(real_data["clean"][:16].data.cpu())
        writer.add_image("images/current_real_clean",grid, iteration)
    
    if fake_data is not None:
        grid = torchvision.utils.make_grid(fake_data["proj"][:16].data.cpu())
        writer.add_image("images/current_fake",grid, iteration)
    
        grid = torchvision.utils.make_grid(fake_data["clean"][:16].data.cpu())
        writer.add_image("images/current_fake_clean",grid, iteration)
    
    
    if rec_data is not None:
        grid = torchvision.utils.make_grid(rec_data["clean"][:16].data.cpu())
        writer.add_image("images/current_rec_clean",grid, iteration)
    return writer


    
def norm_of_weights(module):
        dictionary={}
        for params in module.named_parameters():
            if ("weight" in params[0] and  any(name in params[0] for name in ["conv", "mlp"])) or  "vol" in params[0]:
                dictionary.update({params[0]+"/_weight":params[1].data })
                dictionary.update({params[0]+"/_weight_norm":params[1].data.norm() })
                if params[1].grad is not None:
                    dictionary.update({params[0]+"/_grad": params[1].grad.data})
                    dictionary.update({params[0]+"/_grad_norm_rel": (params[1].grad.data.norm().item()/params[1].data.norm()).item()})
        return dictionary
    
def writer_update_weight(module, writer, iteration):
    dict_weights=norm_of_weights(module)
    for keys in dict_weights:
        if "norm" in keys:
            writer.add_scalar(keys, dict_weights[keys], iteration)
            
        else:
            writer.add_histogram(keys, dict_weights[keys], iteration)
    return writer



def dict_from_gen(gen):
    weights_dict = {}
    weights_dict.update({"gen/vol":gen.projector.vol.data.cpu()})
    if gen.vol.grad is not None:
        weights_dict.update({"gen/vol_grad": gen.projector.vol.grad.data.cpu()})
    return weights_dict

def writer_hist_add_dict(writer, weights_dict, iteration):

    for keys in weights_dict:
        writer.add_histogram(keys,weights_dict[keys], iteration )
        if "output" in keys or "vol" in keys:
            writer.add_scalar("energy/"+keys,weights_dict[keys].abs().mean(), iteration)
            if weights_dict[keys].grad is not None:
                writer.add_scalar("energy/" + keys, weights_dict[keys].grad.abs().mean(), iteration)
    return writer
