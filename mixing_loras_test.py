# some code for later in main - mixing apporaches
import torch
#weight interpolation over diffusion process or merging outputs
#so far more or less notes
import os 
import numpy as np
from torch.nn import functional as F
from collections import OrderedDict

def static_weight_interpolation(model):
    lora_a = load_lora_weights("path")
    lora_b = load_lora_weights("path")

    alpha= 0.5


    interplated_lora = {
        key_ alpha * lora_a[key]+(1-alpha)*lora_b[key]
        for key in lora_a.keys() 
    }

    for key, value in interpolated_lora.items():
        model.state_dict()[key] += value
        
        return model


def apply_dynamically(model, lora1, lora2, alpha):
    
    def modified_forward(*args, **kwargs):
        original_weights = model.forward(*args, **kwargs)
        
         lora_a_out = load_lora_weights(model, lora_a, *args, **kwargs)
         lora_b_out = load_lora_weights(model, lora_b, *args, **kwargs)
         
         interpolated_output = alpha * lora_a_out + (1-alpha) * lora_b_out
         return interpolated_output + original_weights
     
     return modified_forward
model.forward = apply_dynamically()
        
        