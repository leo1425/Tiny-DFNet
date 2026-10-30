#import df.checkpoint as cp
from df.checkpoint import load_model as load_model_cp
import os
from df.config import config
from df.model import ModelParams
from libdf import DF

# model_base_dir = "/Users/leopoldh/Library/CloudStorage/GoogleDrive-notreplystopstop@gmail.com/My Drive/Memoire/OwnModels/ownexec/"

# if not os.path.isdir(model_base_dir):
#     raise NotADirectoryError("Base directory not found at {}".format(model_base_dir))
# config.load(
#     os.path.join(model_base_dir, "config.ini"),
#     config_must_exist=True,
#     allow_defaults=True,
#     allow_reload=True,
# )
# p = ModelParams()
# df_state = DF(
#     sr=p.sr,
#     fft_size=p.fft_size,
#     hop_size=p.hop_size,
#     nb_bands=p.nb_erb,
#     min_nb_erb_freqs=p.min_nb_freqs,
# )
# checkpoint_dir = os.path.join(model_base_dir, "checkpoints")

# model, epoch = load_model_cp(checkpoint_dir, df_state, epoch=12, mask_only=False)

import torch

checkpoint_dir_low = '/Users/leopoldh/Documents/Drawio - Memoire/My Model/DF fine tune/small_model_0.5_without_conv/general_train/model/checkpoints'
checkpoint_dir_big = '/Users/leopoldh/Library/CloudStorage/GoogleDrive-notreplystopstop@gmail.com/My Drive/Memoire/OwnModels/ownexec/checkpoints/'
out_dir = '/Users/leopoldh/Documents/Drawio - Memoire/My Model/DF4 gated/using_small_model_v2_0.5/checkpoints'
raw_ckpt_path = os.path.join(checkpoint_dir_big, "model_12.ckpt.best")
original_state_dict = torch.load(raw_ckpt_path, map_location="cpu")

raw_ckpt_path_low = os.path.join(checkpoint_dir_low, "model_30.ckpt.best")
low_state_dict = torch.load(raw_ckpt_path_low, map_location="cpu")


new_state_dict = original_state_dict
for key, value in low_state_dict.items():
    if key.startswith("erb_dec."):
        new_key = key.replace("erb_dec.", "erb_dec_low.", 1)
        new_state_dict[new_key] = value.clone()
        print(f"Copied : {new_key}")

torch.save(new_state_dict, os.path.join(out_dir, "model_79.ckpt.best"))
