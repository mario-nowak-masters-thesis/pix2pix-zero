import os, pdb
from glob import glob
import argparse
import numpy as np
import torch
import requests
from PIL import Image

from lavis.models import load_model_and_preprocess

from utils.ddim_inv import DDIMInversion
from utils.scheduler import DDIMInverseScheduler

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='assets/test_images/cat_a.png')
    parser.add_argument('--results_folder', type=str, default='output/test_cat')
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--out_file_name', type=str)
    args = parser.parse_args()

    # make the output folders
    os.makedirs(os.path.join(args.results_folder, "inversion"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "inversion_image"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "prompt"), exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32


    # make the DDIM inversion pipeline    
    pipe = DDIMInversion.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)


    # if the input is a folder, collect all the images as a list
    if os.path.isdir(args.input_image):
        l_img_paths = sorted(glob(os.path.join(args.input_image, "*.png")))
    else:
        l_img_paths = [args.input_image]


    for index, img_path in enumerate(l_img_paths):
        bname = f"{args.out_file_name}-{index}" if args.out_file_name else os.path.basename(img_path).split(".")[0]
        img = Image.open(img_path).resize((512,512), Image.Resampling.LANCZOS)
        encoded_image = pipe.encode(img)
        # save the inversion
        torch.save(encoded_image[0], os.path.join(args.results_folder, f"inversion/{bname}.pt"))
