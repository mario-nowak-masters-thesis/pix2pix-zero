import os
import argparse
import torch
from PIL import Image
import numpy as np

from utils.ddim_inv import DDIMInversion
from utils.scheduler import DDIMInverseScheduler

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noise_1', type=str, required=True)
    parser.add_argument('--input_noise_2', type=str, required=True)
    parser.add_argument('--results_folder', type=str, default='output/test_cat')
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--out_file_name', type=str, required=True)
    arguments = parser.parse_args()

    # make the output folders
    # os.makedirs(os.path.join(args.results_folder, "noise_difference_pixel_space"), exist_ok=True)
    os.makedirs(os.path.join(arguments.results_folder, "noise_difference_laten_space"), exist_ok=True)

    torch_dtype = torch.float32

    # make the DDIM inversion pipeline    
    pipe = DDIMInversion.from_pretrained(arguments.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    input_noise_1 = Image.open(arguments.input_noise_1).resize((512,512), Image.Resampling.LANCZOS)
    input_noise_2 = Image.open(arguments.input_noise_2).resize((512,512), Image.Resampling.LANCZOS)
    encoded_noise_1 = pipe.encode(input_noise_1)
    encoded_noise_2 = pipe.encode(input_noise_2)

    input_noise_1_array = np.array(input_noise_1, dtype=np.int64)
    input_noise_2_array = np.array(input_noise_2, dtype=np.int64)
    pixel_noise_difference = input_noise_1_array - input_noise_2_array
    pixel_noise_difference_absolute = np.absolute(pixel_noise_difference)
    pixel_noise_difference_image = Image.fromarray(pixel_noise_difference_absolute.astype(np.uint8))
    encoded_noise_difference = encoded_noise_1 - encoded_noise_2

    decoded_noise_difference = pipe.decode_latents(encoded_noise_difference)
    noise_difference_image = pipe.numpy_to_pil(decoded_noise_difference)

    noise_difference_image[0].save(
        os.path.join(arguments.results_folder, f"noise_difference_laten_space/{arguments.out_file_name}-latent.png")
    )
    pixel_noise_difference_image.save(
        os.path.join(arguments.results_folder, f"noise_difference_laten_space/{arguments.out_file_name}-pixel.png")
    )
