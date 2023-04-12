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


def make_output_folders(args: argparse.Namespace) -> None:
    os.makedirs(os.path.join(args.results_folder, "merged_inversion"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "merged_inversion_image"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "prompt"), exist_ok=True)


def load_image(image_path: str) -> Image.Image:
    assert os.path.isfile(image_path)
    image = Image.open(image_path).resize((512,512), Image.Resampling.LANCZOS)

    return image


def merge_image_tensors(upper_image: torch.Tensor, lower_image: torch.Tensor) -> torch.Tensor:
    _, _, upper_image_height, _ = upper_image.size()
    upper_image_portion = upper_image[:, :, :upper_image_height//2, :]

    _, _, lower_image_height, _ = lower_image.size()
    lower_image_portion = lower_image[:, :, lower_image_height//2:, :]

    merged_image: torch.Tensor = torch.cat((upper_image_portion, lower_image_portion), dim=2)

    return merged_image



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--upper_input_image', type=str, default='assets/test_images/cats/cat_5.png')
    parser.add_argument('--lower_input_image', type=str, default='assets/test_images/dogs/dog_1.png')
    parser.add_argument('--results_folder', type=str, default='output/merging_test_cat_dog')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--out_file_name', type=str)
    args = parser.parse_args()

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # load the BLIP model
    # model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=torch.device(device))

    # make the DDIM inversion pipeline    
    ddim_inversion_pipe: DDIMInversion = DDIMInversion.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    ddim_inversion_pipe.scheduler = DDIMInverseScheduler.from_config(ddim_inversion_pipe.scheduler.config)

    # generate the caption
    # _image = vis_processors["eval"](img).unsqueeze(0).to(device)
    # prompt_str = model_blip.generate({"image": _image})[0]
    # prompt_str = "a photograph of a grown golden retriever standing in a forest surrounded by trees on a sunny day"

    upper_image = load_image(args.upper_input_image)
    lower_image = load_image(args.lower_input_image)
    prompt = "an image"

    inverted_upper_image, _, _ = ddim_inversion_pipe(
        img=upper_image,
        prompt=prompt, 
        guidance_scale=1,
        num_inversion_steps=args.num_ddim_steps,
        torch_dtype=torch_dtype
    )
    inverted_lower_image, _, _ = ddim_inversion_pipe(
        img=lower_image,
        prompt=prompt, 
        guidance_scale=1,
        num_inversion_steps=args.num_ddim_steps,
        torch_dtype=torch_dtype
    )

    merged_image = merge_image_tensors(inverted_upper_image, inverted_lower_image).squeeze(dim=0)

    # save the inversion
    make_output_folders(args)
    upper_image_base_name: str = os.path.basename(args.upper_input_image).split(".")[0]
    lower_image_base_name: str = os.path.basename(args.lower_input_image).split(".")[0]
    output_file_name = args.out_file_name or f"{upper_image_base_name}-{lower_image_base_name}"

    torch.save(merged_image, os.path.join(args.results_folder, f"merged_inversion/{output_file_name}.pt"))

    # save the image 
    # x_inv_image[0].save(os.path.join(args.results_folder, f"merged_inversion_image/{output_file_name}.png"))

    # save the prompt string
    with open(os.path.join(args.results_folder, f"prompt/{output_file_name}.txt"), "w") as f:
        f.write(prompt)
