import sys
import numpy as np

from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from main import instantiate_from_config
import glob
from tqdm import tqdm
import torch
import os
import argparse


from ldm.models.diffusion.ddim import DDIMSampler

MAX_SIZE = 640

import cv2



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model.eval()
    return model


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images



def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)


    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(opt, sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512, image_name=''):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config1 = OmegaConf.load(opt.config)
    first_stage_model = load_model_from_config(config1, opt.ckpt)
    first_stage_model = first_stage_model.to(device)
    model = sampler.model
    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    with torch.no_grad():

        with torch.cuda.amp.autocast():
            batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

            c = model.cond_stage_model.encode(batch["txt"])

            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [num_samples, 4, h // 8, w // 8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            # cond = {"c_concat": [c_cat]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
            # uc_full = {"c_concat": [c_cat]}

            shape = [model.channels, h // 8, w // 8]
            samples_cfg, intermediates = sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=1.0,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc_full,
                x_T=start_code,
            )

            x_samples_ddim = first_stage_model.decode(1. / 0.18215 * samples_cfg, batch["image"], batch["mask"])


            result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                 min=0.0, max=1.0)

            result = result.cpu().numpy().transpose(0, 2, 3, 1)
            # result, has_nsfw_concept = check_safety(result)
            result = result * 255
            outpath = os.path.join(opt.outdir, os.path.split(image_name)[1])
            Image.fromarray(result[0].astype(np.uint8)).save(outpath)

    return result



def run(opt):
    # st.title("Stable Diffusion Inpainting")

    sampler = initialize_model(opt.config_d, opt.ckpt_d)
    masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
    images = [x.replace("_mask.png", ".png") for x in masks]

    for image_name, mask in tqdm(zip(images, masks)):
        image = Image.open(image_name)
        mask = Image.open(mask)
        w, h = image.size
        print(f"loaded input image of size ({w}, {h})")
        if max(w, h) > MAX_SIZE:
            factor = MAX_SIZE / max(w, h)
            w = int(factor * w)
            h = int(factor * h)
        width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((width, height))
        print(f"resized to ({width}, {height})")

        prompt = "photograph of a beautiful empty scene, highest quality settings"

        seed = 0
        num_samples = 1
        scale = 7.5
        ddim_steps = 50
        result = inpaint(
            opt=opt,
            sampler=sampler,
            image=image,
            mask=mask,
            prompt=prompt,
            seed=seed,
            scale=scale,
            ddim_steps=ddim_steps,
            num_samples=num_samples,
            h=height, w=width, image_name=image_name
        )





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="base.ckpt",
        help="dir to save checkpoint",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/mnt/output/git_results",
        help="dir to write results to",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/autoencoder/autoencoder_kl_32x32x4.yaml",
        help="dir to obtain config of decoder",
    )
    parser.add_argument(
        "--indir",
        type=str,
        default="/mnt/output/my_dataset/val",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--config_d",
        type=str,
        default="configs/autoencoder/v1-inpainting-inference.yaml",
        help="dir to obtain config of diffusion mdel",
    )
    parser.add_argument(
        "--ckpt_d",
        type=str,
        default="/mnt/output/pre_models/sd-v1-5-inpainting.ckpt",
        help="dir to save diffusion model",
    )


    opt = parser.parse_args()
    os.makedirs(opt.outdir, exist_ok=True)
    run(opt)
