# Asymmetric VQGAN




[**Designing a Better Asymmetric VQGAN for StableDiffusion**](https://arxiv.org/)<br/>


- Our pre-trained models are available: 
  - [A large 2x deocder](https://drive.google.com/file/d/1Qt40285nFNGBzS5iklZeEjq7ST4ExEiH/view?usp=drive_link)
  - [A large 1.5x decoder](https://drive.google.com/file/d/1m6c5XV6ZW1amGmjEaihvQl2VgxUxyO2f/view?usp=drive_link)
  - [A base decoder](https://drive.google.com/file/d/1jT_otqlNO6AhkOqCEZQY0KEqOyeCNTzi/view?usp=drive_link)
  
## Requirements

```
pip install -r requirements.txt
pip install wandb
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
```

# Pretrained diffusion Models
The inpainting model sd-v1-5-inpainting.ckpt of [StableDiffusion](https://github.com/runwayml/stable-diffusion/tree/main) is [here](https://huggingface.co/runwayml/stable-diffusion-inpainting/blob/main/sd-v1-5-inpainting.ckpt)

The text2image model v1-5-pruned-emaonly.ckpt of [StableDiffusion](https://github.com/runwayml/stable-diffusion/tree/main) is [here](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt)

## Inapinting task
Download our [images and masks ](https://drive.google.com/file/d/1Z9_vGdvs7i8RTQ9GN8RNX1y5i1tP_OSI/view?usp=drive_link).
```
python inpaint_st.py  --config {config_spec}
```
where `config_spec` is one of {`autoencoder_kl_32x32x4.yaml`(base decoder), `autoencoder_kl_32x32x4_large.yaml`(large decode 1.5x), 

`autoencoder_kl_32x32x4_large2.yaml`(large decoder 2x).

## Text2image task

```
python txt2img.py --plms --config_c {config_spec}
```

[//]: # (# Train your own asymmetric vqgan)

[//]: # ()
[//]: # (## Data preparation)

[//]: # ()
[//]: # (### ImageNet)

[//]: # (The code will try to download &#40;through [Academic)

[//]: # (Torrents]&#40;http://academictorrents.com/&#41;&#41; and prepare ImageNet the first time it)

[//]: # (is used. However, since ImageNet is quite large, this requires a lot of disk)

[//]: # (space and time. If you already have ImageNet on your disk, you can speed things)

[//]: # (up by putting the data into)

[//]: # (`./datasets/ImageNet/train`. It should have the following structure:)

[//]: # ()
[//]: # (```)

[//]: # (./datasets/ImageNet/train/)

[//]: # (├── n01440764)

[//]: # (│   ├── n01440764_10026.JPEG)

[//]: # (│   ├── n01440764_10027.JPEG)

[//]: # (│   ├── ...)

[//]: # (├── n01443537)

[//]: # (│   ├── n01443537_10007.JPEG)

[//]: # (│   ├── n01443537_10014.JPEG)

[//]: # (│   ├── ...)

[//]: # (├── ...)

[//]: # (```)

[//]: # ()
[//]: # (### Training autoencoder models)

[//]: # ()
[//]: # (First, download [weights]&#40;https://drive.google.com/file/d/1RaOlCRnkGeCv2Nig-bhHuApNJoA98gfg/view?usp=drive_link&#41; of the autoencoder stable_vqgan.ckpt obtained from [StableDiffusion]&#40;https://github.com/runwayml/stable-diffusion/tree/main&#41;.)

[//]: # ()
[//]: # (Configs for training a KL-regularized autoencoder on ImageNet are provided at `configs/autoencoder`.)

[//]: # (Training can be started by running)

[//]: # (```)

[//]: # (python main.py --base configs/autoencoder/{config_spec} -t --gpus 0,1,2,3,4,5,6,7 --tag <yourtag>   )

[//]: # (```)

where `config_spec` is one of {`autoencoder_kl_woc_32x32x4.yaml`(base decoder), `autoencoder_kl_woc_32x32x4_large.yaml`(large decode 1.5x), 

`autoencoder_kl_woc_32x32x4_large2.yaml`(large decoder 2x).


## Comments 

- Our codebase for the diffusion models builds heavily on [StableDiffusion](https://github.com/runwayml/stable-diffusion/tree/main). 
Thanks for open-sourcing!

[//]: # (- The implementation of the asymmetric vqgan is from [PUT]&#40;https://github.com/liuqk3/PUT&#41; and [Lama]&#40;https://github.com/advimman/lama&#41;. )


## BibTeX

```



```


