from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO
import argparse
from torchvision.utils import save_image
from qwen_utils import Attacker   

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-path", type=str, default="/home/siqingyi/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--n_iters", type=int, default=1000, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", default=False, action='store_true')
    parser.add_argument("--batchsize", type=int, default=1, help=' ')
    parser.add_argument("--save_dir", type=str, default='output', help="save directory")
    args = parser.parse_args()
    return args


print('>>> Initializing Models')
args = parse_args()
print('model = ', args.model_path)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype="auto", device_map="auto"
)

model.eval()
model.requires_grad_(False)
processor = AutoProcessor.from_pretrained(args.model_path)

print('[Initialization Finished]\n')

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

import csv
import pandas as pd
adv_bench_prompt = list(pd.read_csv("/home/siqingyi/fjh/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/harmful_corpus/advbench/train.csv")['prompt'])
adv_bench_target = list(pd.read_csv("/home/siqingyi/fjh/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/harmful_corpus/advbench/train.csv")['target'])
hhh_prompt = list(pd.read_csv("/home/siqingyi/fjh/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/harmful_corpus/anthropic_hhh/train.csv")['prompt'])
hhh_target = list(pd.read_csv("/home/siqingyi/fjh/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/harmful_corpus/anthropic_hhh/train.csv")['target'])

prompts = adv_bench_prompt + hhh_prompt
targets = adv_bench_target + hhh_target
assert len(prompts) == len(targets)
print("Len of prompts and targets: ", len(prompts))
print("Len of target prompts: ", len(targets))

def load_images_from_folder(folder, text_prompt_template, processor, device, image_size=224):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPEG', '.bmp', '.gif')):
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            img = img.resize((image_size, image_size))
            if img is not None:
                img = processor(text=[text_prompt_template], images=[img], return_tensors='pt').to(device)
                images.append(img['pixel_values'])
        if len(images)>110:
            break
    return images, img['image_grid_thw']

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": None,
                "resized_height": 224,
                "resized_width": 224,
            },
            {"type": "text", "text": ""},
        ],
    }
]
text_prompt_template = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(text_prompt_template)

folder_name = '/home/siqingyi/fjh/data/val2017'
imgs, image_grid_thw = load_images_from_folder(folder_name, text_prompt_template, processor, model.device)
print("Len of attack images: ", len(imgs))

unconstrained_save_dir = os.path.join(args.save_dir, "unconstrain")
if not os.path.exists(unconstrained_save_dir):
    os.mkdir(unconstrained_save_dir)

for index, image in enumerate(imgs):
        
    my_attacker = Attacker(args, model, processor, prompts, targets, device=model.device,)

    print('[unconstrained]')
    adv_img_prompt = my_attacker.attack_unconstrained(
                                            img=image, image_grid_thw=image_grid_thw,
                                            batch_size = args.batchsize,
                                            num_iter=args.n_iters, alpha=args.alpha/255 /4)
    save_image(adv_img_prompt, '{}/bad_prompt_{}_unconstrain.bmp'.format(args.save_dir, index))                                       
    
    for eps in (16, 32, 64):
        constrained_save_dir = os.path.join(args.save_dir, f"constrain_{eps}")
        if not os.path.exists(constrained_save_dir):
            os.mkdir(constrained_save_dir)
        print(f'[constrain {eps}]')
        adv_img_prompt = my_attacker.attack_constrained(
                                                img=image, image_grid_thw=image_grid_thw,
                                                batch_size= args.batchsize,
                                                num_iter=args.n_iters, alpha=args.alpha / 255 /4,
                                                epsilon= eps / 255)
    
        save_image(adv_img_prompt, '{}/bad_prompt_{}_constrain_{}.bmp'.format(constrained_save_dir, index, eps))
    
    print("Processing {}/{}".format(index+1, len(imgs)))

    print('[Done]')