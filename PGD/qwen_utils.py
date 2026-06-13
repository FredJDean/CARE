import torch
from tqdm import tqdm
import random
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import seaborn as sns
from PIL import Image
import math
import copy

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18

temporal_patch_size = 2
resized_height = 224
resized_width = 224
patch_size = 14
merge_size = 2
channel = 3
grid_t = 2 // temporal_patch_size
grid_h = resized_height // patch_size
grid_w = resized_width // patch_size

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

def resize2image(patch):
    assert patch.shape[0] == 256
    flattened_patches = patch.reshape(
            grid_t * grid_h * grid_w, channel, temporal_patch_size, patch_size, patch_size
        )
    patches = flattened_patches.reshape(
                grid_t,
                grid_h // merge_size,
                grid_w // merge_size,
                merge_size,
                merge_size,
                channel,
                temporal_patch_size,
                patch_size,
                patch_size,
            )
    patches = patches.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
    image = patches.reshape(
                temporal_patch_size,
                channel,
                resized_height,
                resized_width
            )
    image = image[0].unsqueeze(0)
    return image

def resize2patch(image):
    assert image.shape[2]==224 and image.shape[3]==224
    if image.shape[0] == 1:
        image = image.repeat(temporal_patch_size, 1, 1, 1)

    channel = image.shape[1]
    grid_t = image.shape[0] // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size

    patches = image.reshape(
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)

    flattened_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )
    return flattened_patches

class Attacker:

    def __init__(self, args, model, processor, prompts, targets, device='cuda:0', is_rtp=False, image_processor=None):

        self.args = args
        self.model = model
        self.processor = processor
        self.device = device
        self.is_rtp = is_rtp

        self.message = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": None,
                            "resized_height": 224,
                            "resized_width": 224,
                        },
                        {"type": "text", "text": ""},
                ],}]
        
        self.prompts = prompts
        self.targets = targets
        self.num_targets = len(targets)

        self.loss_buffer = []

        self.model.eval()
        self.model.requires_grad_(False)

        self.image_processor = image_processor
        self.same_seeds(100)
        
        self.p_init = 0.8

    def same_seeds(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(seed)

    def attack_unconstrained(self, img, image_grid_thw, batch_size=8, num_iter=2000, alpha=1/255):

        print('>>> batch_size:', batch_size)
       
        adv_noise = torch.rand_like(img).to(self.model.device) * 2 * 32.0 / 255 - 32.0 / 255
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()
    
        for t in tqdm(range(num_iter + 1)):

            indices = random.sample(range(len(self.targets)), k=batch_size)
            batch_targets = [self.targets[ii] for ii in indices]
            batch_prompts = [self.prompts[ii] for ii in indices]
            
            x_adv = resize2image(adv_noise)
            x_adv = normalize(x_adv)
            x_adv = resize2patch(x_adv)
        
            target_loss = self.attack_loss(batch_prompts, x_adv, image_grid_thw, batch_targets)
            target_loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(0, 1)
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())

            # if t % 20 == 0:
            #     self.plot_loss()

            if t % 500 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = resize2image(adv_noise)
                x_adv = normalize(x_adv)
                x_adv = resize2patch(x_adv)

                self.message[0]["content"][1]["text"] = random.sample(self.prompts, 1)
                text_prompt = self.processor.apply_chat_template(
                        self.message, tokenize=False, add_generation_prompt=True
                    )
                input_prompt = self.processor(text=[text_prompt], images=[Image.new("RGB", (224, 224), (0, 0, 0))], padding=True, return_tensors='pt')
                input_prompt = input_prompt.to(self.model.device)

                generated_ids = self.model.generate(
                                input_ids=input_prompt['input_ids'],
                                attention_mask=input_prompt['attention_mask'],
                                pixel_values=x_adv,
                                image_grid_thw=image_grid_thw,
                                max_new_tokens=256,
                            )
                generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_prompt.input_ids, generated_ids)
                    ]
                output_text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                
                print('>>>', output_text)

                adv_img_prompt = denormalize(resize2image(x_adv)).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                # save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt

    def attack_constrained(self, img, image_grid_thw, batch_size=8, num_iter=2000, alpha=1/255, epsilon=128/255):

        print('>>> batch_size:', batch_size)

        img = resize2image(img)
        adv_noise = torch.rand_like(img).to(self.model.device) * 2 * 16.0 / 255 - 16.0 / 255
        x = denormalize(img).clone().to(self.model.device)

        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        adv_noise = adv_noise.to(self.model.device)
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()
        
        for t in tqdm(range(num_iter + 1)):

            indices = random.sample(range(len(self.targets)), k=batch_size)
            batch_targets = [self.targets[ii] for ii in indices]
            batch_prompts = [self.prompts[ii] for ii in indices]

            x_adv = x + adv_noise
            x_adv = normalize(x_adv)
            x_adv = resize2patch(x_adv)

            target_loss = self.attack_loss(batch_prompts, x_adv, image_grid_thw, batch_targets)
            target_loss.backward()
            
            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())
            
            # if t % 20 == 0:
            #     self.plot_loss()

            if t % 250 == 0:
                print('######### Output - Iter = %d ##########' % t)
                
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)
                x_adv = resize2patch(x_adv)

                self.message[0]["content"][1]["text"] = random.sample(self.prompts, 1)
                text_prompt = self.processor.apply_chat_template(
                        self.message, tokenize=False, add_generation_prompt=True
                    )
                input_prompt = self.processor(text=[text_prompt], images=[Image.new("RGB", (224, 224), (0, 0, 0))], padding=True, return_tensors='pt')
                input_prompt = input_prompt.to(self.model.device)

                generated_ids = self.model.generate(
                                input_ids=input_prompt['input_ids'],
                                attention_mask=input_prompt['attention_mask'],
                                pixel_values=x_adv,
                                image_grid_thw=image_grid_thw,
                                max_new_tokens=256,
                            )
                generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_prompt.input_ids, generated_ids)
                    ]
                output_text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                
                print('>>>', output_text)

                adv_img_prompt = denormalize(resize2image(x_adv)).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                # save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt
    
    # def plot_loss(self):

    #     sns.set_theme()
    #     num_iters = len(self.loss_buffer)

    #     x_ticks = list(range(0, num_iters))

    #     plt.plot(x_ticks, self.loss_buffer, label='Target Loss')

    #     plt.title('Loss Plot')
    #     plt.xlabel('Iters')
    #     plt.ylabel('Loss')

    #     plt.legend(loc='best')
    #     plt.savefig('%s/loss_curve.png' % (self.args.save_dir))
    #     plt.clf()

    #     torch.save(self.loss_buffer, '%s/loss' % (self.args.save_dir))

    def attack_loss(self, prompts, images, image_grid_thw, targets):

        input_tokens = []
        input_masks = []
        
        for prompt in prompts:
            self.message[0]["content"][1]["text"] = prompt
            text_prompt = self.processor.apply_chat_template(
                    self.message, tokenize=False, add_generation_prompt=True
                )
            ori_query = self.processor(text=[text_prompt], images=[Image.new("RGB", (224, 224), (0, 0, 0))], padding=True, return_tensors="pt").to(self.model.device)
            input_tokens.append(ori_query['input_ids'])
            input_masks.append(ori_query['attention_mask'])

        batch_size = len(targets)
        images = images.repeat(batch_size, 1)
        
        to_regress_tokens = []
        to_regress_masks = []
        for target in targets:
            target_prompt = self.processor(text=target, padding=True, return_tensors="pt").to(self.model.device)
            to_regress_tokens.append(target_prompt['input_ids'])
            to_regress_masks.append(target_prompt['attention_mask'])

        seq_tokens_length = []
        labels = []
        input_ids = []
        for i, (item, input_token) in enumerate(zip(to_regress_tokens, input_tokens)):
            L = item.shape[1] + input_token.shape[1]
            seq_tokens_length.append(L)
            context_mask = torch.full([1, input_token.shape[1]], -100,
                                      dtype=to_regress_tokens[0].dtype,
                                      device=to_regress_tokens[0].device)
            labels.append(torch.cat([context_mask, item], dim=1))

            input_ids.append(torch.cat([input_token, item], dim=1))
        
        pad = torch.full([1, 1], 0,
                        dtype=to_regress_tokens[0].dtype,
                        device=to_regress_tokens[0].device
                    ).to(self.model.device)
        
        max_length = max(seq_tokens_length)
        attention_mask = []

        for i in range(batch_size):
            num_to_pad = max_length - seq_tokens_length[i]
            padding_mask = (
                    torch.full([1, num_to_pad], -100,
                    dtype=torch.long,
                    device=self.device)
                )
            labels[i] = torch.cat([labels[i], padding_mask], dim=1)
            input_ids[i] = torch.cat([input_ids[i],
                                       pad.repeat(1, num_to_pad)], dim=1)
            attention_mask.append(torch.LongTensor([[1]* (seq_tokens_length[i]) + [0]*num_to_pad]))

        labels = torch.cat(labels, dim=0).to(self.model.device)
        input_ids = torch.cat(input_ids, dim=0).to(self.model.device)
        attention_mask = torch.cat(attention_mask, dim=0).to(self.model.device)

        input_prompt = self.processor(text=[text_prompt]*batch_size, images=[Image.new("RGB", (224, 224), (0, 0, 0))]*batch_size, padding=True, return_tensors="pt").to(self.model.device)
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,
                 'pixel_values': images, 'image_grid_thw': input_prompt['image_grid_thw']}

        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        return loss