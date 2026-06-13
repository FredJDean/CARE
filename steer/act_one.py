import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoProcessor
import json
import math
from PIL import ImageDraw, ImageFont



class JailBreakBenchSteeringDataset(Dataset):
    """
    基于JailBreakVBench CSV的数据集
    CSV格式: [编号, 提示指令, 恶意问题, 图像路径, ...]
    """
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        processor,
        num_pairs: int,
        benign_mode: bool = False,
        benign_image_dir: str = None
    ):
        """
        Args:
            csv_path: JailBreakVBench CSV路径
            image_root: 对抗图片根目录
            processor: 模型processor
            benign_mode: True=使用良性数据, False=使用恶意数据
            benign_image_dir: 良性图片目录（COCO图片）
        """
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.processor = processor
        self.benign_mode = benign_mode
        self.benign_image_dir = benign_image_dir
        self.samples = []


        if benign_mode == True:
            with open("/root/fujinhu/llava-interp/data/behavior_refusal.json", "r") as f:
                behavior = json.load(f)
            temp = 0
            for j in range(num_pairs):
                temp += 1
                if temp > 99:
                    temp = 0
                row = self.df.iloc[j]
                self.samples.append({
                    "goal": row["jailbreak_query"]+row["redteam_query"],
                    "target": behavior['non_compliant_responses'][temp],
                    "image_path": row['image_path']
                })
        else:
            with open("/root/fujinhu/llava-interp/data/behavior_refusal.json", "r") as f:
                behavior = json.load(f)
            temp = 0
            for j in range(num_pairs):
                temp += 1
                if temp > 99:
                    temp = 0
                row = self.df.iloc[j]
                self.samples.append({
                    "goal": row["jailbreak_query"]+row["redteam_query"],
                    "target": behavior['compliant_responses'][temp],
                    "image_path": f"{image_root}/{row['image_path']}"
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")

        text = item["goal"] + "\nAssistant: " + item["target"]
        return {
            'text': text,
            'image': image
        }


class PGDDataset(Dataset):
    """
    基于JailBreakVBench CSV的数据集
    CSV格式: [编号, 提示指令, 恶意问题, 图像路径, ...]
    """
    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        processor,
        num_pairs: int,
        benign_mode: bool = False,
        benign_image_dir: str = None
    ):
        """
        Args:
            csv_path: PGD path
            image_root: 对抗图片根目录
            processor: 模型processor
            benign_mode: True=使用良性数据, False=使用恶意数据
            benign_image_dir: 良性图片目录（COCO图片）
        """
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.processor = processor
        self.benign_mode = benign_mode
        self.benign_image_dir = benign_image_dir
        self.samples = []

        with open("/root/fujinhu/llava-interp/data/behavior_refusal.json", "r") as f:
            behavior_refusal = json.load(f)

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= num_pairs:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # 跳过格式有问题的行
                    continue

                behavior_text = row.get("behavior", "")
                save_path = row.get("save_path", "")

                # 如果提供了 image_root 并且 save_path 不是绝对路径，就拼接
                if save_path and not os.path.isabs(save_path) and self.image_root:
                    save_path = os.path.join(self.image_root, save_path)
                
                idx = i % max(1, len(behavior_refusal.get('compliant_responses', []) or [0]))

                if self.benign_mode:
                    # non_compliant_responses 用于良性模式（原代码逻辑）
                    non_compliant = behavior_refusal.get('non_compliant_responses', [])
                    # if non_compliant:
                    target = non_compliant[idx % len(non_compliant)]
                else:
                    compliant = behavior_refusal.get('compliant_responses', [])
                    target = compliant[idx % len(compliant)]  
                
                self.samples.append({
                    "goal": behavior_text,
                    "target": target,
                    "save_path": save_path
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["save_path"]).convert("RGB")

        text = item["goal"] + "\nAssistant: " + item["target"]
        return {
            'text': text,
            'image': image
        }

class AdvBenchDataset(Dataset):
    """
    从 dataset/advbench 和 dataset/advimage 读取数据
    CSV 格式: [goal, target]
    """
    def __init__(self, bench_root: str, image_root: str, processor, benign_mode):
        """
        Args:
            bench_root: 包含多个 advbench CSV 文件的目录
            image_root: 对应的 advimage 根目录
            processor: 模型 processor
            benign_mode: True=良性回复, False=恶意回复
        """
        self.processor = processor
        self.samples = []

        csv_files = sorted([
            f for f in os.listdir(bench_root)
            if f.endswith('.csv')
        ])

        print(f"Found {len(csv_files)} CSV files in {bench_root}")

        for i, csv_file in enumerate(csv_files):
            csv_path = os.path.join(bench_root, csv_file)
            df = pd.read_csv(csv_path)

            img_dir = os.path.join(image_root, str(i))
            if not os.path.exists(img_dir):
                print(f"⚠️ Warning: Image dir not found for {csv_file}: {img_dir}")
                continue

            img_files = sorted([
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.endswith(('.jpg', '.png', '.jpeg', '.JPEG', '.webp'))
            ])

            num_pairs = min(len(df), len(img_files))
            if benign_mode == True:
                with open("/root/fujinhu/llava-interp/data/behavior_refusal.json", "r") as f:
                    behavior = json.load(f)
                temp = 0
                for j in range(num_pairs):
                    temp += 1
                    if temp > 99:
                        temp = 0
                    row = df.iloc[j]
                    self.samples.append({
                        "goal": row["goal"],
                        "target": behavior['non_compliant_responses'][temp],
                        "image_path": img_files[j]
                    })
            else:
                for j in range(num_pairs):
                    row = df.iloc[j]
                    self.samples.append({
                        "goal": row["goal"],
                        "target": row["target"],
                        "image_path": img_files[j]
                    })

        print(f"✅ Total samples collected: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        text = item["goal"] + "\nAssistant: " + item["target"]
        return {"text": text, "image": image}


class FigStepDataset(Dataset):
    """
    基于FigStep SafeBench CSV的数据集
    CSV格式: [dataset, category_id, task_id, category_name, question, instruction]
    图像命名: query_{dataset}_{category_id}_{task_id}_6.png
    """
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        processor,
        num_pairs,
        benign_mode: bool = False
    ):
        """
        Args:
            csv_path: FigStep SafeBench CSV路径 (默认: /root/fujinhu/FigStep/data/question/safebench.csv)
            image_root: 图像根目录 (默认: /root/fujinhu/FigStep/data/images/SafeBench)
            processor: 模型processor
            benign_mode: True=使用良性回复, False=使用恶意回复
        """
        self.csv_path = csv_path
        self.image_root = image_root
        self.processor = processor
        self.benign_mode = benign_mode
        self.samples = []

        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 加载行为拒绝响应模板
        with open("/root/fujinhu/llava-interp/data/behavior_refusal.json", "r") as f:
            behavior_refusal = json.load(f)

        print(f"Loading FigStep dataset from {csv_path}")
        print(f"Total rows in CSV: {len(df)}")

        # 遍历CSV中的每一行
        for idx, row in df.iterrows():
            if idx > num_pairs:
                break
            dataset = row['dataset']
            category_id = row['category_id']
            task_id = row['task_id']
            question = row['question']
            
            # 构造图像文件名: query_{dataset}_{category_id}_{task_id}_6.png
            image_filename = f"query_{dataset}_{category_id}_{task_id}_6.png"
            image_path = os.path.join(self.image_root, image_filename)
            
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                print(f"⚠️ Warning: Image not found: {image_path}")
                continue
            
            # 根据模式选择目标回复
            if self.benign_mode:
                # 良性模式：使用non_compliant_responses（拒绝回复）
                non_compliant = behavior_refusal.get('non_compliant_responses', [])
                target = non_compliant[idx % len(non_compliant)] if non_compliant else "I cannot assist with that request."
            else:
                # 恶意模式：使用compliant_responses（配合回复）
                compliant = behavior_refusal.get('compliant_responses', [])
                target = compliant[idx % len(compliant)] if compliant else "Sure, I can help with that."
            
            self.samples.append({
                "goal": question,
                "target": target,
                "image_path": image_path,
                "category": row['category_name'],
                "dataset": dataset
            })

        print(f"✅ Successfully loaded {len(self.samples)} FigStep samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 加载图像
        image = Image.open(item["image_path"]).convert("RGB")
        
        # 构造文本：问题 + 助手回复
        text = item["goal"] + "\nAssistant: " + item["target"]
        
        return {
            'text': text,
            'image': image
        }


class COCODataset(Dataset):
    def __init__(self, json_path: str, processor):
        self.processor = processor
        self.samples = []

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ JSON file not found: {json_path}")

        # 读取JSON数据
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data = data[:360]
        print(f"Found {len(data)} COCO inference samples in {json_path}")

        # 解析样本
        for item in data:
            
            desc = item.get("description", "")
            if "assistant\n" in desc:
                # 只取 assistant 后面的文本
                assistant_text = desc.split("assistant\n", 1)[-1].strip()
            else:
                assistant_text = desc.strip()

            self.samples.append({
                "goal": "Please describe the image.",
                "target": assistant_text,
                "image_path": item["image_path"]
            })

        print(f"✅ Total samples collected: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回与 AdvBenchDataset 一致的结构:
        {
          "text": "Please describe the image.\nAssistant: <assistant_text>",
          "image": PIL.Image
        }
        """
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        text = item["goal"] + "\nAssistant: " + item["target"]
        return {"text": text, "image": image}


def prepare_malicious_dataset(bench_root, image_root, processor):
    """构建合并后的 AdvBench 数据集"""
    malicious_dataset = AdvBenchDataset(
        bench_root=bench_root,
        image_root=image_root,
        processor=processor,
        benign_mode=False
    )
    
    print(f"Final dataset size: {len(malicious_dataset)}")
    return malicious_dataset


def custom_collate_fn(batch):
    """自定义collate函数，处理PIL Image对象"""
    texts = [item['text'] for item in batch]
    images = [item['image'] for item in batch]
    
    return {
        'text': texts,
        'image': images
    }


def compute_visual_text_mi_scores(visual_acts, text_acts):
    """
    计算视觉token与文本token之间的互信息分数 (MI Score)
    
    Args:
        visual_acts: np.ndarray, shape [n_visual, hidden_dim]
        text_acts: np.ndarray, shape [n_text, hidden_dim]
        method: str, {'cosine', 'attention', 'attention_weighted', 'hsic'}
        attention_weights: np.ndarray, shape [n_visual, n_text] (optional)
    
    Returns:
        mi_scores: np.ndarray, shape [n_visual]
    """
    if len(visual_acts) == 0 or len(text_acts) == 0:
        return np.zeros(len(visual_acts))

    eps = 1e-8  # 防止除零
    n_v, n_t = len(visual_acts), len(text_acts)
    # hidden_dim = visual_acts.shape[1]
    # ---- 新HSIC实现：视觉与文本之间的cross-RBF核 ----
    def rbf_cross_kernel(X, Y, sigma=None):
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)
        # 计算 pairwise 距离矩阵
        XX = np.sum(X ** 2, axis=1, keepdims=True)
        YY = np.sum(Y ** 2, axis=1, keepdims=True)
        dists = XX + YY.T - 2 * np.dot(X, Y.T)
        dists = np.clip(dists, 0, 1e6)  # 防止数值爆炸
        if sigma is None:
            nonzero_dists = dists[dists > 0]
            sigma = np.sqrt(0.5 * np.median(nonzero_dists) + eps)
        K = np.exp(-dists / (2 * sigma ** 2 + eps))
        return K

    # cross-kernel between visual and text tokens
    K_cross = rbf_cross_kernel(visual_acts, text_acts)

    # 中心化（按文本维度中心化）
    H_t = np.eye(n_t) - np.ones((n_t, n_t)) / n_t
    K_cross_centered = K_cross @ H_t

    # 视觉token与文本整体依赖性
    mi_scores = np.sum(K_cross_centered ** 2, axis=1) 
    # print(mi_scores.sum())
    mi_scores = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + eps)

    return mi_scores


def _infer_grid_dims(n_visual):
    """
    推断最接近的网格尺寸 (rows, cols) 使 rows*cols >= n_visual，且形状尽量接近正方形。
    """
    if n_visual <= 0:
        return 0, 0
    # 若是完全平方
    r = int(math.floor(math.sqrt(n_visual)))
    if r * r == n_visual:
        return r, r
    # 否则找最接近的行列配对
    for rows in range(r, 0, -1):
        cols = math.ceil(n_visual / rows)
        if rows * cols >= n_visual:
            return rows, cols
    return n_visual, 1

def visualize_selected_visual_tokens(
    image_path: str,
    selected_indices: np.ndarray,
    n_visual: int,
    save_path: str,
    annot_text: str = None,
    outline_width: int = 2,
    alpha: float = 0.45
):
    """
    在原图上高亮 selected_indices 对应的 patch（visual token）。
    - image_path: 原始图片路径
    - selected_indices: 1D array of token indices in [0, n_visual)
    - n_visual: visual token 的总数（用于推断网格）
    - save_path: 保存标注图的路径（含文件名）
    """
    img = Image.open(image_path).convert("RGBA")
    W, H = img.size
    rows, cols = _infer_grid_dims(n_visual)
    if rows == 0 or cols == 0:
        print(f"Warning: invalid grid dims for n_visual={n_visual}")
        return

    patch_w = W / cols
    patch_h = H / rows

    # 创建 overlay
    overlay = Image.new('RGBA', img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)

    # 字体（尽量使用系统默认小字体）
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(12, int(min(patch_w, patch_h) // 4)))
    except Exception:
        font = ImageFont.load_default()

    for rank, tok_idx in enumerate(selected_indices):
        # token index -> row, col
        r = int(tok_idx // cols)
        c = int(tok_idx % cols)
        left = int(c * patch_w)
        top = int(r * patch_h)
        right = int((c + 1) * patch_w)
        bottom = int((r + 1) * patch_h)

        # 半透明填充
        fill_color = (255, 0, 0, int(255 * alpha))  # 红色半透明
        draw.rectangle([left, top, right, bottom], fill=fill_color)

        # 边框
        for w in range(outline_width):
            draw.rectangle([left + w, top + w, right - w, bottom - w], outline=(255,255,255,255))

        # 写序号 (从1开始)
        txt = str(rank + 1)
        bbox = draw.textbbox((0, 0), txt, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tx = left + 4
        ty = top + 4
        draw.rectangle([tx - 2, ty - 2, tx + text_w + 2, ty + text_h + 2], fill=(0,0,0,150))
        draw.text((tx, ty), txt, fill=(255,255,255,255), font=font)

    if annot_text:
        # 在底部写一条小说明
        bbox = draw.textbbox((0, 0), annot_text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rectangle([4, H - th - 10, 4 + tw + 8, H - 4], fill=(0,0,0,150))
        draw.text((8, H - th - 8), annot_text, fill=(255,255,255,255), font=font)

    # 合成并保存
    out = Image.alpha_composite(img, overlay).convert("RGB")
    out.save(save_path)

    return save_path


def extract_activations_with_visual_mi(
    model,
    processor,
    dataset: Dataset,
    layer_idx: int,
    component: str = 'act',
    model_type: str = 'qwen',
    batch_size: int = 4,
    top_k: int = 10,
    save_visual_dir: str = "./visualizations"
):
    """
    在原有提取激活的基础上，额外可视化选中的 visual token 在图片中的位置并保存。
    返回值与原函数相似，同时会在 save_visual_dir 下保存标注图。
    """
    os.makedirs(save_visual_dir, exist_ok=True)

    activations = []
    current_input_ids = None
    sample_counter = 0  # 全局样本计数（按 dataloader 顺序）

    def hook_fn(module, input, output):
        nonlocal current_input_ids, sample_counter

        if component == "act":
            out_seq = output[0]  # [batch, seq_len, hidden]
        else:
            # 若传入其他 component，根据需要调整（此处假设 output 格式一致）
            out_seq = output[0]
        batch_b = out_seq.shape[0]

        for b_idx in range(batch_b):
            seq_act = out_seq[b_idx].detach().cpu().numpy()  # [seq_len, hidden_dim]
            input_ids = current_input_ids[b_idx]

            # 找到 image_end_token_id
            image_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            image_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

            image_start_indices = (input_ids == image_start_token_id).nonzero(as_tuple=True)[0]
            image_end_indices = (input_ids == image_end_token_id).nonzero(as_tuple=True)[0]

            image_start_pos = image_start_indices[-1].item()
            image_end_pos = image_end_indices[-1].item()

            visual_start = image_start_pos
            visual_end = image_end_pos
            n_visual = visual_end - visual_start
            text_start = image_end_pos + 1
            text_end = len(seq_act)

            top_k = int(n_visual/8)
            # print("top k:", top_k)

            visual_acts = seq_act[visual_start:visual_end]
            text_acts = seq_act[text_start:text_end]

            if len(visual_acts) == 0 or len(text_acts) == 0:
                activations.append(np.mean(seq_act, axis=0))
                continue

            try:
                mi_scores = compute_visual_text_mi_scores(visual_acts, text_acts)
            except Exception as e:
                print("MI computation error:", e)
                mi_scores = np.zeros(len(visual_acts))

            # 选择 top_k（与原逻辑保持一致）
            k_actual = min(top_k, len(mi_scores))
            top_k_indices = np.argsort(mi_scores)[-k_actual:][::-1]
            top_k_indices = top_k_indices[mi_scores[top_k_indices] > 1e-6]
            # print(top_k_indices)
            if len(top_k_indices) > 0:
                selected_acts = visual_acts[top_k_indices]
                avg_act = np.mean(selected_acts, axis=0)
                activations.append(avg_act)
            else:
                activations.append(np.mean(visual_acts, axis=0))

            # ========== 可视化部分 ==========
            # global_idx = sample_counter + b_idx
            # if 0 <= global_idx < len(dataset.samples):
            #     img_path = dataset.samples[global_idx]["image_path"]
            #     # 防御性：确保 indices 在范围内
            #     safe_indices = np.array([int(i) for i in top_k_indices if 0 <= i < n_visual], dtype=np.int64)
            #     if safe_indices.size > 0:
            #         save_name = f"sample_{global_idx}_layer{layer_idx}_top{len(safe_indices)}.jpg"
            #         save_file = os.path.join(save_visual_dir, save_name)
            #         annot = f"sample={global_idx} topk={len(safe_indices)}"
            #         try:
            #             visualize_selected_visual_tokens(
            #                 image_path=img_path,
            #                 selected_indices=safe_indices,
            #                 n_visual=n_visual,
            #                 save_path=save_file,
            #                 annot_text=annot
            #             )
            #         except Exception as e:
            #             print(f"Visualization failed for {img_path}: {e}")
            # ================================

        # 结束 batch，不在这里自增 sample_counter，因为在主循环后计数更安全

    # 获取目标模块（与你原函数一致）
    if model_type.lower() == 'qwen':
        base_model = model.model if hasattr(model, 'model') else model.base_model
        if component == 'mlp':
            target_module = base_model.language_model.layers[layer_idx].mlp
        elif component == 'attn':
            target_module = base_model.language_model.layers[layer_idx].self_attn
        elif component == 'act':
            target_module = base_model.language_model.layers[layer_idx]
    elif model_type.lower() == 'llava':
        if component == 'mlp':
            target_module = model.language_model.model.layers[layer_idx].mlp
        elif component == 'attn':
            target_module = model.language_model.model.layers[layer_idx].self_attn
        elif component == 'act':
            target_module = model.language_model.model.layers[layer_idx]

    hook = target_module.register_forward_hook(hook_fn)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0  # 为可视化稳定，建议0
    )
    device = next(model.parameters()).device

    print(f"Extracting activations from layer {layer_idx}, component: {component}")
    print(f"Selecting top-{top_k} visual tokens and saving visualizations to {save_visual_dir}")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Layer {layer_idx}"):
            texts = batch['text']
            images = batch['image']

            # 构建 messages_list（复用你原来的逻辑）
            messages_list = []
            for text in texts:
                user_part, assistant_part = text.split("Assistant:", 1)
                messages_list.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": None},
                            {"type": "text", "text": user_part.strip()},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": assistant_part.strip()},
                        ],
                    },
                ])

            queries = [
                processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
                for msg in messages_list
            ]

            inputs = processor(
                text=queries,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(device)

            # 保存 input_ids 给 hook 使用
            current_input_ids = inputs['input_ids']

            _ = model(**inputs)

            # 批次处理完成，更新全局样本计数
            sample_counter += len(texts)

    hook.remove()

    activations = np.stack(activations) if len(activations) > 0 else np.zeros((0, model.config.hidden_size))
    print(f"Extracted shape: {activations.shape}")

    return activations


if __name__ == "__main__":
    # 激活保存路径
    save_path = "./activations/llava_visual_mi"
    os.makedirs(save_path, exist_ok=True)
    
    # 模型名称
    model_name = "/root/fujinhu/pretrain-models/LLaVA-OneVision-1.5-8B-Instruct"
    
    # 加载模型
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # 数据准备
    print("\nStep 1: Preparing datasets...")


    malicious_dataset1 = AdvBenchDataset(
        bench_root="/root/fujinhu/Jail-MLLM/dataset/advbench",
        image_root="/root/fujinhu/Jail-MLLM/dataset/advimage",
        processor=processor,
        benign_mode=False
    )


    malicious_dataset2 = JailBreakBenchSteeringDataset(
        csv_path="/root/fujinhu/data/JailBreakV_28K/JailBreakV_28K.csv",
        image_root="/root/fujinhu/data/JailBreakV_28K",
        processor=processor,
        benign_mode=False,
        num_pairs=100
    )

    malicious_dataset3 = FigStepDataset(
        csv_path="/root/fujinhu/FigStep/data/question/safebench.csv",
        image_root="/root/fujinhu/FigStep/data/images/SafeBench",
        processor=processor,
        benign_mode=False,
        num_pairs=100
    )

    from torch.utils.data import ConcatDataset

    malicious_dataset = ConcatDataset([malicious_dataset1, malicious_dataset2, malicious_dataset3])
    
    benign_dataset = COCODataset(
    json_path="/root/fujinhu/llava-interp/coco_val2017_infer_results.json",
    processor=processor)

    # 提取激活
    activations_benign = {}
    activations_malicious = {}
    
    # 关键参数
    top_k = 10  # 选择top-10视觉token
    
    target_layers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    
    for layer_idx in target_layers:
        print(f"\n{'='*80}")
        print(f"Processing Layer {layer_idx}")
        print(f"{'='*80}")
        
        # 提取恶意激活
        print("\nExtracting MALICIOUS activations...")
        malicious_acts = extract_activations_with_visual_mi(
            model, processor, malicious_dataset,
            layer_idx, "act", "qwen", 
            batch_size=2,
            top_k=top_k,
        )

        # 提取良性激活
        print("\nExtracting BENIGN activations...")
        benign_acts = extract_activations_with_visual_mi(
            model, processor, benign_dataset,
            layer_idx, "act", "qwen", 
            batch_size=2,
            top_k=top_k,
        )
        activations_benign[layer_idx] = benign_acts

        activations_malicious[layer_idx] = malicious_acts
    
    # 保存激活
    torch.save(activations_benign, os.path.join(save_path, f"img_benign_top{top_k}.pt"))
    torch.save(activations_malicious, os.path.join(save_path, f"img_malicious_top{top_k}.pt"))
    
    print("\n" + "="*80)
    print("Activation extraction complete!")
    print(f"Saved to: {save_path}")
    print(f"Top-K: {top_k}")
    print("="*80)