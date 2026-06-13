import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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
                    "image_path": f"{image_root}/{row['image_path']}"
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



def compute_text_token_mi_scores(text_acts, visual_acts=None, method='self_hsic'):
    """
    计算文本token的重要性分数
    
    Args:
        text_acts: np.ndarray, shape [n_text, hidden_dim] - 文本token激活
        visual_acts: np.ndarray, shape [n_visual, hidden_dim] - 视觉token激活（可选）
        method: str, {'self_hsic', 'cross_hsic', 'variance'}
            - 'self_hsic': 文本token之间的自相关性（捕捉独特信息）
            - 'cross_hsic': 文本token与视觉token的交叉相关性
            - 'variance': 简单的方差方法
    
    Returns:
        mi_scores: np.ndarray, shape [n_text] - 每个文本token的重要性分数
    """
    if len(text_acts) == 0:
        return np.zeros(0)
    
    eps = 1e-8
    n_t = len(text_acts)
    
    if method == 'self_hsic':
        # ---- 文本token自身的独特性（与其他文本token的区分度） ----
        def rbf_kernel(X, sigma=None):
            X = X.astype(np.float64)
            # 计算pairwise距离
            XX = np.sum(X ** 2, axis=1, keepdims=True)
            dists = XX + XX.T - 2 * np.dot(X, X.T)
            dists = np.clip(dists, 0, 1e6)
            
            if sigma is None:
                nonzero_dists = dists[dists > 0]
                if len(nonzero_dists) > 0:
                    sigma = np.sqrt(0.5 * np.median(nonzero_dists) + eps)
                else:
                    sigma = 1.0
            
            K = np.exp(-dists / (2 * sigma ** 2 + eps))
            return K, dists
        
        K, dists = rbf_kernel(text_acts)
        
        # 中心化
        H = np.eye(n_t) - np.ones((n_t, n_t)) / n_t
        K_centered = H @ K @ H
        
        # 每个token的独特性 = 它与其他token的平均距离
        # 距离越大，说明该token越独特，越重要
        avg_dist_to_others = np.mean(dists, axis=1)
        
        # 结合核矩阵的对角元素（自相关强度）
        self_similarity = np.diag(K_centered)
        
        # 综合分数：独特性（距离）+ 自相关强度
        mi_scores = 0.7 * avg_dist_to_others + 0.3 * np.abs(self_similarity)
        
    elif method == 'cross_hsic' and visual_acts is not None:
        # ---- 文本token与视觉token的交叉相关性 ----
        def rbf_cross_kernel(X, Y, sigma=None):
            X = X.astype(np.float64)
            Y = Y.astype(np.float64)
            
            XX = np.sum(X ** 2, axis=1, keepdims=True)
            YY = np.sum(Y ** 2, axis=1, keepdims=True)
            dists = XX + YY.T - 2 * np.dot(X, Y.T)
            dists = np.clip(dists, 0, 1e6)
            
            if sigma is None:
                nonzero_dists = dists[dists > 0]
                if len(nonzero_dists) > 0:
                    sigma = np.sqrt(0.5 * np.median(nonzero_dists) + eps)
                else:
                    sigma = 1.0
            
            K = np.exp(-dists / (2 * sigma ** 2 + eps))
            return K
        
        n_v = len(visual_acts)
        K_cross = rbf_cross_kernel(text_acts, visual_acts)
        
        # 中心化（按视觉维度）
        H_v = np.eye(n_v) - np.ones((n_v, n_v)) / n_v
        K_cross_centered = K_cross @ H_v
        
        # 每个文本token与视觉整体的依赖性
        mi_scores = np.sum(K_cross_centered ** 2, axis=1)
        
    elif method == 'variance':
        # ---- 简单方法：激活的方差/范数 ----
        mi_scores = np.linalg.norm(text_acts, axis=1)
    
    else:
        # 默认使用self_hsic
        return compute_text_token_mi_scores(text_acts, visual_acts, method='self_hsic')
    
    # 归一化到 [0, 1]
    if np.max(mi_scores) > np.min(mi_scores):
        mi_scores = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + eps)
    
    return mi_scores

def extract_text_activations_with_mi(
    model,
    processor,
    dataset: Dataset,
    layer_idx: int,
    component: str = 'act',
    model_type: str = 'qwen',
    batch_size: int = 4,
    top_k_ratio: float = 0.5,  # 选择前50%的重要文本token
    method: str = 'self_hsic',  # 'self_hsic', 'cross_hsic', 'variance'
    verbose: bool = True
):
    """
    提取文本激活，只保留重要的文本token
    
    Args:
        top_k_ratio: float, 选择前k%的重要文本token (0-1)
        method: str, token重要性计算方法
    """
    activations = []
    current_input_ids = None
    
    def hook_fn(module, input, output):
        nonlocal current_input_ids
        
        if component == "act":
            out_seq = output[0]  # [batch, seq_len, hidden]
        else:
            out_seq = output
        
        batch_b = out_seq.shape[0]
        
        for b_idx in range(batch_b):
            seq_act = out_seq[b_idx].detach().cpu().numpy()  # [seq_len, hidden_dim]
            input_ids = current_input_ids[b_idx]
            
            # 找到视觉和文本的边界
            image_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            image_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            
            image_start_indices = (input_ids == image_start_token_id).nonzero(as_tuple=True)[0]
            image_end_indices = (input_ids == image_end_token_id).nonzero(as_tuple=True)[0]
            
            if len(image_start_indices) == 0 or len(image_end_indices) == 0:
                # 没有视觉token，使用全部序列
                text_acts = seq_act
                visual_acts = None
            else:
                image_start_pos = image_start_indices[-1].item()
                image_end_pos = image_end_indices[-1].item()
                
                visual_start = image_start_pos
                visual_end = image_end_pos
                text_start = image_end_pos + 1
                text_end = len(seq_act)
                
                visual_acts = seq_act[visual_start:visual_end]
                text_acts = seq_act[text_start:text_end]
            
            if len(text_acts) == 0:
                # 如果没有文本token，使用全序列平均
                activations.append(np.mean(seq_act, axis=0))
                continue
            
            # 计算文本token重要性分数
            try:
                if method == 'cross_hsic' and visual_acts is not None and len(visual_acts) > 0:
                    mi_scores = compute_text_token_mi_scores(
                        text_acts, visual_acts, method='cross_hsic'
                    )
                else:
                    mi_scores = compute_text_token_mi_scores(
                        text_acts, method='self_hsic'
                    )
            except Exception as e:
                if verbose:
                    print(f"MI computation error: {e}")
                mi_scores = np.ones(len(text_acts))
            
            # 选择top-k重要的文本token
            k_actual = max(1, int(len(mi_scores) * top_k_ratio))
            top_k_indices = np.argsort(mi_scores)[-k_actual:][::-1]
            
            # 过滤掉分数太低的token
            top_k_indices = top_k_indices[mi_scores[top_k_indices] > 1e-6]
            
            if len(top_k_indices) > 0:
                selected_acts = text_acts[top_k_indices]
                avg_act = np.mean(selected_acts, axis=0)
                activations.append(avg_act)
            else:
                # 如果没有选中任何token，使用全部文本token的平均
                activations.append(np.mean(text_acts, axis=0))
    
    # 获取目标模块
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
        num_workers=0
    )
    device = next(model.parameters()).device
    
    if verbose:
        print(f"Extracting TEXT activations from layer {layer_idx}, component: {component}")
        print(f"Method: {method}, Top-K ratio: {top_k_ratio}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Layer {layer_idx} (Text)", disable=not verbose):
            texts = batch['text']
            images = batch['image']
            
            # 构建messages
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
            
            current_input_ids = inputs['input_ids']
            _ = model(**inputs)
    
    hook.remove()
    
    activations = np.stack(activations) if len(activations) > 0 else np.zeros((0, model.config.hidden_size))
    
    if verbose:
        print(f"Extracted text activations shape: {activations.shape}")
    
    return activations



if __name__ == "__main__":
    # 激活保存路径
    save_path = "./activations/qwen_text_mi"
    os.makedirs(save_path, exist_ok=True)
    
    # 模型名称
    model_name = "/root/fujinhu/pretrain-models/Qwen2.5-VL-7B-Instruct"
    
    # 加载模型
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="npu:4",
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

    benign_dataset1 = AdvBenchDataset(
        bench_root="/root/fujinhu/Jail-MLLM/dataset/advbench",
        image_root="/root/fujinhu/Jail-MLLM/dataset/advimage",
        processor=processor,
        benign_mode=True  # 输出拒绝文本
    )
    
    benign_dataset2 = JailBreakBenchSteeringDataset(
        csv_path="/root/fujinhu/data/JailBreakV_28K/JailBreakV_28K.csv",
        image_root="/root/fujinhu/data/JailBreakV_28K",
        processor=processor,
        benign_mode=True,
        num_pairs=100
    )
    
    benign_dataset3 = FigStepDataset(
        csv_path="/root/fujinhu/FigStep/data/question/safebench.csv",
        image_root="/root/fujinhu/FigStep/data/images/SafeBench",
        processor=processor,
        benign_mode=True,
        num_pairs=100
    )

    benign_dataset = ConcatDataset([
        benign_dataset1,
        benign_dataset2,
        benign_dataset3
    ])


    # 提取激活
    activations_benign = {}
    activations_malicious = {}
    
    # 关键参数
    top_k_ratio = 0.2  # 选择前50%的重要文本token
    method = 'self_hsic'  # 使用自相关性方法
    
    target_layers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    
    for layer_idx in target_layers:
        print(f"\n{'='*80}")
        print(f"Processing Layer {layer_idx}")
        print(f"{'='*80}")
        
        # 提取恶意激活
        print("\nExtracting MALICIOUS activations...")
        malicious_acts = extract_text_activations_with_mi(
            model, processor, malicious_dataset,
            layer_idx=layer_idx,
            component="act",
            model_type="qwen",
            batch_size=2,
            top_k_ratio=top_k_ratio,
            method=method
        )

        # 提取良性激活
        print("\nExtracting BENIGN activations...")
        benign_acts = extract_text_activations_with_mi(
            model, processor, benign_dataset,
            layer_idx=layer_idx,
            component="act",
            model_type="qwen",
            batch_size=2,
            top_k_ratio=top_k_ratio,
            method=method
        )
        activations_benign[layer_idx] = benign_acts

        activations_malicious[layer_idx] = malicious_acts
    
    # 保存激活
    save_file_benign = os.path.join(save_path, f"text_benign_top{int(top_k_ratio*100)}pct_{method}.pt")
    save_file_malicious = os.path.join(save_path, f"text_malicious_top{int(top_k_ratio*100)}pct_{method}.pt")

    torch.save(activations_benign, save_file_benign)
    torch.save(activations_malicious, save_file_malicious)

    print("\n" + "="*80)
    print("Activation extraction complete!")
    print(f"Saved to: {save_path}")
    print("="*80)