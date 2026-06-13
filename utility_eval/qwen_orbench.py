import json
import os
from PIL import Image
import pandas as pd
import sys
# 确保路径包含 src 目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.HookedLVLM_qwen import HookedLVLM
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import io
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import random
from steer.methods import HybridSafetyProjectorV3
import argparse


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)


def parse_args():
    parser = argparse.ArgumentParser(description="OR-Bench Evaluation with Safety Projector")
    
    parser.add_argument("--alpha_visual", type=float, default=4.5, help="Projection strength for visual modality")
    parser.add_argument("--alpha_text", type=float, default=4.5, help="Projection strength for text modality")
    parser.add_argument('--target_layers', type=int, nargs='+', default=[12, 13, 14], help="Target layers for steering")
    parser.add_argument('--n_components_visual', type=int, default=64, help="Number of PCA components for visual")
    parser.add_argument('--n_components_text', type=int, default=64, help="Number of PCA components for text")
    parser.add_argument('--fusion_mode', type=str, choices=['sequential', 'parallel', 'adaptive'], 
                        default='adaptive', help="Fusion mode for hybrid projector")
    parser.add_argument('--visual_text_ratio', type=float, default=0.5, 
                        help="Ratio for parallel fusion mode (only used in parallel mode)")
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1,5,7], help="GPU device IDs to use")
    # OR-Bench 文件路径
    parser.add_argument('--or_bench_file', type=str, default='/home/fujinhu/code/llava-interp/datasets/or-bench-hard-1k.csv', 
                        help="Path to OR-Bench CSV file")
    
    args = parser.parse_args()
    return args


def init_worker(gpu_id, model_path):
    """Initialize model for each worker"""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    model = HookedLVLM(model_path, device=device, quantize=False)
    processor = model.processor
    return model, processor


def worker_evaluate_orbench(args):
    """
    Worker function for OR-Bench evaluation
    """
    projector, data_shard, gpu_id, result_queue, shard_id, model_path, steer_enabled = args
    
    # Initialize model
    model, processor = init_worker(gpu_id, model_path)
    device = f"cuda:{gpu_id}"
    
    output_texts = []
    
    # Register projector hooks if steering is enabled
    if steer_enabled:
        hooks = projector.register_hooks(model.model)
    
    # Evaluate each question
    # print(f"[GPU {gpu_id}] Evaluating shard {shard_id}...")
    for sample in tqdm(data_shard, desc=f"GPU {gpu_id} - Shard {shard_id}"):
        try:
            key = sample['key']
            prompt_text = sample['prompt'] # CSV中的 prompt 列
            category = sample['category']  # CSV中的 category 列
            
            # OR-Bench 是纯文本数据集，但 Qwen-VL 是多模态模型。
            # 为了测试 Steering 机制（包含视觉融合）并模拟 VLM 的输入，我们创建一个纯黑的 Dummy Image。
            # 这也是测试 "Over-Refusal" 的标准做法：给模型一张无害图片+有挑战性的文本。
            image = Image.new('RGB', (256, 256), color='black')
            
            # Create prompt
            messages = [
                {"role": "user", "content": [
                    {"type": "image"}, # 占位符
                    {"type": "text", "text": prompt_text}
                ]}
            ]
            prompt = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2, # 低温采样以获得更确定的回答
                    do_sample=True,
                    top_p=0.9,
                )
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)
            
            if "assistant\n" in generated_text.lower():
                generated_text = generated_text.split("assistant\n")[-1].strip()
            
            output_texts.append({
                'key': key,
                'prompt': prompt_text,
                'generated': generated_text,
                'category': category
            })
            
            # Clean up
            del image, messages, prompt, inputs, outputs
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error evaluating sample {sample.get('key', 'unknown')}: {e}")
            output_texts.append({
                'key': sample.get('key', ''),
                'prompt': sample.get('prompt', ''),
                'generated': "Error generating response",
                'category': sample.get('category', '')
            })
    
    # Remove hooks
    if steer_enabled:
        for h in hooks:
            h.remove()
    
    result_queue.put(output_texts)
    
    # Clean up model
    del model, processor
    torch.cuda.empty_cache()


class ORBenchEvaluatorMP:
    def __init__(self, gpu_ids, model_path):
        self.gpu_ids = gpu_ids
        self.model_path = model_path
    
    def evaluate(self, dataset, projector=None, steer_enabled=True):
        num_gpus = len(self.gpu_ids)
        shards = [[] for _ in range(num_gpus)]
        
        for i, sample in enumerate(dataset):
            shards[i % num_gpus].append(sample)
        
        result_queue = mp.Queue()
        processes = []
        
        for rank, gpu_id in enumerate(self.gpu_ids):
            # args 传递给 worker
            args = (projector, shards[rank], gpu_id, result_queue, rank, self.model_path, steer_enabled)
            p = mp.Process(target=worker_evaluate_orbench, args=(args,))
            p.start()
            processes.append(p)
        
        all_results = []
        for _ in range(num_gpus):
            all_results.extend(result_queue.get())
        
        for p in processes:
            p.join()
        
        return all_results


def save_results_json(results, output_file):
    """保存为 Key-Value 格式，方便后续 SARR 脚本读取"""
    formatted = {}
    for result in results:
        # 使用 key (index) 作为 json key，response 作为 value
        formatted[result['key']] = result['generated']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON results to: {output_file}")


def save_detailed_csv(results, output_file):
    """保存详细 CSV，包含 prompt 和 category"""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Saved detailed CSV results to: {output_file}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    args = parse_args()
    print("Experimental Args ===", args)
    
    # 1. 加载 OR-Bench 数据 (CSV)
    print(f"\n[+] Loading OR-Bench dataset from {args.or_bench_file}...")
    if not os.path.exists(args.or_bench_file):
        print(f"Error: File not found: {args.or_bench_file}")
        sys.exit(1)
    
    # 读取 CSV
    df = pd.read_csv(args.or_bench_file)
    dataset = []
    # 转换为 list of dicts
    for idx, row in df.iterrows():
        dataset.append({
            'key': str(idx), # 使用行号作为 ID
            'prompt': row['prompt'],
            'category': row['category']
        })
    
    print(f"Total samples: {len(dataset)}")

    model_path = "/home/fujinhu/pretrain-models/Qwen2.5-VL-7B-Instruct"
    evaluator = ORBenchEvaluatorMP(gpu_ids=args.gpu_ids, model_path=model_path)
    
    output_dir = "./results/utility/qwen/orbench"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Baseline Evaluation (可选，如果你只想跑 Steering 可以注释掉)
    # print("\n" + "="*50)
    # print("[+] Baseline Evaluation (No Steering)")
    # print("="*50)
    # baseline_results = evaluator.evaluate(dataset, projector=None, steer_enabled=False)
    # save_results_json(baseline_results, os.path.join(output_dir, "qwen_baseline_orbench.json"))
    
    # 3. Loading Activations
    print("\n[+] Loading activations...")
    benign_acts_text = torch.load("./activations/qwen_text_mi/text_benign_top20pct_self_hsic.pt", map_location='cpu', weights_only=False)
    malicious_acts_text = torch.load("./activations/qwen_text_mi/text_malicious_top20pct_self_hsic.pt", map_location='cpu', weights_only=False)
    benign_acts_img = torch.load("./activations/qwen_visual_mi/img_benign_top10.pt", map_location='cpu', weights_only=False)
    malicious_acts_img = torch.load("./activations/qwen_visual_mi/img_malicious_top10.pt", map_location='cpu', weights_only=False)
    
    # 4. Initialize Projector
    projector = HybridSafetyProjectorV3(
        benign_acts_img, malicious_acts_img,
        benign_acts_text, malicious_acts_text,
        target_layers=args.target_layers,
        n_components_visual=args.n_components_visual,
        n_components_text=args.n_components_text,
        projection_strength_visual=args.alpha_visual,
        projection_strength_text=args.alpha_text,
        fusion_mode=args.fusion_mode,
        visual_text_ratio=args.visual_text_ratio,
    )
    
    # 5. Steered Evaluation
    print("\n" + "="*50)
    print(f"[+] Steered Evaluation (Layers {args.target_layers}, Alpha_V={args.alpha_visual}, Alpha_T={args.alpha_text})")
    print("="*50)
    steered_results = evaluator.evaluate(dataset, projector=projector, steer_enabled=True)
    
    # 保存结果
    json_output_path = os.path.join(output_dir, f"qwen_steered_orbench_layers_{args.alpha_visual}_t{args.alpha_text}.json")
    csv_output_path = os.path.join(output_dir, f"qwen_steered_orbench_layers_{args.alpha_visual}_t{args.alpha_text}_detailed.csv")
    
    save_results_json(steered_results, json_output_path)
    save_detailed_csv(steered_results, csv_output_path)
    
    print("\nEVALUATION COMPLETED")