import json
import os
from PIL import Image
import pandas as pd
import sys
# 引入 time 模块
import time 

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
    parser = argparse.ArgumentParser(description="MM-Vet Evaluation with Safety Projector")
    
    parser.add_argument("--alpha_visual", type=float, default=4.5, help="Projection strength for visual modality")
    parser.add_argument("--alpha_text", type=float, default=4.5, help="Projection strength for text modality")
    parser.add_argument('--eval', type=str, choices=['test', 'val'], default='val')
    parser.add_argument('--target_layers', type=int, nargs='+', default=[12, 13, 14], help="Target layers for steering")
    parser.add_argument('--n_components_visual', type=int, default=64, help="Number of PCA components for visual")
    parser.add_argument('--n_components_text', type=int, default=64, help="Number of PCA components for text")
    parser.add_argument('--fusion_mode', type=str, choices=['sequential', 'parallel', 'adaptive'], 
                        default='adaptive', help="Fusion mode for hybrid projector")
    parser.add_argument('--visual_text_ratio', type=float, default=0.5, 
                        help="Ratio for parallel fusion mode (only used in parallel mode)")
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[5], help="GPU device IDs to use")
    parser.add_argument('--image_folder', type=str, default='./datasets/mm-vet/images', 
                        help="Path to MM-Vet images")
    
    args = parser.parse_args()
    return args


def init_worker(gpu_id, model_path):
    """Initialize model for each worker (CUDA version)"""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id) 
    model = HookedLVLM(model_path, device=device, quantize=False)
    processor = model.processor
    return model, processor


def worker_evaluate_mmvet(args):
    """
    Worker function for MM-Vet evaluation with projector
    """
    projector, data_shard, gpu_id, result_queue, shard_id, model_path, steer_enabled, image_folder = args
    
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
            question = sample['question']
            answer = sample['answer']
            imagename = sample['imagename']
            
            # Load image
            image_path = os.path.join(image_folder, imagename)
            if not os.path.exists(image_path):
                # print(f"[GPU {gpu_id}] Warning: Image not found: {image_path}")
                image = Image.new('RGB', (224, 224), color='gray')
            else:
                image = Image.open(image_path).convert("RGB")
            
            # Create prompt
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
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
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.9,
                )
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)
            
            if "assistant\n" in generated_text.lower():
                generated_text = generated_text.split("assistant\n")[-1].strip()
            
            output_texts.append({
                'key': key,
                'question': question,
                'generated': generated_text,
                'answer': answer
            })
            
            # Clean up
            del image, messages, prompt, inputs, outputs
            torch.cuda.empty_cache() 
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error evaluating sample {sample.get('key', 'unknown')}: {e}")
            output_texts.append({
                'key': sample.get('key', ''),
                'question': sample.get('question', ''),
                'generated': '',
                'answer': sample.get('answer', '')
            })
    
    # Remove hooks
    if steer_enabled:
        for h in hooks:
            h.remove()
    
    result_queue.put(output_texts)
    
    # Clean up model
    del model, processor
    torch.cuda.empty_cache() 


class MMVetEvaluatorMP:
    def __init__(self, gpu_ids, model_path):
        self.gpu_ids = gpu_ids
        self.model_path = model_path
    
    def evaluate_mmvet(self, dataset, projector=None, steer_enabled=True, image_folder='./datasets/mm-vet/images'):
        num_gpus = len(self.gpu_ids)
        shards = [[] for _ in range(num_gpus)]
        
        for i, sample in enumerate(dataset):
            shards[i % num_gpus].append(sample)
        
        result_queue = mp.Queue()
        processes = []
        
        for rank, gpu_id in enumerate(self.gpu_ids):
            # 将 npu 相关的变量名替换为 gpu 相关
            args = (projector, shards[rank], gpu_id, result_queue, rank, self.model_path, steer_enabled, image_folder)
            p = mp.Process(target=worker_evaluate_mmvet, args=(args,))
            p.start()
            processes.append(p)
        
        all_results = []
        for _ in range(num_gpus):
            all_results.extend(result_queue.get())
        
        for p in processes:
            p.join()
        
        return all_results

# ... [save_results_for_mmvet_evaluation 和 save_detailed_results 函数保持不变] ...
def save_results_for_mmvet_evaluation(results, output_file):
    mmvet_format = {}
    for result in results:
        mmvet_format[result['key']] = result['generated']
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mmvet_format, f, ensure_ascii=False, indent=2)
    print(f"Saved MM-Vet format results to: {output_file}")


def save_detailed_results(results, output_file):
    df_data = []
    for result in results:
        df_data.append({
            'Question ID': result['key'],
            'Question': result['question'],
            'Generated Answer': result['generated'],
            'Ground Truth': result['answer']
        })
    df = pd.DataFrame(df_data)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Saved detailed results to: {output_file}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    args = parse_args()
    print("Experimental Args ===", args)
    
    # [数据加载]
    print(f"\n[+] Loading MM-Vet {args.eval} dataset...")
    questions_file = f"./datasets/mm-vet/{args.eval}_mmvet.json"
    if not os.path.exists(questions_file):
        print(f"Error: Questions file not found: {questions_file}")
        sys.exit(1)
    
    questions = json.load(open(questions_file))
    dataset = []
    for key, line in questions.items():
        dataset.append({
            'key': key, 'question': line['question'],
            'answer': line['answer'], 'imagename': line['imagename']
        })
    
    # 记录数据集大小，用于计算平均时间
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size} samples")

    model_path = "/home/fujinhu/pretrain-models/Qwen2.5-VL-7B-Instruct"
    evaluator = MMVetEvaluatorMP(gpu_ids=args.gpu_ids, model_path=model_path)
    
    output_dir = "./results/utility/qwen/mmvet"
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================
    # 1. Baseline Evaluation (No Steering)
    # ==========================================
    print("\n" + "="*50)
    print("[+] Baseline Evaluation (No Steering)")
    print("="*50)
    
    # >>> 计时开始 <<<
    start_time_baseline = time.time()
    
    baseline_results = evaluator.evaluate_mmvet(
        dataset, projector=None, steer_enabled=False, image_folder=args.image_folder
    )
    
    # >>> 计时结束 <<<
    end_time_baseline = time.time()
    duration_baseline = end_time_baseline - start_time_baseline
    
    baseline_mmvet_file = os.path.join(output_dir, f"qwen_baseline_{args.eval}_mmvet_format.json")
    save_results_for_mmvet_evaluation(baseline_results, baseline_mmvet_file)
    
    
    # [Projector 加载逻辑保持不变]
    print("\n[+] Loading activations...")
    benign_acts_text = torch.load("./activations/qwen_text_mi/text_benign_top20pct_self_hsic.pt", map_location='cpu', weights_only=False)
    malicious_acts_text = torch.load("./activations/qwen_text_mi/text_malicious_top20pct_self_hsic.pt", map_location='cpu', weights_only=False)
    benign_acts_img = torch.load("./activations/qwen_visual_mi/img_benign_top10.pt", map_location='cpu', weights_only=False)
    malicious_acts_img = torch.load("./activations/qwen_visual_mi/img_malicious_top10.pt", map_location='cpu', weights_only=False)
    
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
    
    # ==========================================
    # 2. Steered Evaluation (With Projection)
    # ==========================================
    print("\n" + "="*50)
    print(f"[+] Steered Evaluation (Layers {args.target_layers}, Alpha_V={args.alpha_visual}, Alpha_T={args.alpha_text})")
    print("="*50)
    
    # >>> 计时开始 <<<
    start_time_steered = time.time()
    
    steered_results = evaluator.evaluate_mmvet(
        dataset, projector=projector, steer_enabled=True, image_folder=args.image_folder
    )
    
    # >>> 计时结束 <<<
    end_time_steered = time.time()
    duration_steered = end_time_steered - start_time_steered
    
    steered_mmvet_file = os.path.join(output_dir, f"qwen_steered_{args.eval}_layers_{args.alpha_visual}_t{args.alpha_text}.json")
    save_results_for_mmvet_evaluation(steered_results, steered_mmvet_file)
    
    # ==========================================
    # 3. 运行时间分析报告 (For Rebuttal)
    # ==========================================
    print("\n" + "#"*60)
    print("RUNTIME EFFICIENCY ANALYSIS REPORT")
    print("#"*60)
    
    avg_time_baseline = (duration_baseline / dataset_size) * 1000 # 转毫秒
    avg_time_steered = (duration_steered / dataset_size) * 1000   # 转毫秒
    overhead_per_sample = avg_time_steered - avg_time_baseline
    overhead_percent = (overhead_per_sample / avg_time_baseline) * 100
    
    print(f"Dataset Size: {dataset_size} samples")
    print("-" * 40)
    print(f"Total Time (Baseline): {duration_baseline:.2f} s")
    print(f"Total Time (Steered) : {duration_steered:.2f} s")
    print("-" * 40)
    print(f"Avg Latency (Baseline): {avg_time_baseline:.2f} ms/sample")
    print(f"Avg Latency (Steered) : {avg_time_steered:.2f} ms/sample")
    print("-" * 40)
    print(f"Added Latency (Overhead): {overhead_per_sample:.2f} ms/sample")
    print(f"Overhead Percentage     : {overhead_percent:.2f}%")
    print("#"*60)
    print("\nEVALUATION COMPLETED ON CUDA")