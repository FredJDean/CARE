import json
import os
from PIL import Image
import pandas as pd
import sys
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
    torch.cuda.manual_seed_all(seed) # 保持 CUDA 种子设置
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
    # 修改为 gpu_ids
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help="GPU device IDs to use")
    parser.add_argument('--image_folder', type=str, default='./datasets/mm-vet/images', 
                        help="Path to MM-Vet images")
    
    args = parser.parse_args()
    return args


def init_worker(gpu_id, model_path):
    """Initialize model for each worker (CUDA version)"""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id) # 修改为 cuda
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
    device = f"cuda:{gpu_id}" # 修改为 cuda
    
    output_texts = []
    
    # Register projector hooks if steering is enabled
    if steer_enabled:
        hooks = projector.register_hooks(model.model)
    
    # Evaluate each question
    print(f"[GPU {gpu_id}] Evaluating shard {shard_id}...")
    for sample in tqdm(data_shard, desc=f"GPU {gpu_id} - Shard {shard_id}"):
        try:
            key = sample['key']
            question = sample['question']
            answer = sample['answer']
            imagename = sample['imagename']
            
            # Load image
            image_path = os.path.join(image_folder, imagename)
            if not os.path.exists(image_path):
                print(f"[GPU {gpu_id}] Warning: Image not found: {image_path}")
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
            torch.cuda.empty_cache() # 修改为 cuda
            
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
    torch.cuda.empty_cache() # 修改为 cuda


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
    # CUDA 场景下通常推荐使用 spawn 启动模式以避免多卡初始化冲突
    mp.set_start_method('spawn', force=True)
    
    args = parse_args()
    print("Experimental Args ===", args)
    
    # [数据加载部分保持不变]
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
    
    # 修改模型路径为你的 CUDA 环境路径
    model_path = "/home/fujinhu/pretrain-models/Qwen2.5-VL-7B-Instruct"
    evaluator = MMVetEvaluatorMP(gpu_ids=args.gpu_ids, model_path=model_path)
    
    output_dir = "./results/utility/qwen/mmvet"
    os.makedirs(output_dir, exist_ok=True)
    
    # Baseline
    print("\n" + "="*50)
    print("[+] Baseline Evaluation (No Steering)")
    print("="*50)
    baseline_results = evaluator.evaluate_mmvet(
        dataset, projector=None, steer_enabled=False, image_folder=args.image_folder
    )
    
    # [保存 Baseline 逻辑保持不变]
    baseline_mmvet_file = os.path.join(output_dir, f"qwen_baseline_{args.eval}_mmvet_format.json")
    save_results_for_mmvet_evaluation(baseline_results, baseline_mmvet_file)
    
    # 加载激活值并放到 CPU 或 CUDA
    print("\n[+] Loading activations...")
    # 注意：如果激活值是用 NPU 保存的，torch.load 通常能自动处理，但建议确保 map_location='cpu'
    benign_acts_text = torch.load("./activations/qwen_text_mi/text_benign_top20pct_self_hsic.pt", map_location='cpu', weights_only=False)
    malicious_acts_text = torch.load("./activations/qwen_text_mi/text_malicious_top20pct_self_hsic.pt", map_location='cpu', weights_only=False)
    benign_acts_img = torch.load("./activations/qwen_visual_mi/img_benign_top10.pt", map_location='cpu', weights_only=False)
    malicious_acts_img = torch.load("./activations/qwen_visual_mi/img_malicious_top10.pt", map_location='cpu', weights_only=False)
    
    # Projector 初始化 (HybridSafetyProjectorV3 内部应支持 CUDA)
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
    
    # Steered Evaluation
    print("\n" + "="*50)
    print(f"[+] Steered Evaluation (Layers {args.target_layers}, Alpha_V={args.alpha_visual}, Alpha_T={args.alpha_text})")
    print("="*50)
    steered_results = evaluator.evaluate_mmvet(
        dataset, projector=projector, steer_enabled=True, image_folder=args.image_folder
    )
    
    # [保存结果及配置逻辑保持不变]
    steered_mmvet_file = os.path.join(output_dir, f"qwen_steered_{args.eval}_layers_{args.alpha_visual}_t{args.alpha_text}.json")
    save_results_for_mmvet_evaluation(steered_results, steered_mmvet_file)
    print("\nEVALUATION COMPLETED ON CUDA")