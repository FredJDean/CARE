import json
import os
from PIL import Image
import pandas as pd
import sys
import base64
from io import BytesIO
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.HookedLVLM_qwen import HookedLVLM
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import io
import warnings
warnings.filterwarnings("ignore")
import torch
import torch_npu
import numpy as np
import random
import math
from methods import HybridSafetyProjectorV3
import argparse


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)


def parse_args():
    parser = argparse.ArgumentParser(description="MMBench Evaluation with Safety Projector")
    
    parser.add_argument("--alpha_visual", type=float, default=12.0, help="Projection strength for visual modality")
    parser.add_argument("--alpha_text", type=float, default=12.0, help="Projection strength for text modality")
    parser.add_argument('--eval', type=str, choices=['test', 'val'], default='val')
    parser.add_argument('--target_layers', type=int, nargs='+', default=[12, 13, 14], help="Target layers for steering")
    parser.add_argument('--n_components_visual', type=int, default=96, help="Number of PCA components for visual")
    parser.add_argument('--n_components_text', type=int, default=96, help="Number of PCA components for text")
    parser.add_argument('--fusion_mode', type=str, choices=['sequential', 'parallel', 'adaptive'], 
                        default='adaptive', help="Fusion mode for hybrid projector")
    parser.add_argument('--visual_text_ratio', type=float, default=0.5, 
                        help="Ratio for parallel fusion mode (only used in parallel mode)")
    parser.add_argument('--npu_ids', type=int, nargs='+', default=[0], help="NPU device IDs to use")
    
    args = parser.parse_args()
    return args


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image))).convert("RGB")


def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def init_worker(npu_id, model_path):
    """Initialize model for each worker"""
    device = f"npu:{npu_id}"
    torch.npu.set_device(npu_id)
    model = HookedLVLM(model_path, device=device, quantize=False)
    processor = model.processor
    return model, processor


def worker_evaluate_mmbench(args):
    """
    Worker function for MMBench evaluation with projector
    """
    projector, data_shard, npu_id, result_queue, shard_id, model_path, steer_enabled = args
    
    # Initialize model
    model, processor = init_worker(npu_id, model_path)
    device = f"npu:{npu_id}"
    
    output_texts = []
    
    # Register projector hooks if steering is enabled
    if steer_enabled:
        hooks = projector.register_hooks(model.model)
    
    # Evaluate each question
    print(f"[NPU {npu_id}] Evaluating shard {shard_id}...")
    for sample in tqdm(data_shard, desc=f"NPU {npu_id} - Shard {shard_id}"):
        try:
            idx = sample['index']
            question = sample['question']
            hint = sample['hint']
            image = sample['image']
            options = sample['options']
            answer = sample['answer']
            
            # Build question text
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(['A', 'B', 'C', 'D'][:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            
            # Create prompt
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{question}. Please do not give any explanation. Answer with the option's letter from the given choices directly."}
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
                    max_new_tokens=128,
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.9,
                )
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)
            
            output_texts.append({
                'index': idx,
                'question': question,
                'generated': generated_text,
                'answer': answer
            })
            
            # Clean up
            del image, messages, prompt, inputs, outputs, generated_text
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[NPU {npu_id}] Error evaluating sample {sample.get('index', 'unknown')}: {e}")
            output_texts.append({
                'index': sample.get('index', -1),
                'question': sample.get('question', ''),
                'generated': '',
                'answer': sample.get('answer', '')
            })
    
    # Remove hooks if they were registered
    if steer_enabled:
        for h in hooks:
            h.remove()
    
    result_queue.put(output_texts)
    
    # Clean up model
    del model, processor
    torch.cuda.empty_cache()


class MMBenchEvaluatorMP:
    def __init__(self, npu_ids, model_path):
        self.npu_ids = npu_ids
        self.model_path = model_path
    
    def evaluate_mmbench(self, dataset, projector=None, steer_enabled=True):
        """Evaluate on MMBench with optional steering"""
        num_npus = len(self.npu_ids)
        shards = [[] for _ in range(num_npus)]
        
        # Distribute data across NPUs
        for i, sample in enumerate(dataset):
            shards[i % num_npus].append(sample)
        
        result_queue = mp.Queue()
        processes = []
        
        for rank, npu_id in enumerate(self.npu_ids):
            args = (projector, shards[rank], npu_id, result_queue, rank, self.model_path, steer_enabled)
            p = mp.Process(target=worker_evaluate_mmbench, args=(args,))
            p.start()
            processes.append(p)
        
        # Collect results
        all_results = []
        for _ in range(num_npus):
            all_results.extend(result_queue.get())
        
        for p in processes:
            p.join()
        
        return all_results


def calculate_accuracy(results):
    """Calculate accuracy from results"""
    correct = 0
    total = len(results)
    
    for result in results:
        generated = result['generated'].strip().upper()
        answer = result['answer'].strip().upper()
        
        # Extract first letter as prediction
        pred = ''
        for char in generated:
            if char in ['A', 'B', 'C', 'D']:
                pred = char
                break
        
        if pred == answer:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    args = parse_args()
    print("Experimental Args ===", args)
    
    # Load MMBench dataset
    print(f"\n[+] Loading MMBench {args.eval} dataset...")
    questions = pd.read_table(os.path.expanduser(f"./datasets/MMBench/{args.eval}_mmbench.tsv"))
    print(f"Loaded {len(questions)} questions.")
    
    # Prepare dataset
    all_options = ['A', 'B', 'C', 'D']
    dataset = []
    
    for index, row in questions.iterrows():
        options = get_options(row, all_options)
        image = load_image_from_base64(row['image'])
        
        dataset.append({
            'index': row['index'],
            'question': row['question'],
            'hint': row['hint'],
            'image': image,
            'options': options,
            'answer': row['answer']
        })
    
    model_path = "/root/fujinhu/pretrain-models/Qwen2.5-VL-7B-Instruct"
    evaluator = MMBenchEvaluatorMP(npu_ids=args.npu_ids, model_path=model_path)
    
    # Baseline evaluation (no steering)
    print("\n" + "="*50)
    print("[+] Baseline Evaluation (No Steering)")
    print("="*50)
    baseline_results = evaluator.evaluate_mmbench(dataset, projector=None, steer_enabled=False)
    baseline_acc, baseline_correct, baseline_total = calculate_accuracy(baseline_results)
    print(f"Baseline Accuracy: {baseline_acc:.4f} ({baseline_correct}/{baseline_total})")
    
    # Load activations for steered evaluation
    print("\n[+] Loading activations...")
    benign_acts_text = torch.load("./activations/qwen_text_mi/text_benign_top20pct_self_hsic.pt", weights_only=False)
    malicious_acts_text = torch.load("./activations/qwen_text_mi/text_malicious_top20pct_self_hsic.pt", weights_only=False)
    benign_acts_img = torch.load("./activations/qwen_visual_mi/img_benign_top10.pt", weights_only=False)
    malicious_acts_img = torch.load("./activations/qwen_visual_mi/img_malicious_top10.pt", weights_only=False)
    print("Activations loaded successfully!")
    
    # Initialize projector
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
    
    # Steered evaluation
    print("\n" + "="*50)
    print(f"[+] Steered Evaluation (Layers {args.target_layers}, Alpha_V={args.alpha_visual}, Alpha_T={args.alpha_text})")
    print("="*50)
    steered_results = evaluator.evaluate_mmbench(dataset, projector=projector, steer_enabled=True)
    steered_acc, steered_correct, steered_total = calculate_accuracy(steered_results)
    print(f"Steered Accuracy: {steered_acc:.4f} ({steered_correct}/{steered_total})")
    
    # Save results
    result_file = f"./results/utility/qwen/mmbench_hybrid_projector_{args.eval}_layers_{'_'.join(map(str, args.target_layers))}_alpha_v{int(args.alpha_visual)}_t{int(args.alpha_text)}_{args.fusion_mode}.json"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    results_summary = {
        "config": {
            "eval_set": args.eval,
            "target_layers": args.target_layers,
            "alpha_visual": args.alpha_visual,
            "alpha_text": args.alpha_text,
            "fusion_mode": args.fusion_mode,
            "visual_text_ratio": args.visual_text_ratio,
            "n_components_visual": args.n_components_visual,
            "n_components_text": args.n_components_text,
        },
        "baseline": {
            "accuracy": baseline_acc,
            "correct": baseline_correct,
            "total": baseline_total
        },
        "steered": {
            "accuracy": steered_acc,
            "correct": steered_correct,
            "total": steered_total
        },
        "performance_change": {
            "accuracy_delta": steered_acc - baseline_acc,
            "relative_change": (steered_acc - baseline_acc) / baseline_acc if baseline_acc > 0 else 0
        }
    }
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Baseline Accuracy:  {baseline_acc:.4f}")
    print(f"Steered Accuracy:   {steered_acc:.4f}")
    print(f"Accuracy Change:    {steered_acc - baseline_acc:+.4f} ({(steered_acc - baseline_acc) / baseline_acc * 100:+.2f}%)")
    print(f"\nResults saved to: {result_file}")