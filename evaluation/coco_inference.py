import os
import sys
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.HookedLVLM_qwen import HookedLVLM  # 确保该模块可import

def setup_device():
    """Setup computing device."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # 1. 模型与设备初始化
    device = setup_device()
    model = HookedLVLM(
        model_id="/home/fujinhu/pretrain-models/Qwen2.5-VL-7B-Instruct",
        device=device
    )
    processor = model.processor

    # 2. 数据路径与输出路径
    coco_dir = "/home/fujinhu/data/val2017"
    output_json = "coco_val2017_infer_results.json"

    # 3. 获取前160张图片
    image_files = sorted([
        os.path.join(coco_dir, f)
        for f in os.listdir(coco_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])[:500]

    print(f"Found {len(image_files)} images, running inference on first 160...")

    results = []

    # 4. 批量推理
    for img_path in tqdm(image_files, desc="Running COCO inference"):
        try:
            image = Image.open(img_path).convert("RGB")

            # 统一prompt
            prompt = "Please describe the image."

            # 构建输入模板
            inference_prompt = processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                add_generation_prompt=True
            )

            # 构造输入
            inputs = processor(
                text=inference_prompt,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)

            # 模型生成
            with torch.no_grad():
                outputs = model.model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False
                )

            # 解码生成文本
            generated_text = processor.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            results.append({
                "image_path": img_path,
                "description": generated_text
            })

        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")

    # 5. 保存结果
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Inference completed. Results saved to: {output_json}")

if __name__ == "__main__":
    main()