from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
from pathlib import Path
import json
import torch

BASE_PATH = Path("E:/ML/RAG/")#基础文件夹位置
QWEN_MODEL_PATH = BASE_PATH  / "models" / "Qwen3-VL-4B-Instruct"

base_path = Path(BASE_PATH / "data")
Folder_path = [d for d in base_path.glob("*") if d.is_dir()]

RAG_IMAGE_DESCRIPTION_PROMPT = """
你是一个专业的文档解析助手，正在为学术论文的多模态知识库（RAG）生成图片摘要。请严格按照以下规则描述图片：

1. **图片类型**：指出这是示意图、数据图、照片、表格还是其他。
2. **结合图例分析（重点）**：根据图片描述{img_caption}，详细描述图片的主要内容、数据、趋势等。
3. **关键信息提取**：
    - 文字：所有可见的说明、注释、数字、术语。
    - 数据：趋势、极值、百分比、对比关系等。
    - 实体：专业名词、缩写、产品名等。
4. **空间关系**：元素之间的位置、流程方向、连接方式。
5. **上下文线索**：窗口标题、状态栏、页码等。
6. **无内容处理**：若图片模糊、纯色，只输出"无可识别内容"。

要求：使用中文，详细但避免臆测，绝不编造不存在的信息。生成内容将作为检索索引，请确保信息密度高。
"""
def get_img_descriptions(json_path: str):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    new_paper_data = {}

    for paper_key, paper_info in data.items():
        paper_key = int(paper_key)
        img_paths = paper_info.get("img_paths", [])
        img_captions = paper_info.get("img_captions", [])
        img_descriptions = []
        for img_path, img_caption in zip(img_paths, img_captions):
            message = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(Folder_path[paper_key] / img_path),
                    },
                    {"type": "text", "text": RAG_IMAGE_DESCRIPTION_PROMPT.format(img_caption=img_caption)},
                ],
            }]
            inputs = processor.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            img_descriptions.append(output_text[0])

        new_paper_data[paper_key] = {
            "paper_name": paper_info["paper_name"],
            "img_paths": paper_info["img_paths"],
            "img_captions": paper_info["img_captions"],
            "img_descriptions": img_descriptions,
        }
        print(f"处理完成第{paper_key}篇论文")
        torch.cuda.empty_cache()

    with open("all_papers_data_vision.json", "w+", encoding="utf-8") as f:
        json.dump(new_paper_data, f, ensure_ascii=False, indent=4)





