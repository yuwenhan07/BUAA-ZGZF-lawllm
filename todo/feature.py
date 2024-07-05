import re

def preprocess_text(text):
    # 去除特殊字符和标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 转换为小写
    text = text.lower()
    # 去除多余的空白
    text = ' '.join(text.split())
    return text

# 示例文书数据
documents = [
    "这是第一份法律裁判文书。",
    "这是第二份法律裁判文书，包含一些标点符号！"
]

# 对所有文书进行预处理
cleaned_documents = [preprocess_text(doc) for doc in documents]
print(cleaned_documents)



import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

# 设置可用的GPU设备，例如0-9号GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 本地模型目录
local_model_dir = "/home/yuwenhan/model/GLM-4/finetune/ZhipuAI/glm-4-9b-chat"

# 从本地目录加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)

# 从本地目录加载模型并包装为 DataParallel
model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)
model = torch.nn.DataParallel(model).eval()

def extract_features(text):
    # 准备输入数据
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                           add_generation_prompt=False,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True)
    
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model.module(**inputs)
        # 提取logits作为特征
        logits = outputs.logits
        # 将logits转换为Float32类型
        logits = logits.to(torch.float32)
        # 取每个句子特征的平均值
        feature_vector = logits.mean(dim=1).squeeze().cpu().numpy()
    return feature_vector

# 对所有文书进行特征提取
features = [extract_features(doc) for doc in cleaned_documents]
for i, feature in enumerate(features):
    print(f"Document {i+1} Features: {feature[:5]}")
