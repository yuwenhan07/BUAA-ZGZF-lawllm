import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def predict(messages, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("/home/yuwenhan/law-LLM/buaa&zgzf/finetune/ZhipuAI/glm-4-9b-chat/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/yuwenhan/law-LLM/buaa&zgzf/finetune/ZhipuAI/glm-4-9b-chat/", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

# Load the fine-tuned Lora model
model = PeftModel.from_pretrained(model, model_id="/home/yuwenhan/law-LLM/buaa&zgzf/finetune/output/GLM4-NER-2/checkpoint-200")

# Define the instruction
instruction = ("你是一个法律命名实体识别的专家。请根据给定文本，从以下十个方面（犯罪嫌疑人、受害人、被盗货币、物品价值、"
               "盗窃获利、被盗物品、作案工具、时间、地点、组织机构）提取文中的实体，没有用None表示，并按照以下格式返回结果："
               "[犯罪嫌疑人: xxx; 受害人： xxx; 被盗货币： None; ……]")

# Get user input
input_value = input("请输入文本内容: ")

# Prepare messages for the model
messages = [
    {"role": "system", "content": instruction},
    {"role": "user", "content": f"{input_value}"}
]

# Generate response
response = predict(messages, model, tokenizer)
print(response)

import pandas as pd
import re

# 使用正则表达式解析 response 字符串
pattern = re.compile(r'([^:]+): ([^;]+)')
matches = pattern.findall(response)

data = [{'label': match[0].strip(), 'text': match[1].strip()} for match in matches]

# 显示解析后的数据
print(data)

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 显示 DataFrame
print(df)

# 将 DataFrame 输出为表格形式
df_display = df.to_markdown(index=False)
print(df_display)