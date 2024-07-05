import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

device = "cuda"

# 设置可用的GPU设备，例如0-3号GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
model_parallel = torch.nn.DataParallel(model).eval()

def extract_features(text):
    # 准备输入数据
    textall = ("请你对以下文本进行实体识别，包含案件信息、当事人、法院、日期等。"
               "用以下形式给出：[案件名称:xxxx, 当事人:xxx , 法院:xxxx, 日期:xxxx]\n\n" + text)
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": textall}],
                                           add_generation_prompt=False,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True)
    
    inputs = inputs.to(device)
    
    # 生成模型输出
    with torch.no_grad():
        outputs = model_parallel.module.generate(**inputs, max_length=1000, num_return_sequences=1)
    
    # 解码生成的输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("Generated Text:", generated_text)  # 打印生成的文本
    
    return generated_text

def extract_named_entities(generated_text):
    # 使用正则表达式提取所需的实体信息
    entities = {
        "案件名称": re.search(r"案件名称[:：](.*?)\]", generated_text),
        "当事人": re.search(r"当事人[:：](.*?)\]", generated_text),
        "法院": re.search(r"法院[:：](.*?)\]", generated_text),
        "日期": re.search(r"日期[:：](.*?)\]", generated_text),
    }
    
    # 提取匹配到的文本
    extracted_entities = {key: (match.group(1).strip() if match else None) for key, match in entities.items()}
    
    return extracted_entities

def named_entity_recognition(text):
    generated_text = extract_features(text)
    entities = extract_named_entities(generated_text)
    return entities

text = '''
法院审理：
北京市朝阳区人民法院受理此案，并于2024年3月1日进行了第一次庭审。庭审中，双方主要争议焦点如下：

合同是否存在不可抗力因素

王强辩称，由于供应商张伟未能按时供货，属于不可抗力因素，导致工程无法按时完成。
李明反驳称，供应商问题是王强管理不善，并非不可抗力。
合同是否应继续履行

李明主张合同继续履行，并要求王强赔偿因延误造成的损失。
王强则认为，双方已无法继续合作，合同应解除，并要求李明赔偿损失。
赔偿金额的认定

李明提供了因工程延误导致的商业租赁损失证明，要求赔偿100万元。
王强则提供了因工程停工造成的人工和设备闲置费用，要求赔偿200万元。
法院判决：
2024年5月1日，北京市朝阳区人民法院作出判决：

合同继续履行

法院认为，供应商问题不构成不可抗力，王强应继续履行合同，按期完成工程。
赔偿金额

法院认定，李明提供的损失证明有效，王强需赔偿李明因延误造成的损失人民币50万元。
对于王强的反诉，法院认为其未能充分证明因李明拒绝增加预算而导致的实际损失，因此不予支持。
结案：
双方均表示接受法院判决，案件至此结案。王强在法院监督下，于2024年10月1日按期完成了工程建设。

法律分析：
本案涉及合同法中关于合同履行、不可抗力、违约责任等多项法律规定。法院在判决中，重点考虑了双方的合同履行能力、合同条款的明确性以及实际损失的证据支持情况，最终作出了公平合理的判决。

'''
# 进行命名实体识别
generate = extract_features(text)
print(generate)
