import json
import random

# 读取JSON文件
with open('命名实体识别.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 随机打乱数据
random.shuffle(data)

# 分割数据集
train_data = data[:200]
val_data = data[200:284]

# 保存训练集到train_data.json
with open('train_data.json', 'w', encoding='utf-8') as train_file:
    json.dump(train_data, train_file, ensure_ascii=False, indent=4)

# 保存验证集到val_data.json
with open('val_data.json', 'w', encoding='utf-8') as val_file:
    json.dump(val_data, val_file, ensure_ascii=False, indent=4)

print("数据集分割完成，已保存为train_data.json和val_data.json")
