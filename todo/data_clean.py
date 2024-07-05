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
