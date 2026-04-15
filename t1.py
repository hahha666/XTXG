import os
# 设置 Hugging Face 镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 然后再导入相关的库和执行代码
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("fuck you")
print(result)