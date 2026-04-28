# 📚 CO₂电催化还原论文问答模型训练指南

基于Qwen2-0.5B/3B模型的微调训练脚本，使用论文解析数据作为训练语料。

## 📋 功能特点

- 🔄 自动从论文JSON数据生成问答对
- 🤖 支持Qwen2-0.5B/1.5B/3B模型
- ⚡ LoRA高效微调（节省显存）
- 📊 支持批量训练和断点续训

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_train.txt
```

### 2. 准备训练数据

```bash
# 使用样例数据准备训练数据
python train_qwen.py --prepare \
    --paper-data ./sample_output.json \
    --output ./training_data.jsonl
```

### 3. 开始训练

```bash
# 使用0.5B模型训练（约需6GB显存）
python train_qwen.py --train \
    --data ./training_data.jsonl \
    --output ./output \
    --model Qwen/Qwen2-0.5B-Instruct \
    --batch-size 4 \
    --epochs 3

# 使用1.5B模型训练（约需12GB显存）
python train_qwen.py --train \
    --data ./training_data.jsonl \
    --output ./output \
    --model Qwen/Qwen2-1.5B-Instruct \
    --batch-size 2 \
    --epochs 3

# 一键运行（准备+训练）
python train_qwen.py --all \
    --paper-data ./sample_output.json \
    --output ./output
```

### 4. 模型推理

训练完成后，使用以下代码进行推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./output/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

prompt = "为什么Rh-Cu催化剂在乙烯选择性上表现最优？"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 📁 输出文件

```
output/
├── training_data.jsonl     # 训练数据
├── checkpoint-XXX/         # 检查点
├── runs/                   # TensorBoard日志
└── final_model/            # 最终模型
    ├── config.json
    ├── model.safetensors
    └── tokenizer files
```

## ⚙️ 显存要求

| 模型 | 最低显存 | 推荐显存 | batch_size |
|------|---------|---------|-----------|
| Qwen2-0.5B | 4GB | 8GB | 4-8 |
| Qwen2-1.5B | 8GB | 16GB | 2-4 |
| Qwen2-3B | 16GB | 24GB | 1-2 |

## 🔧 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称 | Qwen/Qwen2-0.5B-Instruct |
| `--batch-size` | 批大小 | 2 |
| `--epochs` | 训练轮数 | 3 |
| `--lr` | 学习率 | 1e-4 |
| `--max-length` | 最大序列长度 | 2048 |
| `--no-lora` | 禁用LoRA全量微调 | False |

## 📝 训练数据格式

生成的训练数据为JSONL格式，每行一个样本：

```json
{"text": "<|im_start|>system\n你是一个专业助手<|im_end|>\n<|im_start|>user\n问题<|im_end|>\n<|im_start|>assistant\n回答<|im_end|>"}
```

## 💡 提示

1. **LoRA vs 全量微调**：LoRA节省显存，效果也很好，建议优先使用
2. **batch_size**：根据显存调整，显存不足可降低batch_size
3. **训练轮数**：3-5轮通常足够
4. **数据增强**：脚本会自动生成多个变体样本
