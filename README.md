# 📚 CO₂电催化还原论文分析工具

基于大模型API的论文分析脚本，自动提取CO₂RR相关论文中的C₂产物反应路径信息。

## 📋 功能特点

- 📄 **PDF解析**: 自动提取论文文本内容
- 🤖 **大模型分析**: 使用One-Shot推理提取结构化数据
- 📊 **JSON输出**: 包含中英文双版本的结果
- 🔄 **批量处理**: 支持批量分析文件夹中的PDF文件
- ⏭️ **断点续传**: 自动跳过已分析的文件

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

**方式A: 环境变量（推荐）**
```bash
# 妙搭API
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_API_BASE="https://api.miaoda.net/v1"
export OPENAI_MODEL="gpt-4o-mini"

# OpenAI API
export OPENAI_API_KEY="sk-xxx"
export OPENAI_API_BASE="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o"
```

**方式B: 命令行参数**
```bash
python paper_analyzer.py --api-key "your-key" ...
```

### 3. 运行分析

```bash
# 分析单个PDF
python paper_analyzer.py --input paper.pdf --output ./results/

# 分析文件夹
python paper_analyzer.py --input ./papers/ --output ./results/

# 强制覆盖已有结果
python paper_analyzer.py --input ./papers/ --output ./results/ --force
```

## 📁 输出格式

```json
{
  "analysis_version": "1.0",
  "paper_info": {
    "title": "论文标题",
    "authors": "作者列表",
    "year": 2024,
    "journal": "期刊名称"
  },
  "reaction_steps": [
    {
      "step_number": 1,
      "step_name_zh": "CO₂活化吸附",
      "step_name_en": "CO₂ Activation and Adsorption",
      "reaction_equation": "CO₂(g) + * + H⁺ + e⁻ → *COOH",
      "site_structure": "Cu(100) 表面",
      "coordination_number": "未明确给出",
      "reaction_energy": "未明确给出",
      "activation_barrier": "未明确给出",
      "adsorption_energy": "未明确给出",
      "key_intermediates": ["*COOH"],
      "notes": "C-C耦合前的*CO生成是RDS"
    },
    // ... 更多步骤
  ],
  "key_findings": {
    "main_descriptor": "氧结合能 (Oxygen Binding Strength)",
    "optimal_catalyst": "Rh-Cu",
    "ethylene_fe": "61.2%",
    "ethanol_fe": "13.6%",
    "selectivity_ratio": "4.51"
  },
  "methodology": {
    "dft_details": "VASP 5.4.4, PBE functional...",
    "experimental_conditions": "1 M KOH, flow cell..."
  },
  "raw_summary": "论文主要内容摘要..."
}
```

## ⚙️ 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input`, `-i` | 输入PDF文件或文件夹 | 必需 |
| `--output`, `-o` | 输出目录 | `./results/` |
| `--api-key` | API密钥 | 环境变量 |
| `--api-base` | API地址 | `https://api.miaoda.net/v1` |
| `--model` | 模型名称 | `gpt-4o-mini` |
| `--force`, `-f` | 强制覆盖已有结果 | False |
| `--delay` | 请求间隔（秒） | 1.0 |

## 🔧 常见问题

**Q: API调用失败怎么办？**
A: 检查API密钥是否正确，网络是否通畅，确认API配额是否充足。

**Q: PDF提取文本为空？**
A: 部分扫描版PDF需要OCR处理，可以先手动转换为文本或使用其他工具预处理。

**Q: 输出JSON格式错误？**
A: 可能是模型输出格式不规范，脚本会自动保存原始响应便于调试。

## 📝 License

MIT
