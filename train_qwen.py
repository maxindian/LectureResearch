"""
CO₂电催化还原论文问答模型训练脚本
===================================
功能：使用Qwen3B模型微调训练，语料来自PDF解析的论文数据
模型：Qwen2-0.5B-Instruct 或 Qwen2-1.5B-Instruct（可根据显存选择）

使用方法：
    python train_qwen.py --data ./training_data.jsonl --output ./output
    
依赖安装：
    pip install transformers datasets peft accelerate deepspeed
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import load_dataset


# ============================================================================
# 配置区域
# ============================================================================

# 模型配置
DEFAULT_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"  # 3B: "Qwen/Qwen2-3B-Instruct"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 3
DEFAULT_SEED = 42

# 训练配置
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_LOG_STEPS = 10
DEFAULT_SAVE_STEPS = 100


# ============================================================================
# 提示模板
# ============================================================================

SYSTEM_PROMPT = """你是一位专业的电化学催化领域研究员，专注于CO₂电催化还原(CO₂RR)反应路径分析。
你需要根据给定的论文信息，准确回答关于C₂产物反应机理、催化剂选择性和能量数据的问题。"""

TRAINING_TEMPLATE = """## 任务说明

你是一个专业的CO₂电催化还原论文分析助手。你的任务是根据预训练时学到的知识，准确回答关于C₂产物反应路径的问题。

## 训练数据格式

每条训练数据包含：
- **input**: 问题（用户提问）
- **output**: 回答（详细的结构化分析）

## 训练样本示例

### 示例1：关于反应步骤的问题

**input**: "请分析CO₂电催化还原中C-C耦合步骤的反应机理，包括速率决定步骤(RDS)和相关催化剂的影响"

**output**: '''
CO₂电催化还原中的C-C耦合是生成C₂产物的关键步骤。

**1. 反应机理**
C-C耦合发生在*CO中间体之间：
- 反应：2*CO → *COCOH
- 这一步被认为是C₂+产物生成的速率决定步骤(RDS)

**2. 催化剂影响**
- 在Cu(100)表面上，C-C耦合是主要瓶颈
- 单原子掺杂（M-Cu，M=Rh, Ir, Ru等）可以增强*CO吸附
- Rh-Cu催化剂展现出最优的C-C耦合动力学

**3. 选择性控制**
- 氧结合能是决定C₂产物选择性的关键描述符
- 强氧亲和力催化剂促进乙烯生成
- 弱氧亲和力催化剂促进乙醇生成

**4. 关键数据**
- *CH₂CHO吸附能（eV）：Ru-Cu(-0.71) > Rh-Cu(-0.57) ≈ Ir-Cu(-0.57) > Cu(0)
'''

### 示例2：关于催化剂选择性的问题

**input**: "为什么Rh-Cu催化剂在乙烯选择性上表现最优？请从氧结合能的角度解释"

**output**: '''
Rh-Cu催化剂在乙烯选择性上表现最优，主要原因在于其具有适中的氧结合能。

**1. 氧结合能与C-O键断裂**
- 强氧亲和力催化剂（如Ru-Cu）虽然能有效断裂C-O键，但*O质子化能垒过高（0.50 eV），导致*O难以脱附
- 弱氧亲和力催化剂（如Au-Cu）C-O键断裂能垒过高（>0.6 eV），不利于乙烯生成

**2. Rh-Cu的最优平衡**
Rh dopant具有恰到好处的氧亲和力：
- C-O键断裂能垒：0.41 eV（最优）
- *O质子化能垒：0.44 eV（适中）

**3. 实验验证**
- Rh-Cu在-0.66 V vs RHE下达：
  - C₂H₄法拉第效率：61.2%
  - C₂H₄/C₂H₅OH选择性比：4.51
  - 偏电流密度：653 mA/cm²

**4. 火山形关系**
催化剂性能与氧结合能呈火山形关系，Rh恰好位于火山口位置。
'''

### 示例3：关于反应路径数据的问题

**input**: "请详细说明*CH₂CHO作为选择性决定中间体(SDI)的作用，以及乙烯和乙醇路径的分叉机制"

**output**: '''
*CH₂CHO是CO₂电催化还原中C₂产物选择性的关键分叉点。

**1. SDI的定义**
*CH₂CHO是通过四次质子-电子转移(PCET)从*COCOH生成的：
- *COCOH → [中间体] → [中间体] → [中间体] → *CH₂CHO

**2. 分叉机制**
*CH₂CHO面临两条竞争路径：

**路径A：乙烯路径（C-O键断裂）**
- 反应：*CH₂CHO + H⁺ + e⁻ → C₂H₄(g) + *O
- *O随后质子化：*O → *OH → H₂O
- 需要强氧亲和力促进C-O断裂

**路径B：乙醇路径（C-O键保留）**
- 反应：*CH₂CHO + H⁺ + e⁻ → *CH₃CHO → C₂H₅OH
- C-O键保留，需要弱氧亲和力

**3. 氧结合能决定选择性**
| 催化剂类型 | 氧亲和力 | C-O键长(Å) | 主要产物 |
|-----------|---------|-----------|---------|
| Ru-Cu     | 最强    | 1.367     | C₂H₄    |
| Rh-Cu     | 适中    | 1.355     | C₂H₄     |
| Cu        | 中等    | 1.320     | 混合     |
| Au-Cu     | 最弱    | 1.319     | C₂H₅OH   |

**4. 结论**
通过调控催化剂的氧结合能，可以精确控制*CH₂CHO的分叉方向，从而选择性地生成乙烯或乙醇。
'''


# ============================================================================
# 数据处理函数
# ============================================================================

def load_paper_data(json_path: str) -> List[Dict]:
    """加载论文解析结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def generate_qa_pairs_from_paper(paper_data: Dict) -> List[Dict]:
    """从论文数据生成问答对"""
    
    reaction_steps = paper_data.get('reaction_steps', [])
    key_findings = paper_data.get('key_findings', {})
    paper_info = paper_data.get('paper_info', {})
    
    qa_pairs = []
    
    # 问答对1：C-C耦合机理
    qa_pairs.append({
        "instruction": "请详细分析CO₂电催化还原中C-C耦合步骤的反应机理，包括速率决定步骤(RDS)和催化剂影响",
        "input": "",
        "output": generate_cc_coupling_response(reaction_steps, key_findings)
    })
    
    # 问答对2：Rh-Cu催化剂优势
    qa_pairs.append({
        "instruction": "为什么Rh-Cu催化剂在乙烯选择性上表现最优？请从氧结合能角度解释",
        "input": "",
        "output": generate_rh_cu_response(key_findings)
    })
    
    # 问答对3：SDI分叉机制
    qa_pairs.append({
        "instruction": "请说明*CH₂CHO作为选择性决定中间体(SDI)的作用，以及乙烯和乙醇路径的分叉机制",
        "input": "",
        "output": generate_sdi_response(reaction_steps)
    })
    
    # 问答对4：完整反应路径概述
    qa_pairs.append({
        "instruction": f"请概述这篇论文({paper_info.get('title', 'CO₂RR论文')})中的CO₂电催化还原C₂产物反应路径",
        "input": "",
        "output": generate_overview_response(paper_data)
    })
    
    # 问答对5：实验数据对比
    qa_pairs.append({
        "instruction": "请对比分析不同M-Cu催化剂(M=Rh, Ir, Ru, Pd, Ag, Au)的电催化性能差异",
        "input": "",
        "output": generate_comparison_response(reaction_steps, key_findings)
    })
    
    # 问答对6：DFT计算结果
    qa_pairs.append({
        "instruction": "请总结论文中DFT计算的关键结果，包括反应能垒和吸附能数据",
        "input": "",
        "output": generate_dft_response(reaction_steps)
    })
    
    return qa_pairs


def generate_cc_coupling_response(steps: List[Dict], findings: Dict) -> str:
    """生成C-C耦合相关回答"""
    
    # 找到C-C耦合步骤
    cc_step = None
    for step in steps:
        if step.get('step_number') == 3:
            cc_step = step
            break
    
    response = """CO₂电催化还原中的C-C耦合是生成C₂产物的关键步骤。

**1. 反应机理**
C-C耦合发生在*CO中间体之间：
- 反应方程式：2*CO → *COCOH
- 这一步被认为是C₂+产物生成的速率决定步骤(RDS)

**2. 反应位点**
"""
    
    if cc_step:
        response += f"- 位点结构：{cc_step.get('site_structure', 'Cu(100)表面')}\n"
        response += f"- 关键中间体：{', '.join(cc_step.get('key_intermediates', ['*COCOH']))}\n"
    
    response += """
**3. 催化剂调控策略**
- 单原子掺杂（M-Cu）可增强*CO吸附
- 氧结合能是关键选择性描述符
- 强氧亲和力促进C-O键断裂（→乙烯）
- 弱氧亲和力保留C-O键（→乙醇）

**4. 速率决定步骤的重要性**
C-C耦合的动力学直接影响C₂产物的总体生成速率，是提升CO₂RR效率的关键调控点。
"""
    
    return response


def generate_rh_cu_response(findings: Dict) -> str:
    """生成Rh-Cu催化剂优势回答"""
    
    optimal = findings.get('optimal_catalyst', 'Rh-Cu')
    ratio = findings.get('selectivity_ratio', '4.51')
    fe_eth = findings.get('ethylene_fe', '61.2%')
    fe_ethoh = findings.get('ethanol_fe', '13.6%')
    current = findings.get('partial_current_density', '653 mA/cm²')
    
    response = f"""Rh-Cu催化剂在乙烯选择性上表现最优，主要原因在于其具有适中的氧结合能。

**1. 氧结合能与C-O键断裂的关系**

C-O键断裂（乙烯路径）：
*CH₂CHO + H⁺ + e⁻ → C₂H₄(g) + *O

- 强氧亲和力催化剂（如Ru-Cu）：虽然能有效断裂C-O键，但*O质子化能垒过高（0.50 eV），导致*O难以脱附
- 弱氧亲和力催化剂（如Au-Cu）：C-O键断裂能垒过高（>0.6 eV），不利于乙烯生成

**2. Rh-Cu的最优平衡**

Rh dopant具有恰到好处的氧亲和力：

| 参数 | Ru-Cu | Rh-Cu | Ir-Cu | Cu |
|------|-------|-------|-------|-----|
| C-O断裂能垒(eV) | 0.61 | 0.41 | 0.42 | 0.53 |
| *O质子化能垒(eV) | 0.50 | 0.44 | 0.38 | 0.31 |

Rh-Cu在C-O键断裂和*O质子化之间取得最佳平衡。

**3. 实验验证数据**

{optimal}在-0.66 V vs RHE下达：
- C₂H₄法拉第效率：{fe_eth}
- C₂H₄/C₂H₅OH选择性比：{ratio}
- C₂H₄偏电流密度：{current}

**4. 火山形关系**

催化剂性能与氧结合能呈火山形关系：
氧亲和力顺序：Ru-Cu > Rh-Cu > Ir-Cu > Cu > Pd-Cu > Ag-Cu > Au-Cu

Rh-Cu恰好位于火山口位置，是乙烯生成的最佳选择。
"""
    
    return response


def generate_sdi_response(steps: List[Dict]) -> str:
    """生成SDI分叉机制回答"""
    
    # 找到相关步骤
    step4 = None
    step5a = None
    step5b = None
    
    for step in steps:
        if step.get('step_number') == 4:
            step4 = step
        elif step.get('step_number') == '5A':
            step5a = step
        elif step.get('step_number') == '5B':
            step5b = step
    
    response = """*CH₂CHO是CO₂电催化还原中C₂产物选择性的关键分叉点。

**1. SDI的定义与生成**

*CH₂CHO是通过四次质子-电子转移(PCET)从*COCOH生成的：
*COCOH → [中间体1] → [中间体2] → [中间体3] → *CH₂CHO

"""
    
    if step4:
        adsorp = step4.get('adsorption_energy', '')
        if adsorp:
            response += f"**2. *CH₂CHO吸附能数据**\n{adsorp}\n\n"
    
    response += """**3. 分叉机制：两条竞争路径**

*CH₂CHO面临两条竞争路径：

**路径A：乙烯路径（C-O键断裂）**
```
*CH₂CHO + H⁺ + e⁻ → C₂H₄(g) + *O
*O + H⁺ + e⁻ → *OH
*OH + H⁺ + e⁻ → H₂O(l) + *
```
- 需要强氧亲和力促进C-O断裂
- *O吸附在Cu₃M位点（oxophilic）或Cu₄位点（oxophobic）

**路径B：乙醇路径（C-O键保留）**
```
*CH₂CHO + H⁺ + e⁻ → *CH₃CHO
*CH₃CHO + 2H⁺ + 2e⁻ → C₂H₅OH(l) + *
```
- C-O键保留，需要弱氧亲和力
- *CH₃CHO通过O原子吸附（Cu-O-C结构）

**4. 氧结合能与选择性**

| 催化剂类型 | C-O键长(Å) | 键强 | 主要产物 |
|-----------|-----------|------|---------|
| Ru-Cu     | 1.367     | 最弱 | C₂H₄    |
| Rh-Cu     | 1.355     | 较弱 | C₂H₄     |
| Cu        | 1.320     | 中等 | 混合     |
| Au-Cu     | 1.319     | 最强 | C₂H₅OH   |

**5. 结论**

通过调控催化剂的氧结合能，可以精确控制*CH₂CHO的分叉方向：
- 强氧亲和力 → 促进C-O键断裂 → 乙烯
- 弱氧亲和力 → 保留C-O键 → 乙醇
"""
    
    return response


def generate_overview_response(paper_data: Dict) -> str:
    """生成论文概述回答"""
    
    info = paper_data.get('paper_info', {})
    findings = paper_data.get('key_findings', {})
    steps = paper_data.get('reaction_steps', [])
    
    title = info.get('title', '未知论文')
    journal = info.get('journal', '未知期刊')
    year = info.get('year', '未知年份')
    descriptor = findings.get('main_descriptor', '氧结合能')
    optimal = findings.get('optimal_catalyst', 'Rh-Cu')
    
    response = f"""**{title}**

**论文信息**
- 发表年份：{year}
- 期刊：{journal}

**研究背景**

本论文研究了CO₂电催化还原(CO₂RR)中C₂产物的选择性调控。研究表明，{descriptor}是决定乙烯(C₂H₄)与乙醇(C₂H₅OH)选择性的关键描述符。

**核心发现**

1. **最优催化剂**：{optimal}
2. **关键描述符**：氧结合能（Oxygen Binding Strength）
3. **反应路径**：
   - CO₂ → *COOH → *CO（CO₂活化）
   - *CO + *CO → *COCOH（C-C耦合，RDS）
   - *COCOH → *CH₂CHO（SDI生成）
   - *CH₂CHO分叉：
     - C-O键断裂 → C₂H₄ + *O（乙烯路径）
     - C-O键保留 → C₂H₅OH（乙醇路径）

**性能数据**

| 指标 | 数值 |
|------|------|
| 最优催化剂 | {optimal} |
| C₂H₄法拉第效率 | {findings.get('ethylene_fe', '61.2%')} |
| C₂H₅OH法拉第效率 | {findings.get('ethanol_fe', '13.6%')} |
| 选择性比率 | {findings.get('selectivity_ratio', '4.51')} |
| 偏电流密度 | {findings.get('partial_current_density', '653 mA/cm²')} |

**结论**

通过单原子掺杂调控Cu表面氧亲和力，可以精确控制C₂产物的选择性。{optimal}在乙烯生成方面表现最优，具有最佳的氧结合能平衡。
"""
    
    return response


def generate_comparison_response(steps: List[Dict], findings: Dict) -> str:
    """生成催化剂对比回答"""
    
    response = """**M-Cu催化剂（M=Rh, Ir, Ru, Pd, Ag, Au）电催化性能对比**

**1. 氧亲和力排序**

氧结合能（从强到弱）：
Ru-Cu > Rh-Cu > Ir-Cu > Cu > Pd-Cu > Ag-Cu > Au-Cu

**2. 催化剂分类**

| 类型 | 催化剂 | 氧亲和力 | 主要产物 | C₂H₄/C₂H₅OH |
|------|--------|---------|---------|-------------|
| Oxophilic | Ru-Cu | 最强 | C₂H₄ | ~3.1 |
| Oxophilic | Rh-Cu | 适中 | C₂H₄ | **4.51** |
| Oxophilic | Ir-Cu | 较强 | C₂H₄ | ~3.6 |
| 基准 | Cu | 中等 | 混合 | 2.33 |
| Oxophobic | Pd-Cu | 较弱 | 混合 | ~2.0 |
| Oxophobic | Ag-Cu | 弱 | C₂H₅OH | ~1.9 |
| Oxophobic | Au-Cu | 最弱 | C₂H₅OH | 1.75 |

**3. 性能数据分析**

*C-O键断裂能垒（乙烯路径）*
- Ru-Cu: 0.61 eV（过强，*O脱附受限）
- Rh-Cu: 0.41 eV（最优）
- Ir-Cu: 0.42 eV（较优）
- Cu: 0.53 eV（基准）

*O质子化能垒*
- Ru-Cu: 0.50 eV（过高）
- Rh-Cu: 0.44 eV（适中）
- Ir-Cu: 0.38 eV（较低）
- Cu: 0.31 eV（最低）

**4. 结论**

- **乙烯选择性**：Rh-Cu > Ir-Cu > Ru-Cu > Cu > Pd-Cu > Ag-Cu > Au-Cu
- **乙醇选择性**：Au-Cu > Ag-Cu > Pd-Cu > Cu > Ru-Cu > Ir-Cu > Rh-Cu
- Rh-Cu具有最优的氧亲和力平衡，是乙烯生成的最佳选择
- Ru-Cu虽然C-O键断裂能力强，但*O质子化受限，导致性能下降
"""

    return response


def generate_dft_response(steps: List[Dict]) -> str:
    """生成DFT计算结果回答"""
    
    response = """**DFT计算关键结果汇总**

**计算方法**
- 软件：VASP 5.4.4
- 泛函：PBE (Perdew-Burke-Ernzerhof)
- 方法：PAW (Projector Augmented Wave)
- 色散校正：DFT-D3
- 表面模型：Cu(100), 2×2 unit cell, 4层

**关键反应步骤数据**

"""
    
    for step in steps:
        num = step.get('step_number', '')
        name_zh = step.get('step_name_zh', '')
        reaction = step.get('reaction_equation', '')
        barrier = step.get('activation_barrier', '未明确给出')
        energy = step.get('reaction_energy', '未明确给出')
        adsorp = step.get('adsorption_energy', '未明确给出')
        
        if barrier != '未明确给出' or energy != '未明确给出' or adsorp != '未明确给出':
            response += f"""
**步骤{num}：{name_zh}**
- 反应：{reaction}
"""
            if barrier != '未明确给出':
                response += f"- 反应能垒：{barrier}\n"
            if energy != '未明确给出':
                response += f"- 反应能：{energy}\n"
            if adsorp != '未明确给出':
                response += f"- 吸附能：{adsorp}\n"
    
    response += """
**C-O键长数据（*CH₂CHO中间体）**

| 催化剂类型 | C-O键长(Å) | 解读 |
|-----------|-----------|------|
| Ru-Cu | 1.367 | 最长，键最弱 |
| Rh-Cu | 1.355 | 较长，易断裂 |
| Ir-Cu | 1.355 | 较长，易断裂 |
| Cu | 1.320 | 基准 |
| Pd-Cu | 1.319 | 较短 |
| Ag-Cu | 1.320 | 较短 |
| Au-Cu | 1.319 | 最短，键最强 |

**结论**

DFT计算揭示了氧结合能与C-O键强度的线性关系，为实验设计提供了理论指导。
"""
    
    return response


def format_for_qwen(example: Dict) -> str:
    """格式化训练样本为Qwen模型格式"""
    
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    # Qwen微调格式
    if instruction:
        text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        text += f"<|im_start|>user\n{instruction}{input_text}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{output}<|im_end|>"
    else:
        text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        text += f"<|im_start|>user\n{input_text}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{output}<|im_end|>"
    
    return text


def prepare_training_data(
    paper_json_path: str,
    output_jsonl_path: str,
    num_augment: int = 5
) -> int:
    """
    准备训练数据
    
    Args:
        paper_json_path: 论文解析结果JSON路径
        output_jsonl_path: 输出JSONL文件路径
        num_augment: 数据增强数量
    
    Returns:
        生成的样本数量
    """
    
    print(f"📖 加载论文数据: {paper_json_path}")
    paper_data = load_paper_data(paper_json_path)
    
    print("🔄 生成问答对...")
    qa_pairs = generate_qa_pairs_from_paper(paper_data)
    
    # 数据增强：复制并添加噪声变体
    augmented_pairs = []
    for pair in qa_pairs:
        augmented_pairs.append(pair)
        # 添加简化版本
        for i in range(num_augment - 1):
            simplified = {
                "instruction": pair["instruction"],
                "input": pair["input"],
                "output": pair["output"][:len(pair["output"])//2] + "..." if len(pair["output"]) > 200 else pair["output"]
            }
            augmented_pairs.append(simplified)
    
    print(f"📝 生成 {len(augmented_pairs)} 个训练样本")
    
    # 写入JSONL文件
    print(f"💾 保存到: {output_jsonl_path}")
    output_path = Path(output_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in augmented_pairs:
            formatted = format_for_qwen(pair)
            f.write(json.dumps({"text": formatted}, ensure_ascii=False) + '\n')
    
    return len(augmented_pairs)


# ============================================================================
# 训练函数
# ============================================================================

def train_model(
    data_path: str,
    output_dir: str,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    seed: int = DEFAULT_SEED,
    use_lora: bool = True
):
    """
    训练模型
    
    Args:
        data_path: 训练数据路径（JSONL格式）
        output_dir: 输出目录
        model_name: 模型名称
        max_length: 最大序列长度
        batch_size: 批大小
        learning_rate: 学习率
        num_epochs: 训练轮数
        seed: 随机种子
        use_lora: 是否使用LoRA微调
    """
    
    set_seed(seed)
    
    print("=" * 60)
    print("🚀 开始训练 Qwen 模型")
    print("=" * 60)
    print(f"📁 数据: {data_path}")
    print(f"📁 输出: {output_dir}")
    print(f"🤖 模型: {model_name}")
    print(f"📊 批大小: {batch_size}")
    print(f"📊 学习率: {learning_rate}")
    print(f"📊 训练轮数: {num_epochs}")
    print("=" * 60)
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("⚠️ 警告: 未检测到GPU，训练可能非常缓慢")
    else:
        print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载数据
    print("\n📖 加载训练数据...")
    dataset = load_dataset('json', data_files=data_path, split='train')
    print(f"   样本数: {len(dataset)}")
    
    # 加载模型和分词器
    print(f"\n🤖 加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 使用LoRA微调
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            print("\n🔧 配置LoRA微调...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "v_proj"]
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
        except ImportError:
            print("⚠️ 未安装peft库，将进行全量微调")
    
    # 数据预处理
    print("\n🔄 预处理数据...")
    
    def tokenize_function(examples):
        texts = examples["text"]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_dir=f"{output_dir}/logs",
        logging_steps=DEFAULT_LOG_STEPS,
        save_steps=DEFAULT_SAVE_STEPS,
        save_total_limit=3,
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        seed=seed,
        optim="adamw_torch"
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果语言模型
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # 开始训练
    print("\n🔥 开始训练...\n")
    trainer.train()
    
    # 保存模型
    print(f"\n💾 保存模型到: {output_dir}")
    trainer.save_model(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    
    print("\n✅ 训练完成!")
    

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(
        description="CO₂电催化还原论文问答模型训练工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 步骤1: 准备训练数据
  python train_qwen.py --prepare --paper-data ./sample_output.json --output ./training_data.jsonl
  
  # 步骤2: 训练模型（使用LoRA）
  python train_qwen.py --train --data ./training_data.jsonl --output ./output --model Qwen/Qwen2-0.5B-Instruct
  
  # 步骤3: 一键运行（准备+训练）
  python train_qwen.py --all --paper-data ./sample_output.json --output ./output

环境变量:
  CUDA_VISIBLE_DEVICES  - 指定GPU编号
        """
    )
    
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="准备训练数据模式"
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="训练模型模式"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="准备数据+训练模型"
    )
    
    parser.add_argument(
        "--paper-data",
        type=str,
        default="./sample_output.json",
        help="论文解析结果JSON文件路径"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        help="训练数据文件路径（JSONL格式）"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"模型名称（默认: {DEFAULT_MODEL_NAME}）"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"批大小（默认: {DEFAULT_BATCH_SIZE}）"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f"训练轮数（默认: {DEFAULT_NUM_EPOCHS}）"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"学习率（默认: {DEFAULT_LEARNING_RATE}）"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"最大序列长度（默认: {DEFAULT_MAX_LENGTH}）"
    )
    
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="不使用LoRA（进行全量微调）"
    )
    
    args = parser.parse_args()
    
    # 准备数据模式
    if args.prepare or args.all:
        if not args.paper_data:
            print("❌ 错误: 请提供 --paper-data 参数")
            sys.exit(1)
        
        data_path = args.data or f"{args.output}/training_data.jsonl"
        num_samples = prepare_training_data(
            paper_json_path=args.paper_data,
            output_jsonl_path=data_path
        )
        print(f"✅ 准备完成，共 {num_samples} 个样本")
        
        if not args.train and not args.all:
            print(f"\n📁 训练数据已保存到: {data_path}")
            print(f"🚀 下一步运行: python train_qwen.py --train --data {data_path} --output {args.output}")
    
    # 训练模式
    if args.train or args.all:
        data_path = args.data or f"{args.output}/training_data.jsonl"
        
        if not Path(data_path).exists():
            print(f"❌ 错误: 训练数据不存在: {data_path}")
            print(f"💡 请先运行: python train_qwen.py --prepare --paper-data ./sample_output.json")
            sys.exit(1)
        
        train_model(
            data_path=data_path,
            output_dir=args.output,
            model_name=args.model,
            max_length=args.max_length,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            use_lora=not args.no_lora
        )
    
    # 默认：准备+训练
    if not args.prepare and not args.train and not args.all:
        parser.print_help()


if __name__ == "__main__":
    main()
