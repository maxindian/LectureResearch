"""
CO₂电催化还原论文分析脚本
==============================
功能：读取PDF论文，使用大模型API提取C₂产物反应路径分析结果
输出：JSON文件（包含中英文版本）

使用方法：
    python paper_analyzer.py --input ./papers/ --output ./results/
    python paper_analyzer.py --input ./paper.pdf --output ./results/

依赖安装：
    pip install openai pymupdf python-dotenv tqdm
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# 第三方库
try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import pymupdf as fitz
    except ImportError:
        print("请安装 PyMuPDF: pip install pymupdf")
        sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("请安装 OpenAI SDK: pip install openai")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # 简化版无进度条


# ============================================================================
# 配置区域 - 请根据实际情况修改
# ============================================================================

# API 配置
DEFAULT_API_BASE = "https://api.miaoda.net/v1"  # 妙搭API
DEFAULT_MODEL = "gpt-4o-mini"  # 或使用 "gpt-4o", "gpt-4-turbo" 等

# 输入输出配置
DEFAULT_INPUT_DIR = "./papers"
DEFAULT_OUTPUT_DIR = "./results"

# 分析Prompt模板
SYSTEM_PROMPT = """你是一位专业的电化学催化领域研究员，专门分析CO₂电催化还原(CO₂RR)相关的学术论文。

你的任务是：
1. 仔细阅读给定的论文内容
2. 提取并结构化分析C₂产物（如乙烯C₂H₄、乙醇C₂H₅OH等）的反应路径
3. 按照预定义的8个反应步骤格式输出结果

重要原则：
- 只提取论文中**明确给出**的数据
- 如果论文没有提供某项数据，标注为"未明确给出"
- 保持数据的原始英文单位
- 使用标准化学术语"""

USER_PROMPT_TEMPLATE = """## 任务背景

我需要你分析一篇关于CO₂电催化还原(CO₂RR)选择性C₂产物形成的论文。请严格按照以下格式提取信息。

## 输出格式要求

请以JSON格式输出，包含以下8个反应步骤的结构化数据。每个步骤都需要中英文两个版本。

```json
{{
  "analysis_version": "1.0",
  "paper_info": {{
    "title": "论文标题",
    "authors": "作者列表",
    "year": 发表年份,
    "journal": "期刊名称",
    "doi": "DOI编号"
  }},
  "reaction_steps": [
    {{
      "step_number": 1,
      "step_name_zh": "第一步名称（中文）",
      "step_name_en": "Step 1 Name (English)",
      "reaction_equation": "反应方程式",
      "site_structure": "位点结构类型",
      "coordination_number": "配位数（如未提及则填写'未明确给出'）",
      "reaction_energy": "反应能（如未提及则填写'未明确给出'）",
      "activation_barrier": "反应能垒（如未提及则填写'未明确给出'）",
      "adsorption_energy": "中间体吸附能（如未提及则填写'未明确给出'）",
      "key_intermediates": ["关键中间体列表"],
      "notes": "补充说明"
    }},
    ... (步骤2-8的完整信息)
  ],
  "key_findings": {{
    "main_descriptor": "主要选择性描述符",
    "optimal_catalyst": "最优催化剂",
    "optimal_conditions": "最优反应条件",
    "ethylene_fe": "乙烯法拉第效率",
    "ethanol_fe": "乙醇法拉第效率",
    "selectivity_ratio": "选择性比率"
  }},
  "methodology": {{
    "dft_details": "DFT计算方法",
    "experimental_conditions": "实验条件"
  }},
  "raw_summary": "论文主要内容摘要（500字以内）"
}}
```

## 反应步骤参考格式

步骤1: CO₂活化吸附
- 反应：CO₂(g) + * + H⁺ + e⁻ → *COOH
- 位点结构：Cu表面等
- 配位数、反应能、能垒、吸附能等

步骤2: COOH还原为CO
- 反应：*COOH + H⁺ + e⁻ → *CO + H₂O

步骤3: C-C耦合（通常是RDS速率决定步骤）
- 反应：2*CO → *COCOH

步骤4: COCOH逐步质子-电子转移生成*CH₂CHO

步骤5A: 乙烯路径（C-O键断裂）
- 反应：*CH₂CHO + H⁺ + e⁻ → C₂H₄(g) + *O

步骤5B: 乙醇路径（C-O键保留）
- 反应：*CH₂CHO + H⁺ + e⁻ → *CH₃CHO → C₂H₅OH

步骤6: *O还原脱附

步骤7-8: 其他相关步骤

## 论文内容

请分析以下论文内容，提取相关信息：

---
{paper_content}
---

## 注意事项

1. 严格按照JSON格式输出，不要包含其他文字
2. 所有数据必须来自论文原文
3. 如果某项数据论文未给出，使用"未明确给出"
4. 保留所有化学式的原始格式（*表示吸附位点）
5. 确保JSON格式正确，可以被解析
"""


# ============================================================================
# 核心函数
# ============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """从PDF中提取文本内容"""
    text_content = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_content.append(f"--- 第 {page_num + 1} 页 ---\n{text}")
        
        doc.close()
        
        if not text_content:
            print(f"警告: {pdf_path} 未提取到文本内容")
            return ""
        
        return "\n\n".join(text_content)
    
    except Exception as e:
        print(f"提取PDF文本时出错: {e}")
        return ""


def extract_pdf_metadata(pdf_path: str) -> dict:
    """提取PDF元数据"""
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        doc.close()
        
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
        }
    except Exception as e:
        print(f"提取PDF元数据时出错: {e}")
        return {}


def analyze_paper_with_api(
    paper_content: str,
    paper_title: str = "",
    api_key: str = "",
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    temperature: float = 0.0
) -> dict:
    """
    调用大模型API分析论文
    
    Args:
        paper_content: 论文文本内容
        paper_title: 论文标题（用于日志）
        api_key: API密钥
        api_base: API地址
        model: 模型名称
        max_tokens: 最大生成token数
        temperature: 温度参数
    
    Returns:
        解析后的JSON结果
    """
    
    # 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url=api_base
    )
    
    # 构建用户提示
    user_prompt = USER_PROMPT_TEMPLATE.format(
        paper_content=paper_content[:15000]  # 限制输入长度，避免超出token限制
    )
    
    print(f"  📤 正在调用 {model} API...")
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        elapsed = time.time() - start_time
        print(f"  ✅ API调用成功，耗时 {elapsed:.1f}秒")
        
        # 解析响应
        result_text = response.choices[0].message.content
        
        # 尝试解析JSON
        try:
            result = json.loads(result_text)
            result["api_response"] = {
                "model": model,
                "elapsed_seconds": elapsed,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                    "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else None,
                }
            }
            return result
        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON解析失败: {e}")
            return {
                "error": "JSON解析失败",
                "raw_response": result_text,
                "api_response": {
                    "model": model,
                    "elapsed_seconds": elapsed
                }
            }
    
    except Exception as e:
        print(f"  ❌ API调用失败: {e}")
        return {
            "error": str(e),
            "paper_title": paper_title
        }


def analyze_paper(
    pdf_path: str,
    output_path: str,
    api_key: str = "",
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    force: bool = False
) -> bool:
    """
    分析单篇论文
    
    Args:
        pdf_path: PDF文件路径
        output_path: 输出JSON文件路径
        api_key: API密钥
        api_base: API地址
        model: 模型名称
        force: 是否强制覆盖已有结果
    
    Returns:
        是否成功
    """
    
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    
    # 检查文件是否存在
    if not pdf_path.exists():
        print(f"❌ PDF文件不存在: {pdf_path}")
        return False
    
    # 检查输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已存在结果
    if output_path.exists() and not force:
        print(f"⏭️  跳过（已存在）: {output_path.name}")
        return True
    
    print(f"\n📄 分析: {pdf_path.name}")
    
    # 提取PDF内容
    print(f"  📖 提取PDF文本...")
    paper_content = extract_text_from_pdf(str(pdf_path))
    
    if not paper_content.strip():
        print(f"  ⚠️ 未能提取到有效文本内容")
        # 保存错误结果
        error_result = {
            "error": "无法从PDF提取文本",
            "pdf_path": str(pdf_path),
            "paper_title": pdf_path.stem
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, ensure_ascii=False, indent=2)
        return False
    
    print(f"  📊 提取到 {len(paper_content)} 字符")
    
    # 提取元数据
    metadata = extract_pdf_metadata(str(pdf_path))
    
    # 调用API分析
    result = analyze_paper_with_api(
        paper_content=paper_content,
        paper_title=metadata.get("title", pdf_path.stem),
        api_key=api_key,
        api_base=api_base,
        model=model
    )
    
    # 添加原始信息
    result["source_file"] = str(pdf_path)
    result["pdf_metadata"] = metadata
    result["extracted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 保存结果
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  💾 结果已保存: {output_path.name}")
        return True
    except Exception as e:
        print(f"  ❌ 保存结果失败: {e}")
        return False


def batch_analyze(
    input_dir: str,
    output_dir: str,
    api_key: str = "",
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    force: bool = False,
    delay: float = 1.0
) -> dict:
    """
    批量分析文件夹中的PDF论文
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        api_key: API密钥
        api_base: API地址
        model: 模型名称
        force: 是否强制覆盖已有结果
        delay: 请求间隔（秒），避免API限流
    
    Returns:
        分析统计信息
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有PDF文件
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        pdf_files = [input_path]
    else:
        pdf_files = list(input_path.glob("**/*.pdf"))
    
    if not pdf_files:
        print(f"❌ 在 {input_dir} 中未找到PDF文件")
        return {"success": 0, "failed": 0, "skipped": 0}
    
    print(f"\n🔍 找到 {len(pdf_files)} 个PDF文件")
    print(f"📁 输出目录: {output_path}\n")
    
    # 统计
    stats = {"success": 0, "failed": 0, "skipped": 0}
    
    # 批量处理
    for i, pdf_file in enumerate(tqdm(pdf_files, desc="分析进度")):
        output_file = output_path / f"{pdf_file.stem}_analysis.json"
        
        # 检查是否跳过
        if output_file.exists() and not force:
            stats["skipped"] += 1
            continue
        
        # 分析论文
        success = analyze_paper(
            pdf_path=str(pdf_file),
            output_path=str(output_file),
            api_key=api_key,
            api_base=api_base,
            model=model,
            force=force
        )
        
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
        
        # 请求间隔
        if i < len(pdf_files) - 1:
            time.sleep(delay)
    
    return stats


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(
        description="CO₂电催化还原论文分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析单个PDF文件
  python paper_analyzer.py --input paper.pdf --output ./results/
  
  # 分析文件夹中的所有PDF
  python paper_analyzer.py --input ./papers/ --output ./results/
  
  # 使用自定义API配置
  python paper_analyzer.py --input ./papers/ --output ./results/ \\
      --api-key sk-xxx --api-base https://api.openai.com/v1 \\
      --model gpt-4o-mini
  
  # 强制重新分析（覆盖已有结果）
  python paper_analyzer.py --input ./papers/ --output ./results/ --force

环境变量:
  OPENAI_API_KEY     - API密钥
  OPENAI_API_BASE    - API地址（可选）
  OPENAI_MODEL       - 模型名称（可选）
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入PDF文件或文件夹路径"
    )
    
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）"
    )
    
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API密钥（可设置环境变量 OPENAI_API_KEY）"
    )
    
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE),
        help=f"API地址（默认: {DEFAULT_API_BASE}）"
    )
    
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"模型名称（默认: {DEFAULT_MODEL}）"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制重新分析，覆盖已有结果"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="API请求间隔（秒，默认1.0）"
    )
    
    args = parser.parse_args()
    
    # 检查API密钥
    if not args.api_key:
        print("❌ 错误: 请提供API密钥（--api-key）或设置环境变量 OPENAI_API_KEY")
        print("\n提示: 对于妙搭API，可以从控制台获取密钥")
        sys.exit(1)
    
    print("=" * 60)
    print("📚 CO₂电催化还原论文分析工具")
    print("=" * 60)
    print(f"📍 输入: {args.input}")
    print(f"📍 输出: {args.output}")
    print(f"🤖 模型: {args.model}")
    print(f"🌐 API:  {args.api_base}")
    print("=" * 60 + "\n")
    
    # 执行分析
    stats = batch_analyze(
        input_dir=args.input,
        output_dir=args.output,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        force=args.force,
        delay=args.delay
    )
    
    # 输出统计
    print("\n" + "=" * 60)
    print("📊 分析完成!")
    print(f"  ✅ 成功: {stats['success']}")
    print(f"  ⏭️  跳过: {stats['skipped']}")
    print(f"  ❌ 失败: {stats['failed']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
