"""
LLM Caller

APIs:
- OpenAI (GPT)
- Anthropic (Claude)
- DeepSeek
- 阿里云 (Aliyun)
- 硅基流动

usage

1. 安装所需依赖：
```
pip install openai anthropic python-dotenv zhipuai
```

2. 创建包含 API 密钥的 `.env` 文件：
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
ALIYUN_API_KEY=your_aliyun_api_key
SILICON_API_KEY=your_silicon_api_key
```

## Usage

```python
from llm_caller import call_llm

# Example prompt
prompt = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]

# 调用 OpenAI GPT
response = call_llm(prompt, "gpt-3.5-turbo")
print(response)

# Claude
response = call_llm(prompt, "claude-3-opus-20240229")
print(response)

# DeepSeek
response = call_llm(prompt, "deepseek-chat")
print(response)

# 阿里云
response = call_llm(prompt, "aliyun/qwen-max")
print(response)

# 硅基流动
response = call_llm(prompt, "silicon/claude-3-opus-20240229")
print(response)
```

## 模型命名
- OpenAI 模型：使用标准 OpenAI 模型名称（例如"gpt-3.5-turbo"、"gpt-4"）
- Anthropic 模型：使用标准 Claude 模型名称（例如"claude-3-opus-20240229"）
- DeepSeek 模型：使用标准 DeepSeek 模型名称
- 阿里云模型：前缀为"aliyun/"（例如"aliyun/qwen-max"）
- 硅基流动 模型：前缀为"silicon/"（例如"silicon/claude-3-opus-20240229"）

@author: 杨绎
@date: 2025-03-12
"""

from dotenv import load_dotenv
import os
import openai
import anthropic
from zhipuai import ZhipuAI

# 从 .env 文件加载环境变量
load_dotenv()

def get_llm_provider(model_name:str):
    """
    根据模型名称获取提供商
    
    参数:
        model_name: 模型名称
        
    Returns:
        Provider name as string
    """
    if model_name.startswith("silicon/"):
        return "silicon"
    elif model_name.startswith("aliyun/"):
        return "aliyun"
    elif "gpt" in model_name:
        return "openai"
    elif "claude" in model_name:
        return "anthropic"
    elif "deepseek" in model_name:
        return "deepseek"
    return None

def call_llm(prompt:list, model:str, temperature=0):
    """
    Call LLM model with provided prompt
    
    参数:
        prompt: 格式为 [{'role': 'user', 'content': 'Hello'}]
        model: 模型名称
        
    Returns:
        Generated text response
    """
    provider = get_llm_provider(model)
    if provider is None:
        raise ValueError(f"不支持的大语言模型: {model}")
        
    if provider == "openai":
        return call_openai_llm(prompt, model, temperature)
    elif provider == "anthropic":
        return call_anthropic_llm(prompt, model, temperature)
    elif provider == "deepseek":
        return call_deepseek_model(prompt, model, temperature)
    elif provider == "aliyun":
        return call_aliyun_model(prompt, model, temperature)
    elif provider == "silicon":
        return call_silicon_model(prompt, model, temperature)

def call_openai_llm(prompt:list, model:str, temperature=0):
    """
    Call OpenAI model
    
    参数:
        model: OpenAI 模型名称（例如"gpt-3.5-turbo"、"gpt-4"）
        
    Returns:
        Generated text
    """
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=temperature,
    )
    
    return response.choices[0].message.content

def call_anthropic_llm(prompt:list, model:str, temperature=0):
    """
    调用 Anthropic Claude 模型
    """
    if prompt[0]["role"] == "system":
        system_prompt = prompt[0]["content"]
        prompt = prompt[1:]
    else:
        system_prompt = None
        prompt = prompt

    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    message = client.messages.create(
        model=model,
        max_tokens=8192,
        temperature=temperature,
        system=system_prompt,
        messages=prompt
    )
    
    return message.content[0].text

def call_deepseek_model(prompt:list, model:str, temperature=0):
    """
    调用 DeepSeek 模型
    """
    client = openai.OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=temperature,
        max_tokens=8192
    )
    
    return response.choices[0].message.content

def call_aliyun_model(prompt:list, model:str, temperature=0):
    """
    调用阿里云模型
    
    参数:
        model: 阿里云模型名称（以"aliyun/"开头）
        
    """
    client = openai.OpenAI(
        api_key=os.getenv("ALIYUN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    reasoning_content = ""
    answer_content = ""
    
    stream = client.chat.completions.create(
        model=model[7:],  # 移除"aliyun/"前缀
        messages=prompt,
        temperature=temperature,
        stream=True
    )
    
    for chunk in stream:
        if not getattr(chunk, 'choices', None):
            continue
        delta = chunk.choices[0].delta
        if not getattr(delta, 'reasoning_content', None) and not getattr(delta, 'content', None):
            continue
        if getattr(delta, 'reasoning_content', None):
            reasoning_content += delta.reasoning_content
        elif getattr(delta, 'content', None):
            answer_content += delta.content
            
    response = f"<think>{reasoning_content}</think>{answer_content}"
    return response

def call_silicon_model(prompt:list, model:str, temperature=0):
    """
    硅基流动
    
    参数:
        model: Silicon 模型名称（以"silicon/"开头）
    """
    model = model[8:]  # 移除"silicon/"前缀
    reasoning_content = ""
    answer_content = ""
    
    client = openai.OpenAI(
        api_key=os.getenv("SILICON_API_KEY"),
        base_url="https://api.siliconflow.cn/v1"
    )
    
    stream = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=temperature,
        stream=True
    )
    
    for chunk in stream:
        if not getattr(chunk, 'choices', None):
            continue
        delta = chunk.choices[0].delta
        if not getattr(delta, 'reasoning_content', None) and not getattr(delta, 'content', None):
            continue
        if getattr(delta, 'reasoning_content', None):
            reasoning_content += delta.reasoning_content
        elif getattr(delta, 'content', None):
            answer_content += delta.content
            
    response = f"<think>{reasoning_content}</think>{answer_content}"
    return response

# Example usage
if __name__ == "__main__":
    # 调用大语言模型示例
    user_prompt ='''
        The following is a paragraph from an academic paper. Refinish writing to conform to academic style，improve spelling, grammar, clarity, conciseness and overall readability. If necessary, rewrite the entire sentence. In addition,list all modifications in the Markdown table and explain the reasons for doing so.               
        Paragraph ：Large language models (LLMs) employ in-context learning (ICL) in downstream tasks. By default, ICL selects demonstrations from a labeled example set to perform few-shot learning. Unfortunately, labeled examples may not always be available, and our study reveals a counterintuitive finding that labeled demonstrations sometimes result in suboptimal ICL performance. Therefore, we unlock an unexplored paradigm {\em unsupervised in-context learning}: amplify ICL using demonstrations selected from unlabeled examples with principally assigned inspiring labels. We mathematically reveal the key challenge that the demonstration construction complexity of unsupervised ICL is exponential times as much as traditional ICL. We propose a principled unsupervised ICL framework with heuristic pruning and importance sampling that decreases the complexity to a practically applicable level and verify its effectiveness in ICL with intensive experiments and analysis. 
    ''' 
    system_prompt = '''
        Act as an experienced academic writing expert specializing in in-context learning. Review my research paper draft, focusing on improving the logical flow, strengthening arguments, and refining language for publication quality. Highlight areas needing clarification or further development, and suggest specific improvements.
    '''
    prompt = [{'role': 'system', 'content': system_prompt},{"role": "user", "content": user_prompt}]
    model = "deepseek-r1"  # 更改为您想要的模型
    
    try:
        response = call_llm(prompt, model)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
