# LLM 调用器

一个用于调用各种大语言模型的简化 API。

## 支持的模型

- OpenAI GPT 模型（gpt-3.5-turbo、gpt-4 等）
- Anthropic Claude 模型
- DeepSeek 模型
- 阿里云模型
- Silicon 模型

## 设置

1. 安装所需依赖：
```
pip install openai anthropic python-dotenv zhipuai
```

2. 创建包含 API 密钥的 `.env` 文件：
```
OPENAI_API_KEY=你的_openai_api_密钥
ANTHROPIC_API_KEY=你的_anthropic_api_密钥
DEEPSEEK_API_KEY=你的_deepseek_api_密钥
ALIYUN_API_KEY=你的_aliyun_api_密钥
SILICON_API_KEY=你的_silicon_api_密钥
```

## 使用方法

```python
from llm_caller import call_llm

# 示例提示
prompt = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "什么是机器学习？"}
]

# 调用 OpenAI GPT
response = call_llm(prompt, "gpt-3.5-turbo")
print(response)

# 调用 Claude
response = call_llm(prompt, "claude-3-opus-20240229")
print(response)

# 调用 DeepSeek
response = call_llm(prompt, "deepseek-chat")
print(response)

# 调用阿里云
response = call_llm(prompt, "aliyun/qwen-max")
print(response)

# 调用 Silicon
response = call_llm(prompt, "silicon/claude-3-opus-20240229")
print(response)
```

## 模型命名规则

- OpenAI 模型：使用标准 OpenAI 模型名称（例如"gpt-3.5-turbo"、"gpt-4"）
- Anthropic 模型：使用标准 Claude 模型名称（例如"claude-3-opus-20240229"）
- DeepSeek 模型：使用标准 DeepSeek 模型名称
- 阿里云模型：前缀为"aliyun/"（例如"aliyun/qwen-max"）
- Silicon 模型：前缀为"silicon/"（例如"silicon/claude-3-opus-20240229"）