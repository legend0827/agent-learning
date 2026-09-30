import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 读 .env 文件

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],  # ← 变量名，不是 key 本身
    base_url="https://api.deepseek.com",
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "用三句话解释什么是 AI Agent"}],
)
print(resp.choices[0].message.content)
print("用量:", resp.usage)
