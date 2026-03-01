import requests  # 如果没有安装，请 pip install requests

class LLMClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model_name: str = "your_model_name",
        *,
        timeout_s: float = 60.0,
        max_tokens: int = 2048,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)

    def chat(self, messages, temperature: float = 0.1, *, max_tokens: int | None = None, timeout_s: float | None = None):
        """
        发送对话历史给 vLLM，返回模型的回复文本。
        """
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": int(self.max_tokens if max_tokens is None else max_tokens),
            "stream": False
        }

        try:
            # 发送请求
            response = requests.post(url, json=payload, timeout=self.timeout_s if timeout_s is None else timeout_s)
            response.raise_for_status()
            
            # 解析结果
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content.strip()
            
        except Exception as e:
            try:
                status = getattr(response, "status_code", None)
                body = getattr(response, "text", "")
                if status is not None:
                    print(f"[LLM Error] status: {status}")
                if body:
                    print(f"[LLM Error] body: {body[:2000]}")
            except Exception:
                pass
            print(f"[LLM Error] Connection failed: {e}")
            return "Error: LLM service unavailable."
