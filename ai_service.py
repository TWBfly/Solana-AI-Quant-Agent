"""
Solana AI Quant Agent - Unified AI Service & Provider Management
Supports mainstream AI providers (EvoMap, DeepSeek, OpenAI, Claude, Qwen, Custom).
Defaults to EvoMap configured in .env.
"""

import os
import re
import json
import time
from typing import Dict, Any, Optional, List
import requests
from openai import OpenAI


class AIService:
    """Manages AI provider configurations, connection testing, and LLM calls."""

    CONFIG_FILE = "data/ai_config.json"

    PROVIDER_PRESETS = {
        "evomap": {
            "name": "EvoMap (默认)",
            "base_url": "https://api.evomap.ai/v1",
            "model": "evomap-deepseek-v4-flash",
            "description": "默认高吞吐低延迟量化模型",
            "is_default": True
        },
        "deepseek": {
            "name": "DeepSeek (官方)",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "description": "DeepSeek-V3 / R1 官方因果推理模型",
            "is_default": False
        },
        "openai": {
            "name": "OpenAI",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "description": "GPT-4o / GPT-4o-mini 旗舰通用模型",
            "is_default": False
        },
        "claude": {
            "name": "Claude (Anthropic)",
            "base_url": "https://api.anthropic.com/v1",
            "model": "claude-3-5-sonnet-20241022",
            "description": "Claude 3.5 高阶逻辑架构与代码分析",
            "is_default": False
        },
        "qwen": {
            "name": "通义千问 (DashScope)",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-plus",
            "description": "阿里云通义千问兼容模式",
            "is_default": False
        },
        "custom": {
            "name": "自定义接口 (OpenAI兼容)",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "llama3",
            "description": "本地 Ollama / vLLM 或第三方中转站",
            "is_default": False
        }
    }

    def __init__(self, config_path: str = "data/ai_config.json"):
        self.config_path = config_path
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

    @classmethod
    def get_env_keys(cls) -> Dict[str, str]:
        """Scans current and parent .env files to extract AI API keys."""
        env_files = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(os.getcwd()), ".env")
        ]
        keys = {}

        for p in env_files:
            if not os.path.exists(p):
                continue
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # 1. EvoMap section regex
                if "evomap" not in keys:
                    evo_m = re.search(r'#\s*evomap[\s\S]*?(?:APIkey|api_key|EVOMAP_API_KEY)\s*=\s*([^\r\n]+)', content, re.IGNORECASE)
                    if evo_m:
                        keys["evomap"] = evo_m.group(1).split('#')[0].strip()

                # 2. DeepSeek
                if "deepseek" not in keys:
                    ds_m = re.search(r'DEEPSEEK_API_KEY\s*=\s*([^\r\n]+)', content)
                    if ds_m:
                        keys["deepseek"] = ds_m.group(1).split('#')[0].strip()

                # 3. OpenAI
                if "openai" not in keys:
                    oa_m = re.search(r'OPENAI_API_KEY\s*=\s*([^\r\n]+)', content)
                    if oa_m:
                        keys["openai"] = oa_m.group(1).split('#')[0].strip()

                # 4. Anthropic Claude
                if "claude" not in keys:
                    cl_m = re.search(r'(?:ANTHROPIC_API_KEY|CLAUDE_API_KEY)\s*=\s*([^\r\n]+)', content)
                    if cl_m:
                        keys["claude"] = cl_m.group(1).split('#')[0].strip()

                # 5. Qwen
                if "qwen" not in keys:
                    qw_m = re.search(r'(?:DASHSCOPE_API_KEY|QWEN_API_KEY)\s*=\s*([^\r\n]+)', content)
                    if qw_m:
                        keys["qwen"] = qw_m.group(1).split('#')[0].strip()

            except Exception as e:
                print(f"[AIService] 警告: 读取 {p} 失败: {e}")

        return keys

    def get_config(self) -> Dict[str, Any]:
        """Returns active AI configuration merged with .env defaults."""
        env_keys = self.get_env_keys()
        saved = {}

        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
            except Exception as e:
                print(f"[AIService] 读取持久化配置失败 ({e})，使用默认配置")

        active_provider = saved.get("active_provider", "evomap")
        providers = {}

        for p_id, preset in self.PROVIDER_PRESETS.items():
            saved_p = saved.get("providers", {}).get(p_id, {})
            env_key = env_keys.get(p_id, "")
            
            # Key priority: saved key > env key > default empty
            api_key = saved_p.get("api_key") if saved_p.get("api_key") is not None else env_key
            base_url = saved_p.get("base_url") or preset["base_url"]
            model = saved_p.get("model") or preset["model"]

            providers[p_id] = {
                "id": p_id,
                "name": preset["name"],
                "base_url": base_url,
                "model": model,
                "api_key": api_key,
                "has_env_key": bool(env_key),
                "description": preset["description"],
                "is_active": (p_id == active_provider)
            }

        return {
            "active_provider": active_provider,
            "providers": providers
        }

    def save_config(self, req_data: Dict[str, Any]) -> Dict[str, Any]:
        """Persists AI settings to data/ai_config.json."""
        active_provider = req_data.get("active_provider", "evomap")
        new_providers = req_data.get("providers", {})

        current_cfg = self.get_config()
        to_save = {
            "active_provider": active_provider,
            "providers": {},
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        for p_id in self.PROVIDER_PRESETS.keys():
            p_data = new_providers.get(p_id, {})
            cur_p = current_cfg["providers"].get(p_id, {})
            to_save["providers"][p_id] = {
                "api_key": p_data.get("api_key", cur_p.get("api_key", "")),
                "base_url": p_data.get("base_url", cur_p.get("base_url", "")),
                "model": p_data.get("model", cur_p.get("model", ""))
            }

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)

        return self.get_config()

    def reset_to_default(self) -> Dict[str, Any]:
        """Restores AI settings to .env defaults (EvoMap active)."""
        if os.path.exists(self.config_path):
            try:
                os.remove(self.config_path)
            except Exception:
                pass
        return self.get_config()

    def test_connection(
        self,
        provider_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Tests live connectivity to the specified or active AI provider."""
        cfg = self.get_config()
        p_info = cfg["providers"].get(provider_id, {})

        key = api_key if api_key is not None else p_info.get("api_key", "")
        url = (base_url or p_info.get("base_url", "")).rstrip("/")
        mod = model or p_info.get("model", "")

        if not key:
            return {
                "success": False,
                "latency_ms": 0,
                "error": "未配置 API Key，请先输入密钥"
            }

        start_t = time.time()

        try:
            # Special handling for Anthropic official API if URL contains anthropic.com
            if provider_id == "claude" and "anthropic.com" in url:
                headers = {
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                payload = {
                    "model": mod,
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "ping"}]
                }
                res = requests.post(f"{url}/messages", headers=headers, json=payload, timeout=8.0)
                latency = int((time.time() - start_t) * 1000)
                if res.status_code == 200:
                    data = res.json()
                    txt = data.get("content", [{}])[0].get("text", "pong")
                    return {"success": True, "latency_ms": latency, "reply": txt.strip()}
                else:
                    return {"success": False, "latency_ms": latency, "error": f"HTTP {res.status_code}: {res.text[:200]}"}

            # Standard OpenAI Compatible Endpoint (EvoMap, DeepSeek, OpenAI, Qwen, Custom)
            client = OpenAI(
                base_url=url,
                api_key=key,
                timeout=8.0
            )
            resp = client.chat.completions.create(
                model=mod,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=10
            )
            latency = int((time.time() - start_t) * 1000)
            content = resp.choices[0].message.content or "pong"
            return {
                "success": True,
                "latency_ms": latency,
                "reply": content.strip()
            }

        except Exception as e:
            latency = int((time.time() - start_t) * 1000)
            err_msg = str(e)
            if "insufficient_user_quota" in err_msg or "quota" in err_msg.lower():
                err_msg = "账户余额或额度不足 (Quota Insufficient)"
            elif "invalid_api_key" in err_msg or "Incorrect API key" in err_msg or "401" in err_msg:
                err_msg = "API Key 认证失败，请检查密钥是否正确"
            elif "timed out" in err_msg.lower() or "timeout" in err_msg.lower():
                err_msg = f"连接超时 ({latency}ms)，请检查网络或代理地址"
            return {
                "success": False,
                "latency_ms": latency,
                "error": err_msg
            }

    def call_llm(
        self,
        messages: List[Dict[str, str]],
        provider_id: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2500
    ) -> str:
        """Invokes active LLM for text generation, code transpilation, or diagnosis."""
        cfg = self.get_config()
        pid = provider_id or cfg["active_provider"]
        p_info = cfg["providers"].get(pid, {})

        key = p_info.get("api_key", "")
        url = p_info.get("base_url", "").rstrip("/")
        mod = p_info.get("model", "")

        if not key:
            raise ValueError(f"AI 服务商 [{p_info.get('name', pid)}] 未配置 API Key，请在【设置】中配置。")

        # Claude native
        if pid == "claude" and "anthropic.com" in url:
            headers = {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": mod,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            res = requests.post(f"{url}/messages", headers=headers, json=payload, timeout=45.0)
            if res.status_code == 200:
                data = res.json()
                return data.get("content", [{}])[0].get("text", "")
            raise RuntimeError(f"Claude API 响应错误 ({res.status_code}): {res.text}")

        # Standard OpenAI client
        client = OpenAI(base_url=url, api_key=key, timeout=45.0)
        resp = client.chat.completions.create(
            model=mod,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content or ""


# Singleton instance
ai_service = AIService()
