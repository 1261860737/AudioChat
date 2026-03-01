import json
import re
from utils.skill_registry import SkillRegistry
from utils.llm_client import LLMClient

class AgentEngine:
    def __init__(
        self,
        skills_dir: str = "./skills",
        vllm_url: str = "http://127.0.0.1:8000/v1",
        model_name: str = "Qwen3-Coder-30B-A3B-Instruct-FP8",
        *,
        llm_temperature: float = 0.1,
        llm_max_tokens: int = 2048,
        llm_timeout_s: float = 60.0,
        system_prompt: str | None = None,
    ):
        # 1. 初始化组件
        self.registry = SkillRegistry(skills_dir)
        self.registry.build_index() # 建立索引
        self.llm = LLMClient(
            base_url=vllm_url,
            model_name=model_name,
            timeout_s=llm_timeout_s,
            max_tokens=llm_max_tokens,
        )
        self.llm_temperature = float(llm_temperature)
        self.llm_max_tokens = int(llm_max_tokens)
        self.llm_timeout_s = float(llm_timeout_s)
        
        # 2. 基础设定
        default_system_prompt = (
            "你是一个能够使用本地工具的高级AI助手.\n"
            "协议:\n"
            "1. 查看[Skill Index]. 如果需要某个技能但该技能未[激活]，请调用`activate_skill(name)`。\n"
            "2. 阅读系统提供的文档。\n"
            "3. 只有在需要调用工具时才输出JSON，且必须包含非空字符串字段 tool，例如: "
            "{\"tool\": \"calculator\", \"args\": {\"expression\": \"2+2\"}}。\n"
            "4. 如果需要调用工具，只输出 JSON（不得包含任何自然语言解释/说明）。\n"
            "5. 如果不调用工具，请直接输出自然语言答案，且不要输出任何JSON。\n"
        )
        self.system_prompt = system_prompt or default_system_prompt
        self.history = []
        self.max_history_messages = 10
        self._menu_injected = False

    def run(self, user_query, max_turns=10):
        """
        执行一次完整的任务循环。
        :param user_query: 用户的输入 (来自 ASR)
        :return: 最终生成的文本 (给 TTS)
        """
        # 保留历史记忆，首次调用时初始化 System Prompt
        if not self.history:
            self._init_history()

        # 仅首次注入技能菜单，避免重复堆叠
        if not self._menu_injected:
            menu = self.registry.get_menu_prompt()
            self.history[0]["content"] += f"\n{menu}"
            self._menu_injected = True
        
        self.history.append({"role": "user", "content": user_query})
        self._trim_history()
        
        print(f"\n--- New Task: {user_query} ---")

        for turn in range(max_turns):
            # 1. LLM 思考
            print(f"[Turn {turn+1}] Thinking...")
            response = self.llm.chat(
                self.history,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                timeout_s=self.llm_timeout_s,
            )
            # print(f"  [LLM Raw] {response[:300]}{'...' if len(response) > 300 else ''}")
            
            # 2. 尝试解析工具调用
            tool_data = self._parse_json(response)
            if tool_data:
                print(f"  [LLM Parsed] tool={tool_data.get('tool')}")
            else:
                # 二次转换：把普通文本转换成工具 JSON（如果需要）
                tool_data = self._convert_text_to_tool_json(response)
                if tool_data:
                    print(f"  [LLM Parsed/Converted] tool={tool_data.get('tool')}")
            
            # 分支 A: 模型想要执行工具
            if tool_data:
                tool_name = tool_data.get('tool')
                args = tool_data.get('args', {})
                print(f"  [Action] Calling: {tool_name} | Args: {args}")

                # 自动激活：当模型直接调用技能但尚未激活时，先激活再执行
                if (
                    tool_name != "activate_skill"
                    and tool_name in self.registry.index
                    and tool_name not in self.registry.active_skills
                ):
                    success, activation_msg = self.registry.activate_skill(tool_name)
                    if success:
                        # 注入文档，便于下一轮模型理解
                        self.history.append({"role": "system", "content": activation_msg})
                        print(f"  [Auto-Activate] {tool_name} activated.")
                    else:
                        # 如果激活失败，直接返回错误给模型，不要硬执行了
                        err_msg = f"System Error: Failed to auto-activate {tool_name}. {activation_msg}"
                        self.history.append({"role": "assistant", "content": response})
                        self.history.append({"role": "user", "content": err_msg})
                        print(f"  [Error] {err_msg}")
                        continue # 跳过本次循环
                
                # 执行工具 (SkillRegistry 会处理 激活 vs 执行)

                # Guard: prevent executing send-email with placeholder/invalid args
                if tool_name == 'send-email' and (not self._is_send_email_args_valid(args)):
                    err_msg = self._send_email_args_error()
                    # Record model output then ask it to retry with correct JSON (no tool execution this turn)
                    self.history.append({"role": "assistant", "content": response})
                    self.history.append({"role": "user", "content": err_msg})
                    self._trim_history()
                    print(f"  [Guard] {err_msg}")
                    continue
                result_text = self.registry.execute_tool(tool_name, args)
                
                # 将结果写入历史
                # 注意：如果是 activate_skill，SkillRegistry 返回的是 System Prompt 的一部分
                # 我们统一作为 function 角色或者 System 角色注入
                if tool_name == "activate_skill":
                    # 如果是显式激活，结果是系统消息
                    observation_msg = {"role": "system", "content": result_text}
                else:
                    # 如果是普通工具，结果是用户反馈
                    observation_msg = {"role": "user", "content": f"Tool output: {str(result_text)}"}

                self.history.append({"role": "assistant", "content": response})
                self.history.append(observation_msg)

                # If send-email ran, force next assistant message to include full meeting minutes markdown + link
                if tool_name == 'send-email':
                    self.history.append({
                        "role": "system",
                        "content": (
                            "下一条回复必须输出【完整会议纪要 Markdown】（包含‘行动项/Action Items’小节与条目），"
                            "并附上工具返回的链接。不要只回复‘已上传/已生成’这样的状态句。"
                        )
                    })
                self._trim_history()
                
                print(f"  [Observation] Result: {str(result_text)[:100]}...") # 只打前100字
                # 继续循环，让模型根据结果生成下一句
                
            # 分支 B: 模型输出普通文本 (任务结束或反问)
            else:
                print(f"  [Response] {response}")
                self.history.append({"role": "assistant", "content": response})
                self._trim_history()
                return response

        return "Task finished (Max turns reached)."

    # ----------------------------
    # Tool-call guards (debug/robustness)
    # ----------------------------
    def _is_send_email_args_valid(self, args: dict) -> bool:
        """Validate args for send-email to prevent empty/placeholder dispatch."""
        if not isinstance(args, dict):
            return False
        # Prefer structured meeting_result
        mr = args.get('meeting_result')
        if isinstance(mr, dict):
            action_items = mr.get('action_items')
            if isinstance(action_items, list) and len(action_items) >= 1:
                return True
        llm_text = args.get('llm_text')
        if not isinstance(llm_text, str):
            return False
        s = llm_text.strip()
        if len(s) < 200:
            return False
        # Must include an action items section (Chinese or English) and at least 2 bullet-like items
        if ('行动项' not in s) and ('Action Items' not in s):
            return False
        bullets = s.count('\n- ') + s.count('\n* ')
        if bullets < 2 and s.count('：') < 2:
            return False
        # Reject common placeholder/status-only strings
        bad_phrases = ['已生成', '已成功', '已上传', '请通过以下链接查看', '查看：', '查看:']
        if len(s) < 260 and any(p in s for p in bad_phrases):
            return False
        return True

    def _send_email_args_error(self) -> str:
        return (
            "System Error: send-email 参数不合格。\n"
            "- 你必须在 args 中提供 meeting_result（推荐）或 llm_text（完整会议纪要 Markdown 原文）。\n"
            "- llm_text 不能是状态句（例如：‘会议纪要已生成并上传至Gitea。’）。\n"
            "- llm_text 必须包含‘行动项/Action Items’小节，并至少包含 2 条行动项。\n"
            "请重写工具调用 JSON。"
        )

    def _init_history(self):
        self.history = [{"role": "system", "content": self.system_prompt}]
        self._menu_injected = False

    def _trim_history(self):
        """保留 system + 最近 N 条非 system 消息，避免历史无限增长。"""
        if not self.history:
            return
        system_msg = self.history[0]
        non_system = [m for m in self.history[1:] if m.get("role") != "system"]
        if len(non_system) <= self.max_history_messages:
            self.history = [system_msg, *non_system]
            return
        self.history = [system_msg, *non_system[-self.max_history_messages:]]

    def _parse_json(self, text):
        """辅助函数：从回复中提取 JSON"""
        try:
            # 匹配 markdown 代码块
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match: text = match.group(1)
            # 匹配纯 JSON 结构
            match_raw = re.search(r'\{.*\}', text, re.DOTALL)
            if match_raw: 
                data = json.loads(match_raw.group(0))
                if not isinstance(data, dict):
                    return None
                tool = data.get("tool") or data.get("name")
                if "args" not in data and "arguments" in data:
                    data["args"] = data.get("arguments")
                data["tool"] = tool
                if not isinstance(tool, str) or not tool.strip():
                    return None
                return data
        except:
            return None
        return None

    def _convert_text_to_tool_json(self, text: str):
        """把普通文本强制转换成工具调用 JSON（如果文本里明确提到要用工具）。"""
        convert_system = (
            "你是工具调用转换器。"
            "如果文本明确表示需要调用某个技能/工具，请输出且只输出 JSON："
            "{\"tool\": \"<tool_name>\", \"args\": {..}}。"
            "如果文本不需要调用工具，只输出：NO_TOOL。"
            "禁止输出任何解释。"
        )
        convert_messages = [
            {"role": "system", "content": convert_system},
            {"role": "user", "content": text},
        ]
        converted = self.llm.chat(
            convert_messages,
            temperature=0.0,
            max_tokens=256,
            timeout_s=self.llm_timeout_s,
        )
        if isinstance(converted, str) and converted.strip() == "NO_TOOL":
            return None
        return self._parse_json(converted)

# if __name__ == "__main__":
#     engine = AgentEngine(model_name="Qwen3-Coder-30B-A3B-Instruct-FP8")
#     print(engine.run("算一下 123 乘以 45 等于多少"))