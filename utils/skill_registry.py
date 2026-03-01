import os
import glob
import yaml
import importlib.util
import re
import json
import sys

class SkillRegistry:
    def __init__(self, skills_dir="./skills"):
        self.skills_dir = skills_dir
        self.index = {} 
        self.active_skills = {} 

    # ==========================
    # 1. 发现阶段 (保持不变)
    # ==========================
    def build_index(self):
        if not os.path.exists(self.skills_dir): return
        print(f"[*] [Registry] Indexing skills in {self.skills_dir}...")
        
        for meta_file in glob.glob(os.path.join(self.skills_dir, "**", "SKILL.md"), recursive=True):
            try:
                header = self._read_frontmatter(meta_file)
                if not header: continue
                
                meta = yaml.safe_load(header)
                name = meta.get('name')
                if name:
                    self.index[name] = {
                        "path": meta_file,
                        "description": meta.get('description', ''),
                        "license": meta.get('license', None),
                        "loaded": False
                    }
            except Exception as e:
                print(f"Error indexing {meta_file}: {e}")

    def _read_frontmatter(self, path):
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            if f.readline().strip() != '---': return None
            for line in f:
                if line.strip() == '---': break
                lines.append(line)
        return "".join(lines)

    def get_menu_prompt(self):
        if not self.index: return "No skills found."
        prompt = "### Skill Library (Index)\nTo access detailed docs, call `activate_skill(name)`.\n"
        for name, info in self.index.items():
            loaded = "[ACTIVE]" if info['loaded'] else ""
            prompt += f"- {name} {loaded}: {info['description']}\n"
        return prompt

    # ==========================
    # 2. 激活阶段 (简化版：专注文档)
    # ==========================
    def activate_skill(self, skill_name):
        if skill_name not in self.index: return False, f"Skill {skill_name} not found."
        if self.index[skill_name]['loaded']: return True, f"Skill {skill_name} already active."

        info = self.index[skill_name]
        try:
            with open(info['path'], 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析完整 YAML 和 文档正文
            match = re.search(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)
            yaml_content = match.group(1) if match else ""
            doc_body = match.group(2).strip() if match else content

            # Debug: print skill doc body when activated (verify LLM actually loaded SKILL.md)
            try:
                preview_len = 400
                preview = doc_body[:preview_len]
                if len(doc_body) > preview_len:
                    preview += "..."
                print(f"[SKILL_DOC_LOADED] name={skill_name} path={info['path']}")
                print("[SKILL_DOC_BEGIN]\n" + preview + "\n[SKILL_DOC_END]")
            except Exception:
                pass
            full_meta = yaml.safe_load(yaml_content) if yaml_content else {}

            self.active_skills[skill_name] = {
                "meta_path": info['path'],
                "doc": doc_body,
                "yaml": full_meta
            }
            self.index[skill_name]['loaded'] = True
            
            # [修改点 1] 不再展示文件树，专注业务逻辑
            msg = f"\n✅ **System: {skill_name} Activated**\n"
            msg += f"Docs:\n{'-'*20}\n{doc_body}\n{'-'*20}\n"
            msg += "Usage:\n"
            msg += "Output JSON to use this tool: `{\"tool\": \"" + skill_name + "\", \"args\": {...}}`\n"
            # [修改点 2] 移除了关于 __script__ 的引导，防止模型绕过 main.py
            
            return True, msg
            
        except Exception as e:
            return False, str(e)

    # ==========================
    # 3. 执行阶段 (简化版：锁定入口)
    # ==========================
    def execute_tool(self, tool_name, args):
        if tool_name == 'activate_skill':
            _, msg = self.activate_skill(args.get('name'))
            return msg
            
        if tool_name not in self.active_skills:
            return "Skill not activated. Call activate_skill first."
            
        folder = os.path.dirname(self.active_skills[tool_name]['meta_path'])
        
        # [修改点 3] 移除 args.pop('__script__') 逻辑，不接受外部指定文件
        
        try:
            # 查找入口文件
            script_path = self._resolve_entry_script(folder, tool_name)
            
            # 执行
            return self._run_python_script(script_path, args)
        except Exception as e:
            return f"Execution failed: {str(e)}"

    def _resolve_entry_script(self, folder, skill_name, user_specified=None):
        """
        核心逻辑：锁定寻找 main.py 或 tool.py
        """
        # 1. 优先看 YAML 是否配置了 entry
        yaml_entry = self.active_skills[skill_name]['yaml'].get('entry')
        if yaml_entry:
            entry_path = os.path.join(folder, yaml_entry)
            if os.path.exists(entry_path):
                return entry_path
            
        scripts_dir = os.path.join(folder, "scripts")

        # 2. 优先寻找 scripts/main.py
        if os.path.exists(os.path.join(scripts_dir, "main.py")):
            return os.path.join(scripts_dir, "main.py")
            
        # 3. 其次寻找 scripts/tool.py
        if os.path.exists(os.path.join(scripts_dir, "tool.py")):
            return os.path.join(scripts_dir, "tool.py")

        # 4. 最后尝试寻找 scripts/同名文件
        norm_name = skill_name.replace("-", "_") + ".py"
        if os.path.exists(os.path.join(scripts_dir, norm_name)):
            return os.path.join(scripts_dir, norm_name)
            
        # 5. 如果都没有，找 scripts/ 下唯一的 .py 文件
        candidates = glob.glob(os.path.join(scripts_dir, "*.py"))
        candidates = [c for c in candidates if not c.endswith('__init__.py')]
        if len(candidates) == 1:
            return candidates[0]

        # 6. 回退：在根目录寻找 main.py/tool.py/同名文件
        if os.path.exists(os.path.join(folder, "main.py")):
            return os.path.join(folder, "main.py")
        if os.path.exists(os.path.join(folder, "tool.py")):
            return os.path.join(folder, "tool.py")
        if os.path.exists(os.path.join(folder, norm_name)):
            return os.path.join(folder, norm_name)
            
        raise FileNotFoundError(f"Cannot find entry point (main.py/tool.py) in {folder}")

    def _run_python_script(self, path, args):
        # 将相对路径转为绝对路径
        # importlib 对 "./skills/..." 这种路径支持不好，必须用 "/home/user/..."
        abs_path = os.path.abspath(path)
        
        spec = importlib.util.spec_from_file_location("dynamic_skill", abs_path)
        
        # 增加防御性检查
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load Python spec from: {abs_path}")
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        sys.path.append(os.path.dirname(abs_path)) 
        spec.loader.exec_module(module)
        
        for func_name in ['execute', 'main', 'run']:
            if hasattr(module, func_name):
                print(f"[*] Calling {os.path.basename(path)} -> {func_name}({list(args.keys())})")
                return getattr(module, func_name)(**args)
        
        raise AttributeError(f"No execute/main function found in {os.path.basename(path)}")