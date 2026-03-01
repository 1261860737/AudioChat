# AGENTS.md (Repo Guide for Agentic Coding)

> 目标：让自动化 coding agents 在本仓库里"少猜测、可复现、可验证"。
> 本仓库以 Python 为主，核心代码在 `audiochat/`、`scripts/`、`utils/`，并 vendored 了大量第三方代码在 `third_party/`。

## 0) Server Environment / 服务器环境

| 项目 | 值 |
|------|-----|
| **Python** | `/home/ditx/llm/voice/3D-Speaker/.3ds/bin/python3` (3.11.14) |
| **架构** | ARM64 (aarch64) |
| **GPU** | NVIDIA GB10 |
| **OS** | Ubuntu 24.04 LTS |
| **Docker** | 28.3.3 |

**重要**：本服务器为 ARM64 架构，部分 x86-only 的 Python 包可能需要从源码编译，注意依赖冲突。

## 1) Project Layout / 目录结构

- `audiochat/`: 核心库（diarization / ASR / LLM glue code）
- `scripts/`: CLI entrypoints（离线 pipeline）
- `utils/`: 常量、CosyVoice token <-> audio 相关工具
- `examples/`: 测试音频
- `third_party/`: 上游代码（不作为风格参考，尽量不改）
- `actions/`: post-processing actions（如任务派发/邮件草稿生成等，轻依赖，便于复用）
- `actions/mail_dispatcher/`: 根据会议 action items 按负责人邮箱生成邮件草稿（默认不发送）

**Scope**：修改集中在 `audiochat/`、`scripts/`、`utils/`；`third_party/` 仅在必要时最小化修改。

## 2) Models / 模型路径

所有模型存放在 `/data/models/Voice/`：

| 模型 | 路径 | 用途 |
|------|------|------|
| FunASR (Paraformer) | `/data/models/Voice/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | ASR 语音识别 |
| FunASR (SenseVoice) | `/data/models/Voice/iic/SenseVoiceSmall` | ASR 备选 |
| VAD | `/data/models/Voice/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 语音活动检测 |
| 标点 | `/data/models/Voice/iic/punc_ct-transformer_cn-en-common-vocab471067-large` | 标点恢复 |
| Speaker | `/data/models/Voice/iic/speech_campplus_sv_zh-cn_16k-common` | 说话人分离 |
| Fun-Audio-Chat | `/data/models/Voice/FunAudioLLM/Fun-Audio-Chat-8B` | LLM 对话 |
| CosyVoice | `/data/models/Voice/FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | TTS 合成 (S2S) |

## 3) Build / Lint / Test Commands

### 3.1 快速验证（语法检查）

```bash
/home/ditx/llm/voice/3D-Speaker/.3ds/bin/python3 -m compileall audiochat scripts utils
```

### 3.2 离线 Pipeline（Text-only）

```bash
/home/ditx/llm/voice/3D-Speaker/.3ds/bin/python3 scripts/offline_pipeline.py \
  --audio examples/2speakers_example.wav \
  --funasr-model /data/models/Voice/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  --funaudiochat-model /data/models/Voice/FunAudioLLM/Fun-Audio-Chat-8B \
  --output-dir AudioChat_saves
```

### 3.3 离线 Pipeline（Speech-to-Speech）

```bash
/home/ditx/llm/voice/3D-Speaker/.3ds/bin/python3 scripts/offline_pipeline_s2s.py \
  --audio examples/2speakers_example.wav \
  --funasr-model /data/models/Voice/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  --funaudiochat-model /data/models/Voice/FunAudioLLM/Fun-Audio-Chat-8B \
  --output-dir AudioChat_saves
```

### 3.4 单次最小验证

```bash
# 仅验证 import 是否正常（不运行推理）
/home/ditx/llm/voice/3D-Speaker/.3ds/bin/python3 -c "
from audiochat.audio_io import ensure_mono_16k
from audiochat.asr.funasr_asr import FunASRTranscriber
from audiochat.diarization.diarizer_3dspeaker import ThreeDSpeakerDiarizer
from audiochat.llm.funaudiochat_llm import FunAudioChatLLM
print('All imports OK')
"
```

## 4) Code Style / 代码规范

### 4.1 总体

- 4 空格缩进，PEP 8 风格
- 明确类型标注（type hints）与 dataclass
- 防御式输入检查 + 明确异常信息

### 4.2 Imports

- 顺序：stdlib -> third-party -> local
- 新文件保留 `from __future__ import annotations`
- 重依赖优先**函数内 import**（避免 import-time side effects）

### 4.3 Typing

- 优先内置泛型：`list[T]`, `dict[K, V]`, `tuple[...]`
- `Optional[T]` 用于可空
- 类型仅用于静态检查时使用 `TYPE_CHECKING`

### 4.4 Naming

- `snake_case`：函数/变量/模块
- `PascalCase`：类
- `UPPER_SNAKE_CASE`：常量

### 4.5 Error Handling

- 输入不符合预期：抛 `ValueError`，信息包含期望形状/范围
- 外部依赖缺失：抛 `RuntimeError`，给出安装提示
- 避免裸 `except:`；使用 `except Exception as exc` 并 `raise ... from exc`
- 不要吞异常（empty catch）

### 4.6 Formatting

- 行宽 <= 100（软约束）
- 新代码保持与邻近文件一致

## 5) ARM64 注意事项

本服务器为 ARM64 架构（NVIDIA GB10），常见问题：

1. **包编译**：部分 wheel 无 ARM64 预编译版本，需从源码编译
2. **环境冲突**：注意 CUDA/cuDNN 版本与 PyTorch 的兼容性
3. **性能差异**：某些算子在 ARM 上性能可能与 x86 不同

遇到 `pip install` 失败时，优先检查：
- 是否有 ARM64 wheel
- 是否需要安装编译依赖（`build-essential`, `cmake` 等）

## 6) Working with `third_party/`

- 默认原则：**不要修改**，除非修复集成问题必须改
- 如必须改：改动最小化，不要批量格式化，保持上游风格

## 7) Agent Operating Notes

1. **修改前定位调用链**：`scripts/` entrypoint -> `audiochat/` 逻辑 -> `utils/` 工具
2. **验证优先级**：
   - 语法：`python -m compileall ...`
   - Import：运行 3.4 的 import 测试
   - 功能：跑 `scripts/offline_pipeline.py`
3. **不要**：为补 lint 引入新依赖、硬写版本号、修改 `third_party/`
