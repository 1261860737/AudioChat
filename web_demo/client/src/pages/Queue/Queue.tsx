import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useState, useCallback, useRef, MutableRefObject } from "react";
import eruda from "eruda";
import { useSearchParams } from "react-router-dom";
import { Conversation, ConversationMode } from "../Conversation/Conversation";
import { Button } from "../../components/Button/Button";
import { useModelParams } from "../Conversation/hooks/useModelParams";
import { env } from "../../env";


function getFloatFromStorage(val: string | null) {
  return (val == null) ? undefined : parseFloat(val)
}

function getIntFromStorage(val: string | null) {
  return (val == null) ? undefined : parseInt(val)
}

export const Queue: FC = () => {
  const [searchParams] = useSearchParams();
  const overrideWorkerAddr = searchParams.get("worker_addr");
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const [shouldConnect, setShouldConnect] = useState<boolean>(false);
  const [connectionMode, setConnectionMode] = useState<ConversationMode | null>(null);
  // Repurposed: this controls TTS "style/emotion" instruction (not LLM system prompt).
  const [systemPrompt, setSystemPrompt] = useState<string>("用冷静、专业、平稳的语气说，音调中等，不要激动。");
  const [userInstruction, setUserInstruction] = useState<string>("");
  const userInstructionPlaceholder =
    "你是会议纪要行动项助手。请基于分角色转写抽取姓名（会议开始通常会有“我是张三”之类自我介绍），"
    + "并在输出中使用真实姓名替代 spk0/spk1。"
    + "你的输出必须是合法 JSON，且只能输出一个 JSON 对象，禁止 Markdown、代码块和额外解释。"
    + "固定输出结构："
    + '{"meeting_info":{"title":"会议标题","date":"YYYY-MM-DD","participants":["张三","李四"]},'
    + '"action_items":["张三：完成接口联调","李四：整理会议纪要并同步"]}'
    + "其中 action_items 的每一项必须是“责任人：任务”格式。";
  const modelParams = useModelParams({
    textTemperature: getFloatFromStorage(localStorage.getItem("textTemperature")),
    textTopk: getIntFromStorage(localStorage.getItem("textTopk")),
    audioTemperature: getFloatFromStorage(localStorage.getItem("audioTemperature")),
    audioTopk: getIntFromStorage(localStorage.getItem("audioTopk")),
    padMult: getFloatFromStorage(localStorage.getItem("padMult")),
    repetitionPenalty: getFloatFromStorage(localStorage.getItem("repetitionPenalty")),
    repetitionPenaltyContext: getIntFromStorage(localStorage.getItem("repetitionPenaltyContext")),
    imageResolution: getIntFromStorage(localStorage.getItem("imageResolution"))
  });

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);
  // enable eruda in development
  useEffect(() => {
    if (env.VITE_ENV === "development") {
      eruda.init();
    }
    () => {
      if (env.VITE_ENV === "development") {
        eruda.destroy();
      }
    };
  }, []);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch (e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
  }, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage]);

  const startProcessor = useCallback(async () => {
    if (!audioContext.current) {
      audioContext.current = new AudioContext();
    }
    if (worklet.current) {
      return;
    }
    let ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    } catch (err) {
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    }
    worklet.current.connect(ctx.destination);
  }, [audioContext, worklet]);

  const onConnect = useCallback(async (mode: ConversationMode) => {
    await startProcessor();
    const hasAccess = await getMicrophoneAccess();
    if (hasAccess) {
      setConnectionMode(mode);
      setShouldConnect(true);
    }
  }, [setShouldConnect, startProcessor, getMicrophoneAccess, setConnectionMode]);

  if (hasMicrophoneAccess && audioContext.current && worklet.current && connectionMode) {
    // workerAddr 直接使用模式名，Vite 代理会根据路径转发
    const workerAddr = overrideWorkerAddr ?? connectionMode;
    return (
      <Conversation
        workerAddr={workerAddr}
        audioContext={audioContext as MutableRefObject<AudioContext>}
        worklet={worklet as MutableRefObject<AudioWorkletNode>}
        mode={connectionMode}
        systemPrompt={systemPrompt}
        {...modelParams}
      />
    );
  }

  return (
    <div className="text-white text-center h-screen w-screen p-4 flex flex-col items-center ">
      <div>
        <h1 className="text-4xl" style={{ letterSpacing: "5px" }}>S2S-Demo</h1>
        <div className="pt-8 text-sm flex justify-center items-center flex-col ">
          <div className="presentation text-center">
            <p>你好 欢迎使用</p>
          </div>
        </div>
      </div>
      <div className="flex flex-grow justify-center items-center flex-col presentation">
        <div className="w-full max-w-xl mb-6 px-4">
          <label className="block text-left mb-2 text-sm opacity-80">TTS 语气/情绪指令（默认：冷静、专业、平稳）</label>
          <textarea
            className="w-full h-32 p-3 bg-black border-2 border-white text-white rounded-none resize-none focus:outline-none focus:border-blue-400"
            placeholder="例如：用冷静、专业、平稳的语气说，音调中等，不要激动。"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
          />
        </div>
        <div className="mt-2 text-xs opacity-70 text-left">
            当前后端固定 CustomVoice：speaker=Vivian，language=Chinese（后续可开放选择）
        </div>
        <div className="w-full max-w-xl mb-6 px-4">
          <label className="block text-left mb-2 text-sm opacity-80">User Instruction (会议指令)</label>
          <textarea
            className="w-full h-32 p-3 bg-black border-2 border-white text-white rounded-none resize-none focus:outline-none focus:border-blue-400"
            placeholder={userInstructionPlaceholder}
            value={userInstruction}
            onChange={(e) => setUserInstruction(e.target.value)}
          />
        </div>
        <div className="flex gap-4">
          <Button onClick={async () => await onConnect('simplex')}>
            <span className="flex flex-col items-center">
              <span>开始连接</span>
            </span>
          </Button>
        </div>
      </div>
      <div className="flex flex-grow justify-center items-center flex-col">
        {showMicrophoneAccessMessage &&
          <p className="text-center">Please enable your microphone before proceeding</p>
        }
      </div>
    </div >
  )
};
