import { FC, useEffect, useRef, useState } from "react";
import { useServerText } from "../../hooks/useServerText";

type TextDisplayProps = {
  containerRef: React.RefObject<HTMLDivElement>;
  displayColor: boolean | undefined;
};

export const TextDisplay: FC<TextDisplayProps> = ({
  containerRef,
  displayColor: _displayColor,
}) => {
  const { text } = useServerText();
  const [displayText, setDisplayText] = useState("");
  const [hasPending, setHasPending] = useState(false);
  const pendingTextRef = useRef("");
  const isAtBottomRef = useRef(true);
  const bottomThreshold = 24;

  const scrollToBottom = () => {
  const container = containerRef.current;
  if (!container) {
    return;
  }
  container.scroll({
    top: container.scrollHeight,
    behavior: "auto",
  });
};
  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    const onScroll = () => {
      const distanceToBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
      const atBottom = distanceToBottom <= bottomThreshold;
      isAtBottomRef.current = atBottom;
      if (atBottom && hasPending) {
        setDisplayText(pendingTextRef.current);
        setHasPending(false);
        requestAnimationFrame(scrollToBottom);
      }
    };
    container.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => {
      container.removeEventListener("scroll", onScroll);
    };
  }, [containerRef, hasPending]);

  useEffect(() => {
    if (isAtBottomRef.current) {
      setDisplayText(text);
      setHasPending(false);
      requestAnimationFrame(scrollToBottom);
    } else {
      pendingTextRef.current = text;
      if (text) {
        setHasPending(true);
      }
    }
  }, [text]);

  const handleJumpToBottom = () => {
    if (pendingTextRef.current) {
      setDisplayText(pendingTextRef.current);
    }
    setHasPending(false);
    isAtBottomRef.current = true;
    requestAnimationFrame(scrollToBottom);
  };

  return (
    <div className="relative h-full w-full max-w-full max-h-full p-2 text-white">
      <div className="whitespace-pre-wrap break-words">
        {displayText}
      </div>
      {hasPending && (
        <button
          type="button"
          className="absolute bottom-2 right-2 border border-white bg-black px-3 py-1 text-xs"
          onClick={handleJumpToBottom}
        >
          有新内容 · 回到底部
        </button>
      )}
    </div>
  );
};
