import { useCallback, useRef, useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { ActivityPanel } from "./components/ActivityPanel";
import { StatePanel } from "./components/StatePanel";
import { Splitter } from "./components/Splitter";
import { useEventStream } from "./hooks/useEventStream";

function App() {
  const {
    chatMessages,
    activityItems,
    isRunning,
    needsInput,
    startResearch,
    resumeResearch,
    researchers,
    coordinator,
  } = useEventStream();

  const hasStarted = chatMessages.length > 0;

  // Resizable layout state (percentages)
  const [leftWidth, setLeftWidth] = useState(55);
  const [topHeight, setTopHeight] = useState(35);
  const containerRef = useRef<HTMLDivElement>(null);
  const leftRef = useRef<HTMLDivElement>(null);

  const handleHorizontalResize = useCallback((delta: number) => {
    if (!containerRef.current) return;
    const totalWidth = containerRef.current.offsetWidth;
    const pct = (delta / totalWidth) * 100;
    setLeftWidth((prev) => Math.min(80, Math.max(20, prev + pct)));
  }, []);

  const handleVerticalResize = useCallback((delta: number) => {
    if (!leftRef.current) return;
    const totalHeight = leftRef.current.offsetHeight;
    const pct = (delta / totalHeight) * 100;
    setTopHeight((prev) => Math.min(80, Math.max(15, prev + pct)));
  }, []);

  return (
    <div
      ref={containerRef}
      className="h-screen flex bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100"
    >
      {/* Left: Activity (top) + State (bottom) */}
      <div
        ref={leftRef}
        className="flex flex-col bg-gray-50 dark:bg-gray-950 overflow-hidden"
        style={{ width: `${leftWidth}%` }}
      >
        <div className="overflow-hidden" style={{ height: `${topHeight}%` }}>
          <ActivityPanel items={activityItems} isRunning={isRunning} />
        </div>

        <Splitter direction="vertical" onResize={handleVerticalResize} />

        <div className="flex-1 overflow-hidden">
          <StatePanel
            researchers={researchers}
            coordinator={coordinator}
            isRunning={isRunning}
          />
        </div>
      </div>

      <Splitter direction="horizontal" onResize={handleHorizontalResize} />

      {/* Right: Chat */}
      <div className="flex-1 overflow-hidden">
        <ChatPanel
          messages={chatMessages}
          isRunning={isRunning}
          needsInput={needsInput}
          onSend={resumeResearch}
          onStart={startResearch}
          hasStarted={hasStarted}
        />
      </div>
    </div>
  );
}

export default App;
