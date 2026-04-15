import { ChatPanel } from "./components/ChatPanel";
import { ActivityPanel } from "./components/ActivityPanel";
import { useEventStream } from "./hooks/useEventStream";

function App() {
  const {
    chatMessages,
    activityItems,
    isRunning,
    needsInput,
    startResearch,
    resumeResearch,
  } = useEventStream();

  const hasStarted = chatMessages.length > 0;

  return (
    <div className="h-screen flex bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      {/* Chat Panel */}
      <div className="flex-[55] border-r border-gray-200 dark:border-gray-700">
        <ChatPanel
          messages={chatMessages}
          isRunning={isRunning}
          needsInput={needsInput}
          onSend={resumeResearch}
          onStart={startResearch}
          hasStarted={hasStarted}
        />
      </div>

      {/* Activity Panel */}
      <div className="flex-[45] bg-gray-50 dark:bg-gray-950">
        <ActivityPanel items={activityItems} isRunning={isRunning} />
      </div>
    </div>
  );
}

export default App;
