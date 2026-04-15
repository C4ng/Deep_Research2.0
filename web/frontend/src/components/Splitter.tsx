import { useCallback, useEffect, useRef } from "react";

interface SplitterProps {
  direction: "horizontal" | "vertical";
  onResize: (delta: number) => void;
}

export function Splitter({ direction, onResize }: SplitterProps) {
  const dragging = useRef(false);
  const lastPos = useRef(0);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      dragging.current = true;
      lastPos.current = direction === "horizontal" ? e.clientX : e.clientY;
      document.body.style.cursor =
        direction === "horizontal" ? "col-resize" : "row-resize";
      document.body.style.userSelect = "none";
    },
    [direction]
  );

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!dragging.current) return;
      const pos = direction === "horizontal" ? e.clientX : e.clientY;
      const delta = pos - lastPos.current;
      if (delta !== 0) {
        onResize(delta);
        lastPos.current = pos;
      }
    };

    const handleMouseUp = () => {
      if (dragging.current) {
        dragging.current = false;
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [direction, onResize]);

  const isH = direction === "horizontal";

  return (
    <div
      onMouseDown={handleMouseDown}
      className={`${
        isH ? "w-1 cursor-col-resize" : "h-1 cursor-row-resize"
      } bg-gray-200 dark:bg-gray-700 hover:bg-blue-400 dark:hover:bg-blue-500 transition-colors shrink-0`}
    />
  );
}
