import { useState, useRef, useEffect } from "react";
import type { Message } from "../types/eval-protocol";
import { MessageBubble } from "./MessageBubble";

interface ChatInterfaceProps {
  messages: Message[];
}

export const ChatInterface = ({ messages }: ChatInterfaceProps) => {
  const [chatWidth, setChatWidth] = useState(600); // Default width in pixels
  const [chatHeight, setChatHeight] = useState(512); // Default height in pixels (32rem = 512px)
  const [isResizingWidth, setIsResizingWidth] = useState(false);
  const [isResizingHeight, setIsResizingHeight] = useState(false);
  const [initialWidth, setInitialWidth] = useState(0);
  const [initialHeight, setInitialHeight] = useState(0);
  const [initialMouseX, setInitialMouseX] = useState(0);
  const [initialMouseY, setInitialMouseY] = useState(0);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const resizeHandleRef = useRef<HTMLDivElement>(null);
  const heightResizeHandleRef = useRef<HTMLDivElement>(null);

  // Handle horizontal resizing
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isResizingWidth) {
        e.preventDefault();
        const deltaX = e.clientX - initialMouseX;
        const newWidth = initialWidth + deltaX;

        // Calculate max width as 66% of available width
        // Get the parent container that has the flex layout
        const parentContainer = chatContainerRef.current?.closest(".flex");
        const containerWidth =
          parentContainer?.clientWidth || window.innerWidth;
        const maxWidth = containerWidth * 0.66;

        setChatWidth(Math.max(300, Math.min(maxWidth, newWidth))); // Min 300px, max 70% of container
      }
    };

    const handleMouseUp = () => {
      setIsResizingWidth(false);
    };

    if (isResizingWidth) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingWidth, initialMouseX, initialWidth]);

  // Handle vertical resizing
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isResizingHeight) {
        e.preventDefault();
        const deltaY = e.clientY - initialMouseY;
        const newHeight = initialHeight + deltaY;
        setChatHeight(Math.max(200, Math.min(800, newHeight))); // Min 200px, max 800px
      }
    };

    const handleMouseUp = () => {
      setIsResizingHeight(false);
    };

    if (isResizingHeight) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingHeight, initialMouseY, initialHeight]);

  const startWidthResize = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setInitialMouseX(e.clientX);
    setInitialWidth(chatWidth);
    setIsResizingWidth(true);
  };

  const startHeightResize = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setInitialMouseY(e.clientY);
    setInitialHeight(chatHeight);
    setIsResizingHeight(true);
  };

  const startCornerResize = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setInitialMouseX(e.clientX);
    setInitialMouseY(e.clientY);
    setInitialWidth(chatWidth);
    setInitialHeight(chatHeight);
    setIsResizingWidth(true);
    setIsResizingHeight(true);
  };

  return (
    <div
      ref={chatContainerRef}
      className="relative"
      style={{ width: `${chatWidth}px` }}
    >
      <div
        className="bg-white border border-gray-200 p-4 overflow-y-auto relative"
        style={{ height: `${chatHeight}px` }}
      >
        {messages.map((message, msgIndex) => (
          <MessageBubble key={msgIndex} message={message} />
        ))}
      </div>

      {/* Vertical resize handle - positioned outside the scrollable container */}
      <div
        ref={heightResizeHandleRef}
        className="absolute left-0 w-full h-1 bg-gray-300 cursor-row-resize hover:bg-gray-400 transition-colors select-none"
        style={{ top: `${chatHeight}px` }}
        onMouseDown={startHeightResize}
        onDragStart={(e) => e.preventDefault()}
      />

      {/* Horizontal resize handle */}
      <div
        ref={resizeHandleRef}
        className="absolute top-0 right-0 w-1 bg-gray-300 cursor-col-resize hover:bg-gray-400 transition-colors select-none"
        style={{ height: `${chatHeight}px` }}
        onMouseDown={startWidthResize}
        onDragStart={(e) => e.preventDefault()}
      />

      {/* Corner resize handle - positioned outside the scrollable container */}
      <div
        className="absolute w-3 h-3 bg-gray-300 cursor-nw-resize hover:bg-gray-400 transition-colors select-none"
        style={{ top: `${chatHeight - 8}px`, right: "0px" }}
        onMouseDown={startCornerResize}
        onDragStart={(e) => e.preventDefault()}
      />
    </div>
  );
};
