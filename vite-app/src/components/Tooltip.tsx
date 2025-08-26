import React from "react";
import { Tooltip as ReactTooltip } from "react-tooltip";

interface TooltipProps {
  children: React.ReactNode;
  content: string;
  position?: "top" | "bottom" | "left" | "right";
  className?: string;
}

export const Tooltip: React.FC<TooltipProps> = ({
  children,
  content,
  position = "top",
  className = "",
}) => {
  const tooltipId = `tooltip-${Math.random().toString(36).substr(2, 9)}`;

  return (
    <>
      <div
        data-tooltip-id={tooltipId}
        className={`cursor-pointer ${className}`}
      >
        {children}
      </div>
      <ReactTooltip
        id={tooltipId}
        place={position}
        className="px-2 py-1 text-xs text-white bg-gray-800 rounded z-10"
        style={{
          fontSize: "0.75rem",
          lineHeight: "1rem",
          backgroundColor: "#1f2937",
          color: "white",
          borderRadius: "0.25rem",
          padding: "0.5rem",
          zIndex: 10,
          userSelect: "text",
          pointerEvents: "auto",
        }}
        noArrow={true}
      >
        {content}
      </ReactTooltip>
    </>
  );
};
