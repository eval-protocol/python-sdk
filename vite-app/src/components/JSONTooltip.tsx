import React from "react";
import { Tooltip } from "react-tooltip";

interface JSONTooltipProps {
  children: React.ReactNode;
  data: any;
  position?: "top" | "bottom" | "left" | "right";
  className?: string;
}

export const JSONTooltip: React.FC<JSONTooltipProps> = ({
  children,
  data,
  position = "top",
  className = "",
}) => {
  const tooltipId = `json-tooltip-${Math.random().toString(36).substr(2, 9)}`;
  const formattedJSON = JSON.stringify(data, null, 2);

  return (
    <>
      <div data-tooltip-id={tooltipId} className={`cursor-help ${className}`}>
        {children}
      </div>
      <Tooltip
        id={tooltipId}
        place={position}
        className="px-2 py-1 text-xs text-white bg-gray-800 rounded z-10 max-w-md"
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
        delayShow={200}
        delayHide={300}
        clickable={true}
        noArrow={true}
        render={() => (
          <pre
            className="whitespace-pre-wrap text-left text-xs"
            style={{
              userSelect: "text",
              pointerEvents: "auto",
              cursor: "text",
            }}
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => e.stopPropagation()}
            onDoubleClick={(e) => e.stopPropagation()}
            onContextMenu={(e) => e.stopPropagation()}
          >
            {formattedJSON}
          </pre>
        )}
      />
    </>
  );
};
