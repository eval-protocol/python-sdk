import React from "react";

interface StatusIndicatorProps {
  isConnected: boolean;
  className?: string;
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  isConnected,
  className = "",
}) => {
  return (
    <div
      className={`inline-flex items-center px-2 py-0.5 border text-xs font-medium ${className}`}
      style={{ boxShadow: "none" }}
    >
      <div
        className={`w-1 h-1 border mr-1 ${
          isConnected
            ? "bg-green-500 border-green-500"
            : "bg-red-500 border-red-500"
        }`}
      />
      {isConnected ? "Connected" : "Disconnected"}
    </div>
  );
};

export default StatusIndicator;
