import React from "react";

interface StatusIndicatorProps {
  status: string;
  className?: string;
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  className = "",
}) => {
  const getStatusConfig = (status: string) => {
    switch (status.toLowerCase()) {
      case "connected":
        return {
          dotColor: "bg-green-500",
          textColor: "text-green-700",
          text: "Connected",
        };
      case "disconnected":
        return {
          dotColor: "bg-red-500",
          textColor: "text-red-700",
          text: "Disconnected",
        };
      case "finished":
        return {
          dotColor: "bg-green-500",
          textColor: "text-green-700",
          text: "finished",
        };
      case "running":
        return {
          dotColor: "bg-blue-500",
          textColor: "text-blue-700",
          text: "running",
        };
      case "error":
        return {
          dotColor: "bg-red-500",
          textColor: "text-red-700",
          text: "error",
        };
      default:
        return {
          dotColor: "bg-gray-500",
          textColor: "text-gray-700",
          text: status,
        };
    }
  };

  const config = getStatusConfig(status);

  return (
    <div
      className={`inline-flex items-center gap-1.5 text-xs font-medium ${config.textColor} ${className}`}
    >
      <div className={`w-1.5 h-1.5 rounded-full ${config.dotColor}`} />
      {config.text}
    </div>
  );
};

export default StatusIndicator;
