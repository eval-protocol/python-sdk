import React from "react";
import { getStatusCodeName, type Status } from "../types/eval-protocol";

interface StatusIndicatorProps {
  status: Status;
  className?: string;
  showSpinner?: boolean;
}

const Spinner: React.FC<{ color: string }> = ({ color }) => (
  <div
    className={`animate-spin w-1.5 h-1.5 rounded-full border border-current ${color} border-t-transparent`}
  />
);

const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  className = "",
  showSpinner = false,
}) => {
  const getStatusConfig = (status: Status) => {
    const statusCodeName = getStatusCodeName(status.code);

    switch (statusCodeName) {
      case "OK":
        return {
          dotColor: "bg-green-500",
          textColor: "text-green-700",
          text: "Connected",
        };
      case "CANCELLED":
        return {
          dotColor: "bg-red-500",
          textColor: "text-red-700",
          text: "Disconnected",
        };
      case "FINISHED":
        return {
          dotColor: "bg-green-500",
          textColor: "text-green-700",
          text: "finished",
        };
      case "RUNNING":
        return {
          dotColor: "bg-blue-500",
          textColor: "text-blue-700",
          text: "running",
        };
      case "INTERNAL":
        return {
          dotColor: "bg-red-500",
          textColor: "text-red-700",
          text: "error",
        };
      case "ABORTED":
        return {
          dotColor: "bg-yellow-500",
          textColor: "text-yellow-700",
          text: "stopped",
        };
      default:
        return {
          dotColor: "bg-gray-500",
          textColor: "text-gray-700",
          text: status.message,
        };
    }
  };

  const config = getStatusConfig(status);
  const shouldShowSpinner = showSpinner && status.code === 101;

  return (
    <div
      className={`inline-flex items-center gap-1.5 text-xs font-medium ${config.textColor} ${className}`}
    >
      {shouldShowSpinner ? (
        <Spinner color={config.textColor} />
      ) : (
        <div className={`w-1.5 h-1.5 rounded-full ${config.dotColor}`} />
      )}
      {config.text}
    </div>
  );
};

export default StatusIndicator;
