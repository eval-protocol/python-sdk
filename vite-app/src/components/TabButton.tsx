import React from "react";

interface TabButtonProps {
  label: string;
  isActive: boolean;
  onClick: () => void;
  title?: string;
}

const TabButton: React.FC<TabButtonProps> = ({
  label,
  isActive,
  onClick,
  title,
}) => {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={isActive}
      title={title}
      onClick={onClick}
      className={`text-xs font-medium px-2 py-0.5 border-b-2 focus:outline-none cursor-pointer transition-colors ${
        isActive
          ? "text-gray-900 border-gray-900 bg-transparent"
          : "text-gray-700 hover:text-gray-900 hover:border-gray-400 border-transparent bg-transparent hover:bg-gray-100"
      }`}
    >
      {label}
    </button>
  );
};

export default TabButton;
