import React from "react";

interface SelectProps
  extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "size"> {
  size?: "sm" | "md";
  className?: string;
}

const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className = "", size = "sm", children, ...props }, ref) => {
    const baseClasses =
      "border text-xs font-medium focus:outline-none bg-white text-gray-700 border-gray-300 hover:border-gray-400 focus:border-gray-500";

    const sizeClasses = {
      sm: "px-2 py-0.5",
      md: "px-3 py-1",
    };

    return (
      <select
        ref={ref}
        className={`${baseClasses} ${sizeClasses[size]} ${className}`}
        style={{ boxShadow: "none" }}
        {...props}
      >
        {children}
      </select>
    );
  }
);

Select.displayName = "Select";

export default Select;
