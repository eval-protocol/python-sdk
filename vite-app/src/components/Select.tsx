import React from "react";
import { commonStyles } from "../styles/common";

interface SelectProps
  extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "size"> {
  size?: "sm" | "md";
  className?: string;
}

const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className = "", size = "sm", children, ...props }, ref) => {
    return (
      <select
        ref={ref}
        className={`${commonStyles.input.base} ${commonStyles.input.size[size]} ${className}`}
        style={{ boxShadow: commonStyles.input.shadow }}
        {...props}
      >
        {children}
      </select>
    );
  }
);

Select.displayName = "Select";

export default Select;
