import React from "react";
import { commonStyles } from "../styles/common";

interface TextareaProps
  extends Omit<React.TextareaHTMLAttributes<HTMLTextAreaElement>, "size"> {
  size?: "sm" | "md";
  className?: string;
}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className = "", size = "sm", disabled = false, ...props }, ref) => {
    const disabledStyles = disabled
      ? "bg-gray-50 text-gray-300 border-gray-200 cursor-not-allowed opacity-60"
      : "";

    return (
      <textarea
        ref={ref}
        className={`${commonStyles.input.base} ${commonStyles.input.size[size]} ${disabledStyles} ${className}`}
        style={{ boxShadow: commonStyles.input.shadow }}
        disabled={disabled}
        {...props}
      />
    );
  }
);

Textarea.displayName = "Textarea";

export default Textarea;
