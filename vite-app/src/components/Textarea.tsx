import React from "react";
import { commonStyles } from "../styles/common";

interface TextareaProps
  extends Omit<React.TextareaHTMLAttributes<HTMLTextAreaElement>, "size"> {
  size?: "sm" | "md";
  className?: string;
}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className = "", size = "sm", ...props }, ref) => {
    return (
      <textarea
        ref={ref}
        className={`${commonStyles.input.base} ${commonStyles.input.size[size]} ${className}`}
        style={{ boxShadow: commonStyles.input.shadow }}
        {...props}
      />
    );
  }
);

Textarea.displayName = "Textarea";

export default Textarea;
