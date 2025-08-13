import React from "react";
import { commonStyles } from "../styles/common";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary";
  size?: "sm" | "md";
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { className = "", variant = "secondary", size = "sm", children, ...props },
    ref
  ) => {
    return (
      <button
        ref={ref}
        className={`${commonStyles.button.base} ${commonStyles.button.variant[variant]} ${commonStyles.button.size[size]} ${className}`}
        style={{ boxShadow: commonStyles.button.shadow }}
        {...props}
      >
        {children}
      </button>
    );
  }
);

Button.displayName = "Button";

export default Button;
