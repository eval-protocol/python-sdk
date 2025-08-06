import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary";
  size?: "sm" | "md";
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { className = "", variant = "secondary", size = "sm", children, ...props },
    ref
  ) => {
    const baseClasses = "border text-xs font-medium focus:outline-none";

    const variantClasses = {
      primary: "border-gray-300 bg-gray-100 text-gray-700 hover:bg-gray-200",
      secondary: "border-gray-300 bg-gray-100 text-gray-700 hover:bg-gray-200",
    };

    const sizeClasses = {
      sm: "px-2 py-0.5",
      md: "px-3 py-1",
    };

    return (
      <button
        ref={ref}
        className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
        style={{ boxShadow: "none" }}
        {...props}
      >
        {children}
      </button>
    );
  }
);

Button.displayName = "Button";

export default Button;
