// Common styling classes for consistent component appearance
export const commonStyles = {
  // Base input/select styling (matching Select component)
  input: {
    base: "border text-xs font-medium focus:outline-none bg-white text-gray-700 border-gray-300 hover:border-gray-400 focus:border-gray-500",
    size: {
      sm: "px-2 py-0.5",
      md: "px-3 py-1",
    },
    shadow: "none", // boxShadow: "none"
  },

  // Button styling (matching Button component)
  button: {
    base: "border text-xs font-medium focus:outline-none",
    variant: {
      primary: "border-gray-300 bg-gray-100 text-gray-700 hover:bg-gray-200",
      secondary: "border-gray-300 bg-gray-100 text-gray-700 hover:bg-gray-200",
    },
    size: {
      sm: "px-2 py-0.5",
      md: "px-3 py-1",
    },
    shadow: "none", // boxShadow: "none"
  },

  // Common spacing and layout
  spacing: {
    sm: "space-y-2",
    md: "space-y-4",
    lg: "space-y-6",
  },

  // Common widths
  width: {
    sm: "min-w-32",
    md: "min-w-48",
    lg: "min-w-64",
  },
} as const;
