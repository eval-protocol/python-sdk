import React from "react";

export interface TableContainerProps {
  /**
   * The table content to be wrapped
   */
  children: React.ReactNode;
  /**
   * Optional extra class names applied to the wrapping container
   */
  className?: string;
}

export interface TableHeaderProps {
  /**
   * The header content
   */
  children: React.ReactNode;
  /**
   * Optional extra class names
   */
  className?: string;
  /**
   * Text alignment. Default: "left"
   */
  align?: "left" | "center" | "right";
  /**
   * Whether to apply whitespace-nowrap. Default: false
   */
  nowrap?: boolean;
}

export interface TableCellProps {
  /**
   * The cell content
   */
  children: React.ReactNode;
  /**
   * Optional extra class names
   */
  className?: string;
  /**
   * Text alignment. Default: "left"
   */
  align?: "left" | "center" | "right";
  /**
   * Whether to apply whitespace-nowrap. Default: false
   */
  nowrap?: boolean;
  /**
   * Whether to apply font-medium styling. Default: false
   */
  medium?: boolean;
  /**
   * Whether to apply font-semibold styling. Default: false
   */
  semibold?: boolean;
  /**
   * Whether to apply text-xs styling. Default: false
   */
  small?: boolean;
  /**
   * Number of columns this cell should span
   */
  colSpan?: number;
}

export interface TableRowProps {
  /**
   * The row content
   */
  children: React.ReactNode;
  /**
   * Optional extra class names
   */
  className?: string;
  /**
   * Whether to apply gray background styling. Default: false
   */
  gray?: boolean;
}

export interface TableRowInteractiveProps {
  /**
   * The row content
   */
  children: React.ReactNode;
  /**
   * Optional extra class names
   */
  className?: string;
  /**
   * Click handler for the row
   */
  onClick?: () => void;
  /**
   * Whether the row is interactive (hoverable, clickable). Default: true
   */
  interactive?: boolean;
}

export interface TableBodyProps {
  /**
   * The body content
   */
  children: React.ReactNode;
  /**
   * Optional extra class names
   */
  className?: string;
}

export interface TableHeadProps {
  /**
   * The head content
   */
  children: React.ReactNode;
  /**
   * Optional extra class names
   */
  className?: string;
}

/**
 * Wrapper component that provides consistent table styling across the application.
 * Applies white background, gray borders, and horizontal scroll for overflow.
 */
export function TableContainer({
  children,
  className = "",
}: TableContainerProps) {
  return (
    <div
      className={`bg-white border border-gray-200 overflow-x-auto ${className}`}
    >
      {children}
    </div>
  );
}

/**
 * Table head component with consistent styling
 */
export function TableHead({ children, className = "" }: TableHeadProps) {
  return (
    <thead className={`bg-gray-50 border-b border-gray-200 ${className}`}>
      {children}
    </thead>
  );
}

/**
 * Table body component with consistent styling
 */
export function TableBody({ children, className = "" }: TableBodyProps) {
  return (
    <tbody className={`divide-y divide-gray-200 ${className}`}>
      {children}
    </tbody>
  );
}

/**
 * Table header component with consistent styling
 */
export function TableHeader({
  children,
  className = "",
  align = "left",
  nowrap = false,
}: TableHeaderProps) {
  const alignClasses = {
    left: "text-left",
    center: "text-center",
    right: "text-right",
  };

  return (
    <th
      className={`px-3 py-2 text-xs font-semibold text-gray-700 ${
        alignClasses[align]
      } ${nowrap ? "whitespace-nowrap" : ""} ${className}`}
    >
      {children}
    </th>
  );
}

/**
 * Table row component with consistent styling
 */
export function TableRow({
  children,
  className = "",
  gray = false,
}: TableRowProps) {
  return (
    <tr className={`${gray ? "bg-gray-50" : ""} ${className}`}>{children}</tr>
  );
}

/**
 * Interactive table row component with hover effects and click handling
 */
export function TableRowInteractive({
  children,
  className = "",
  onClick,
  interactive = true,
}: TableRowInteractiveProps) {
  const interactiveClasses = interactive
    ? "hover:bg-gray-50 cursor-pointer"
    : "";

  return (
    <tr
      className={`text-sm ${interactiveClasses} ${className}`}
      onClick={onClick}
    >
      {children}
    </tr>
  );
}

/**
 * Table cell component with consistent styling
 */
export function TableCell({
  children,
  className = "",
  align = "left",
  nowrap = false,
  medium = false,
  semibold = false,
  colSpan,
}: TableCellProps) {
  const alignClasses = {
    left: "text-left",
    center: "text-center",
    right: "text-right",
  };

  const fontClasses = [];
  if (medium) fontClasses.push("font-medium");
  if (semibold) fontClasses.push("font-semibold");

  return (
    <td
      colSpan={colSpan}
      className={`px-3 py-2 text-gray-900 ${alignClasses[align]} ${
        nowrap ? "whitespace-nowrap" : ""
      } ${fontClasses.join(" ")} ${className}`}
    >
      {children}
    </td>
  );
}

export default TableContainer;
