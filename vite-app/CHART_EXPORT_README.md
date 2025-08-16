# Chart Export Functionality

This document describes the new Chart Export feature that allows users to export pivot table data as interactive charts and save them as high-resolution PNG images.

## Overview

The Chart Export component (`ChartExport.tsx`) integrates with Chart.js to provide visualization capabilities for pivot table data. Users can:

- Choose from multiple chart types (Bar, Line, Doughnut, Pie)
- View real-time chart updates as pivot table configuration changes
- Export charts as high-resolution PNG images
- Customize chart appearance and data representation

## Features

### Chart Types

1. **Bar Chart**: Best for comparing values across categories
2. **Line Chart**: Ideal for showing trends over time or sequences
3. **Doughnut Chart**: Good for showing proportions of a whole
4. **Pie Chart**: Similar to doughnut but shows complete proportions

### Data Visualization

- **Row-based grouping**: Row fields become chart labels
- **Column-based datasets**: Each column field combination becomes a separate dataset
- **Totals integration**: Row totals can be included as an additional dataset
- **Dynamic coloring**: Automatic color generation for different datasets
- **Responsive design**: Charts adapt to container size

### Export Capabilities

- **High-resolution output**: 2x scale for crisp images
- **PNG format**: Lossless image format suitable for presentations and reports
- **Automatic naming**: Files include chart type and timestamp
- **Background handling**: Clean white background for professional appearance

## Technical Implementation

### Dependencies

- `chart.js` (v4.5.0): Core charting library
- `react-chartjs-2` (v5.3.0): React wrapper for Chart.js
- `html2canvas` (v1.4.1): HTML to canvas conversion for image export

### Component Structure

```tsx
<ChartExport
  pivotData={pivotComputationResult}
  rowFields={selectedRowFields}
  columnFields={selectedColumnFields}
  valueField={selectedValueField}
  aggregator={selectedAggregator}
  chartType="bar"
  showRowTotals={true}
  showColumnTotals={true}
/>
```

### Data Flow

1. **Pivot Data**: Raw pivot table computation results
2. **Chart Conversion**: Data transformation for Chart.js format
3. **Rendering**: Chart display using react-chartjs-2
4. **Export**: HTML to canvas conversion and PNG download

## Usage

### Basic Setup

1. Ensure pivot table has both row and column fields selected
2. The Chart Export component will automatically appear above the pivot table
3. Select desired chart type from the dropdown
4. Click "Export as Image" to download the chart

### Chart Type Selection

- **Bar/Line**: Best for comparing multiple categories with multiple datasets
- **Pie/Doughnut**: Best for showing proportions when you have one main dimension

### Export Process

1. Click "Export as Image" button
2. Wait for processing (button shows "Exporting...")
3. Browser automatically downloads PNG file
4. File is named: `pivot-chart-{type}-{timestamp}.png`

## Integration

The component is automatically integrated into the PivotTab and only appears when:
- At least one row field is selected
- At least one column field is selected
- Valid pivot data exists

## Styling

- Follows the existing design system with minimal, clean appearance
- Uses Tailwind CSS classes for consistent styling
- Responsive design that works on different screen sizes
- Colorblind-friendly color generation using HSL color space

## Performance Considerations

- Charts are rendered only when pivot data changes
- Export process uses `html2canvas` for reliable image generation
- Chart data is memoized to prevent unnecessary re-renders
- Responsive design maintains good performance on various devices

## Browser Compatibility

- Modern browsers with ES6+ support
- Canvas API support required for image export
- File download API support required for automatic downloads

## Troubleshooting

### Common Issues

1. **Chart not appearing**: Ensure both row and column fields are selected
2. **Export fails**: Check browser console for errors, ensure canvas is properly rendered
3. **Poor image quality**: Export uses 2x scale by default for high resolution
4. **Chart data missing**: Verify pivot table configuration and data availability

### Debug Information

- Check browser console for any JavaScript errors
- Verify pivot data structure matches expected format
- Ensure all required dependencies are properly installed

## Future Enhancements

Potential improvements for future versions:
- Additional chart types (scatter, radar, etc.)
- Custom color schemes
- Chart configuration options (axes, legends, etc.)
- Multiple export formats (SVG, PDF)
- Chart templates and presets
- Batch export capabilities
