import { observer } from "mobx-react";
import PivotTable from "./PivotTable";

interface PivotTabProps {
  data: any[];
}

const PivotTab = observer(({ data }: PivotTabProps) => {
  return (
    <div>
      <div className="text-xs text-gray-600 mb-2">
        Showing pivot of flattened rows (JSONPath keys). Defaults: rows by eval
        name and status; columns by model; values average score.
      </div>
      <PivotTable
        data={data}
        rowFields={[
          "$.eval_metadata.name" as keyof (typeof data)[number],
          "$.eval_metadata.status" as keyof (typeof data)[number],
        ]}
        columnFields={[
          "$.input_metadata.completion_params.model" as keyof (typeof data)[number],
        ]}
        valueField={"$.evaluation_result.score" as keyof (typeof data)[number]}
        aggregator="avg"
        showRowTotals
        showColumnTotals
      />
    </div>
  );
});

export default PivotTab;
