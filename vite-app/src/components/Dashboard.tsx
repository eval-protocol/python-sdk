import { observer } from "mobx-react";
import { state } from "../App";

const Dashboard = observer(() => {
  return (
    <div>
      Dashboard<div>{JSON.stringify(state.dataset)}</div>
    </div>
  );
});

export default Dashboard;
