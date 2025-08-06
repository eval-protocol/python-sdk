import { makeAutoObservable } from "mobx";
import type { EvaluationRow } from "./types/eval-protocol";

export class GlobalState {
  isConnected: boolean = false;
  dataset: EvaluationRow[] = [];
  constructor() {
    makeAutoObservable(this);
  }

  setDataset(dataset: EvaluationRow[]) {
    this.dataset = dataset;
  }
}
