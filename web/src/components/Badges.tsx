import { EXP_ICON, EXP_LABEL, STATUS_ICON, STATUS_LABEL } from "../format";
import type { ExperimentStatus, RunStatus } from "../data/types";

export const StatusBadge = ({ status }: { status: RunStatus }) => (
  <span className={`badge badge-${status}`}>
    {STATUS_ICON[status]} {STATUS_LABEL[status]}
  </span>
);

export const ExpBadge = ({ status }: { status: ExperimentStatus }) => (
  <span className={`badge badge-exp-${status}`}>
    {EXP_ICON[status]} {EXP_LABEL[status]}
  </span>
);
