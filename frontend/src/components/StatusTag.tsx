import { Tag } from "antd";
import { useTranslation } from "react-i18next";
import type { RunStatus } from "../types";

const COLORS: Record<RunStatus, string> = {
  queued: "default",
  running: "processing",
  succeeded: "success",
  failed: "error",
  cancelled: "warning",
};

export default function StatusTag({ status }: { status: RunStatus }) {
  const { t } = useTranslation();
  return <Tag color={COLORS[status]}>{t(`status.${status}`)}</Tag>;
}
