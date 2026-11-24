import { ArrowLeftOutlined, StopOutlined, RedoOutlined } from "@ant-design/icons";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Alert,
  App as AntdApp,
  Button,
  Card,
  Descriptions,
  List,
  Progress,
  Space,
  Tag,
  Typography,
} from "antd";
import dayjs from "dayjs";
import { useTranslation } from "react-i18next";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import HtmlArtifactFrame from "../components/HtmlArtifactFrame";
import StatusTag from "../components/StatusTag";
import type { LogEntry, RunMessage } from "../types";

const { Title, Paragraph, Text } = Typography;

function isActive(status: string | undefined): boolean {
  return status === "queued" || status === "running";
}

export default function RunDetailPage() {
  const { t } = useTranslation();
  const { runId = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { message } = AntdApp.useApp();

  const runQuery = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRun(runId),
    enabled: Boolean(runId),
    refetchInterval: (query) =>
      isActive(query.state.data?.status) ? 1500 : false,
  });

  const logsQuery = useQuery({
    queryKey: ["logs", runId],
    queryFn: () => api.logs(runId),
    enabled: Boolean(runId),
    refetchInterval: () => (isActive(runQuery.data?.status) ? 1500 : false),
  });

  const cancelMutation = useMutation({
    mutationFn: () => api.cancelRun(runId),
    onSuccess: () => {
      message.success(t("run.cancelled"));
      void queryClient.invalidateQueries({ queryKey: ["run", runId] });
    },
    onError: () => message.error(t("common.error")),
  });

  const retryMutation = useMutation({
    mutationFn: () => api.retryRun(runId),
    onSuccess: (created) => {
      void queryClient.invalidateQueries({ queryKey: ["runs"] });
      navigate(`/runs/${created.id}`);
    },
    onError: () => message.error(t("common.error")),
  });

  if (runQuery.isLoading) {
    return <Text type="secondary">{t("common.loading")}</Text>;
  }
  if (runQuery.isError || !runQuery.data) {
    return <Alert type="error" showIcon message={t("common.error")} />;
  }

  const run = runQuery.data;
  const messageParams = run.message.params as Record<string, string | number>;
  const translatedMessage = run.message.key
    ? t(run.message.key, messageParams)
    : "";

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Space>
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate("/runs")}>
          {t("run.back")}
        </Button>
      </Space>

      <Card>
        <Space direction="vertical" size={10} style={{ width: "100%" }}>
          <Space wrap>
            <Title level={2} style={{ margin: 0 }}>
              {run.task_key}
            </Title>
            <StatusTag status={run.status} />
            <Tag>{run.module}</Tag>
          </Space>
          <Paragraph type="secondary" style={{ margin: 0 }}>
            {t("run.shareHint")}
          </Paragraph>
          <Progress
            percent={run.progress}
            status={
              run.status === "failed"
                ? "exception"
                : run.status === "succeeded"
                  ? "success"
                  : "active"
            }
          />
          {translatedMessage ? (
            <Text className="run-message">{translatedMessage}</Text>
          ) : null}
          {run.stage_key ? (
            <Text type="secondary">
              {t("run.stage")}: {run.stage_key}
            </Text>
          ) : null}
          <Space wrap>
            {isActive(run.status) ? (
              <Button
                danger
                icon={<StopOutlined />}
                loading={cancelMutation.isPending}
                onClick={() => cancelMutation.mutate()}
              >
                {t("run.cancel")}
              </Button>
            ) : (
              <Button
                icon={<RedoOutlined />}
                loading={retryMutation.isPending}
                onClick={() => retryMutation.mutate()}
              >
                {t("run.retry")}
              </Button>
            )}
          </Space>
        </Space>
      </Card>

      {run.error_text ? (
        <Alert
          type="error"
          showIcon
          message={t("run.error")}
          description={
            <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{run.error_text}</pre>
          }
        />
      ) : null}

      <Card size="small" title={t("run.params")}>
        <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
          {JSON.stringify(run.params, null, 2)}
        </pre>
      </Card>

      {run.inputs.length ? (
        <Card size="small" title={t("run.inputs")}>
          <List
            dataSource={run.inputs}
            renderItem={(item) => (
              <List.Item>
                <Space>
                  <Tag>{item.role}</Tag>
                  <span>{item.filename}</span>
                </Space>
              </List.Item>
            )}
          />
        </Card>
      ) : null}

      <Card size="small" title={t("run.artifacts")}>
        {run.artifacts.length ? (
          <List
            dataSource={run.artifacts}
            renderItem={(artifact) => (
              <List.Item
                actions={[
                  <Button
                    key="download"
                    type="link"
                    href={api.artifactUrl(artifact.id, "download")}
                  >
                    {t("artifact.download")}
                  </Button>,
                ]}
              >
                <List.Item.Meta
                  title={
                    <Space>
                      <span>{artifact.logical_name}</span>
                      {artifact.is_primary ? (
                        <Tag color="blue">{t("artifact.primary")}</Tag>
                      ) : null}
                    </Space>
                  }
                  description={`${artifact.kind} · ${artifact.size_bytes} B`}
                />
                {artifact.kind === "html" ? (
                  <div style={{ width: "100%", marginTop: 12 }}>
                    <HtmlArtifactFrame artifactId={artifact.id} />
                  </div>
                ) : artifact.kind === "png" || artifact.kind === "svg" ? (
                  <img
                    src={api.artifactUrl(artifact.id, "content")}
                    alt={artifact.logical_name}
                    style={{ maxWidth: "100%", marginTop: 12 }}
                  />
                ) : null}
              </List.Item>
            )}
          />
        ) : (
          <Text type="secondary">{t("run.noArtifacts")}</Text>
        )}
      </Card>

      <Card size="small" title={t("run.logs")}>
        <div className="log-list">
          {(logsQuery.data ?? []).map((entry: LogEntry) => (
            <div
              key={entry.seq}
              className={
                entry.level === "error"
                  ? "log-list__entry log-list__entry--error"
                  : "log-list__entry"
              }
            >
              {entry.message_key
                ? t(
                    entry.message_key,
                    entry.message_params as Record<string, string | number>,
                  )
                : entry.raw_text}
            </div>
          ))}
        </div>
      </Card>

      <Descriptions size="small" column={1} bordered>
        <Descriptions.Item label={t("column.created")}>
          {dayjs(run.created_at).format("YYYY-MM-DD HH:mm:ss")}
        </Descriptions.Item>
        <Descriptions.Item label="Started">
          {run.started_at ? dayjs(run.started_at).format("YYYY-MM-DD HH:mm:ss") : "—"}
        </Descriptions.Item>
        <Descriptions.Item label="Finished">
          {run.finished_at ? dayjs(run.finished_at).format("YYYY-MM-DD HH:mm:ss") : "—"}
        </Descriptions.Item>
      </Descriptions>
    </Space>
  );
}
