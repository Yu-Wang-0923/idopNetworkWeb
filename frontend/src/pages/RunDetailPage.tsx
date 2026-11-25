import { ArrowLeftOutlined, StopOutlined, RedoOutlined } from "@ant-design/icons";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import {
  Alert,
  App as AntdApp,
  Button,
  Card,
  Descriptions,
  Image,
  List,
  Modal,
  Progress,
  Select,
  Space,
  Table,
  Tag,
  Typography,
} from "antd";
import dayjs from "dayjs";
import { useTranslation } from "react-i18next";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import HtmlArtifactFrame from "../components/HtmlArtifactFrame";
import StatusTag from "../components/StatusTag";
import type { Artifact, BicSummary, LogEntry, RunMessage } from "../types";

const { Title, Paragraph, Text } = Typography;

function formatCell(
  value: string | number | boolean | null,
): string | number {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "boolean") {
    return value ? "True" : "False";
  }
  if (typeof value === "number" && !Number.isInteger(value)) {
    return Number(value.toPrecision(6));
  }
  return value;
}

function isActive(status: string | undefined): boolean {
  return status === "queued" || status === "running";
}

function chartPriority(artifact: Artifact): number {
  if (artifact.semantic_type === "netrecon_multilayer_overview_html") return 0;
  if (artifact.semantic_type === "netrecon_network_html") return 1;
  if (artifact.semantic_type === "netrecon_effect_html") return 2;
  if (artifact.semantic_type === "netrecon_heatmap_html") return 3;
  if (artifact.logical_name.includes("compare")) return 9;
  return 4;
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

  const [selectedChart, setSelectedChart] = useState<string | null>(null);
  const [previewArtifact, setPreviewArtifact] = useState<Artifact | null>(null);
  const tableQuery = useQuery({
    queryKey: ["artifact-table", previewArtifact?.id],
    queryFn: () => api.artifactTable(previewArtifact?.id ?? ""),
    enabled: Boolean(previewArtifact),
  });
  const runStatus = runQuery.data?.status;
  const artifactList = runQuery.data?.artifacts;
  const bicSummaryId = artifactList?.find(
    (item) => item.semantic_type === "funclu_bic_summary",
  )?.id;
  const bicSummaryQuery = useQuery({
    queryKey: ["artifact-json", bicSummaryId],
    queryFn: () => api.artifactJson<BicSummary>(bicSummaryId ?? ""),
    enabled: Boolean(bicSummaryId),
    staleTime: Infinity,
  });
  useEffect(() => {
    if (!artifactList) {
      return;
    }
    const charts = artifactList
      .filter((item) => item.kind === "html")
      .sort(
        (a, b) =>
          chartPriority(a) - chartPriority(b) ||
          a.logical_name.localeCompare(b.logical_name),
      );
    if (charts.length && !charts.some((item) => item.id === selectedChart)) {
      setSelectedChart(charts[0].id);
    }
  }, [artifactList, selectedChart]);

  useEffect(() => {
    if (runStatus && !isActive(runStatus)) {
      void queryClient.invalidateQueries({ queryKey: ["logs", runId] });
    }
  }, [runStatus, runId, queryClient]);

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

  const htmlArtifacts = run.artifacts
    .filter((item) => item.kind === "html")
    .sort(
      (a, b) =>
        chartPriority(a) - chartPriority(b) ||
        a.logical_name.localeCompare(b.logical_name),
    );
  const pngArtifacts = run.artifacts.filter((item) => item.kind === "png");
  const svgArtifacts = run.artifacts.filter((item) => item.kind === "svg");
  const leadOrder = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
  const leadRank = (name: string) => {
    const match = name.match(/\/([^/]+)_fitting\.(?:png|svg)$/);
    const index = match ? leadOrder.indexOf(match[1]) : -1;
    return index === -1 ? leadOrder.length : index;
  };
  const previewImages = (pngArtifacts.length ? pngArtifacts : svgArtifacts)
    .slice()
    .sort(
      (a, b) =>
        leadRank(a.logical_name) - leadRank(b.logical_name) ||
        a.logical_name.localeCompare(b.logical_name),
    );
  const downloadArtifacts = run.artifacts.filter(
    (item) => !["html", "png", "svg"].includes(item.kind),
  );
  const chartArtifact =
    htmlArtifacts.find((item) => item.id === selectedChart) ?? htmlArtifacts[0];

  const bicSummary = bicSummaryQuery.data;
  // Summaries written before boundary_side existed only carry selected_at_boundary.
  const bicBoundary =
    bicSummary?.boundary_side !== undefined
      ? bicSummary.boundary_side
      : bicSummary?.selected_at_boundary
        ? bicSummary.recommended_k === bicSummary.k_max
          ? "upper"
          : "lower"
        : null;
  const bicWarnings: string[] = [];
  if (bicSummary && bicBoundary) {
    bicWarnings.push(
      t(`funclu.message.bic_boundary_${bicBoundary}`, { k: bicSummary.recommended_k }),
    );
  }
  if (bicSummary?.min_size_violated) {
    bicWarnings.push(
      t("funclu.message.bic_min_size_violated", {
        min_size: bicSummary.params?.min_cluster_size ?? 1,
      }),
    );
  }

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
              {t("run.stage")}: {t(run.stage_key)}
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

      {bicSummary ? (
        <Alert
          type={bicWarnings.length ? "warning" : "success"}
          showIcon
          message={t("run.bicRecommended", {
            model: bicSummary.selected_model,
            k: bicSummary.recommended_k,
            kmin: bicSummary.k_values[0],
            kmax: bicSummary.k_values[bicSummary.k_values.length - 1],
          })}
          description={
            bicWarnings.length ? (
              <Space direction="vertical" size={2}>
                {bicWarnings.map((text) => (
                  <span key={text}>{text}</span>
                ))}
              </Space>
            ) : undefined
          }
        />
      ) : null}

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

      {htmlArtifacts.length || previewImages.length ? (
        <Card size="small" title={t("run.figures")}>
          {htmlArtifacts.length ? (
            <Space direction="vertical" size={10} style={{ width: "100%" }}>
              <Space wrap>
                <Select
                  value={chartArtifact?.id}
                  onChange={(value) => setSelectedChart(value)}
                  options={htmlArtifacts.map((item) => ({
                    value: item.id,
                    label: item.logical_name,
                  }))}
                  style={{ minWidth: 280, maxWidth: "100%" }}
                />
                {chartArtifact ? (
                  <>
                    <Button
                      href={api.artifactUrl(chartArtifact.id, "content")}
                      target="_blank"
                    >
                      {t("artifact.openTab")}
                    </Button>
                    <Button href={api.artifactUrl(chartArtifact.id, "download")}>
                      {t("artifact.download")}
                    </Button>
                  </>
                ) : null}
              </Space>
              {chartArtifact ? (
                <HtmlArtifactFrame
                  artifactId={chartArtifact.id}
                  height={
                    chartArtifact.semantic_type === "netrecon_network_html" ||
                    chartArtifact.semantic_type ===
                      "netrecon_multilayer_overview_html"
                      ? 950
                      : 820
                  }
                  title={chartArtifact.logical_name}
                />
              ) : null}
            </Space>
          ) : null}
          {previewImages.length ? (
            <Image.PreviewGroup>
              <div className="figure-grid" style={{ marginTop: htmlArtifacts.length ? 14 : 0 }}>
                {previewImages.map((artifact) => (
                  <div className="figure-tile" key={artifact.id}>
                    <Image
                      src={api.artifactUrl(artifact.id, "content")}
                      alt={artifact.logical_name}
                      height={120}
                      style={{
                        objectFit: "contain",
                        background: "#fff",
                        border: "1px solid #e3ebf5",
                        borderRadius: 6,
                      }}
                    />
                    <Text
                      ellipsis={{ tooltip: artifact.logical_name }}
                      style={{ fontSize: 12 }}
                    >
                      {artifact.logical_name}
                    </Text>
                    <Button
                      type="link"
                      size="small"
                      href={api.artifactUrl(artifact.id, "download")}
                    >
                      {t("artifact.download")}
                    </Button>
                  </div>
                ))}
              </div>
            </Image.PreviewGroup>
          ) : null}
        </Card>
      ) : null}

      {downloadArtifacts.length ? (
        <Card size="small" title={t("run.downloads")}>
          <List
            dataSource={downloadArtifacts}
            renderItem={(artifact) => (
              <List.Item
                actions={[
                  ...(artifact.kind === "csv"
                    ? [
                        <Button
                          key="preview"
                          type="link"
                          onClick={() => setPreviewArtifact(artifact)}
                        >
                          {t("artifact.previewTable")}
                        </Button>,
                      ]
                    : []),
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
              </List.Item>
            )}
          />
        </Card>
      ) : !htmlArtifacts.length && !previewImages.length ? (
        <Card size="small" title={t("run.artifacts")}>
          <Text type="secondary">{t("run.noArtifacts")}</Text>
        </Card>
      ) : null}

      <Card size="small" title={t("run.logs")}>
        <div className="log-list">
          {(logsQuery.data ?? []).map((entry: LogEntry) => (
            <div
              key={entry.seq}
              className={
                entry.level === "error" || entry.level === "warning"
                  ? `log-list__entry log-list__entry--${entry.level}`
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
        <Descriptions.Item label={t("run.started")}>
          {run.started_at ? dayjs(run.started_at).format("YYYY-MM-DD HH:mm:ss") : "—"}
        </Descriptions.Item>
        <Descriptions.Item label={t("run.finished")}>
          {run.finished_at ? dayjs(run.finished_at).format("YYYY-MM-DD HH:mm:ss") : "—"}
        </Descriptions.Item>
      </Descriptions>

      <Modal
        open={Boolean(previewArtifact)}
        onCancel={() => setPreviewArtifact(null)}
        footer={null}
        width={1000}
        title={previewArtifact?.logical_name}
      >
        {tableQuery.isLoading ? (
          <Text type="secondary">{t("common.loading")}</Text>
        ) : tableQuery.isError || !tableQuery.data ? (
          <Alert type="error" showIcon message={t("common.error")} />
        ) : (
          <Table
            size="small"
            bordered
            pagination={false}
            scroll={{ x: "max-content", y: 420 }}
            columns={tableQuery.data.columns.map((name, index) => ({
              title: name,
              key: `${name}-${index}`,
              render: (
                _value: unknown,
                record: {
                  cells: Array<string | number | boolean | null>;
                },
              ) => formatCell(record.cells[index]),
            }))}
            dataSource={tableQuery.data.rows.map((row, index) => ({
              key: index,
              cells: row,
            }))}
          />
        )}
      </Modal>
    </Space>
  );
}
