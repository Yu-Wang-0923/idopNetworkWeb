import { InboxOutlined, PlayCircleOutlined } from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import {
  Alert,
  App as AntdApp,
  Button,
  Card,
  Col,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Switch,
  Tag,
  Typography,
  Upload,
} from "antd";
import dayjs from "dayjs";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";

const { Title, Paragraph, Text } = Typography;
const { Dragger } = Upload;

type Source = "upload" | "history";

interface SelectedInput {
  source: "dataset" | "artifact";
  id: string;
  name: string;
  members: string[];
}

interface GlmyValues {
  member: string;
  mode: "all_edges_colored" | "separate_signs";
  n_stages: number;
  max_columns: number;
  thresholds: string;
  export_vector: boolean;
}

function memberLabel(member: string): string {
  if (member === "from_to.csv") {
    return "single_layer";
  }
  if (member.endsWith("/from_to.csv")) {
    return member.slice(0, -"/from_to.csv".length);
  }
  return member;
}

export default function NetAnalPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { message } = AntdApp.useApp();
  const [form] = Form.useForm<GlmyValues>();
  const [source, setSource] = useState<Source>("upload");
  const [selected, setSelected] = useState<SelectedInput | null>(null);
  const [uploading, setUploading] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const artifactsQuery = useQuery({
    queryKey: ["artifacts", "netrecon_exports"],
    queryFn: () => api.listArtifacts(),
    enabled: source === "history",
  });
  const netreconArtifacts = (artifactsQuery.data ?? []).filter((item) =>
    ["single_layer_idopnetwork_export", "multi_layer_idopnetwork_export"].includes(
      item.semantic_type,
    ),
  );

  useEffect(() => {
    if (selected && selected.members.length) {
      const current = form.getFieldValue("member");
      if (!current || !selected.members.includes(current)) {
        form.setFieldsValue({ member: selected.members[0] });
      }
    }
  }, [selected, form]);

  const chooseDataset = async (file: File) => {
    setUploading(true);
    try {
      const dataset = await api.uploadDataset(file);
      if (dataset.detected_kind !== "netrecon_export") {
        message.error(t("netanal.input.invalid"));
        return;
      }
      const inspect = await api.inspectDataset(dataset.id);
      setSelected({
        source: "dataset",
        id: dataset.id,
        name: dataset.original_filename,
        members: inspect.members,
      });
    } catch {
      message.error(t("netanal.uploadFailed"));
    } finally {
      setUploading(false);
    }
  };

  const chooseArtifact = async (artifactId: string) => {
    try {
      const inspect = await api.inspectArtifact(artifactId);
      const artifact = netreconArtifacts.find((item) => item.id === artifactId);
      setSelected({
        source: "artifact",
        id: artifactId,
        name: artifact?.logical_name ?? artifactId,
        members: inspect.members,
      });
    } catch {
      message.error(t("netanal.uploadFailed"));
    }
  };

  const submit = async (values: GlmyValues) => {
    if (!selected) {
      message.warning(t("netanal.needInput"));
      return;
    }
    let thresholds: number[] | null = null;
    const text = (values.thresholds ?? "").trim();
    if (text) {
      try {
        const parsed = text
          .split(/[,\s]+/)
          .filter(Boolean)
          .map((value) => Number(value));
        if (parsed.some((value) => !Number.isFinite(value) || value < 0)) {
          throw new Error("invalid");
        }
        if (
          parsed.some(
            (value, index) => index > 0 && value <= parsed[index - 1],
          )
        ) {
          throw new Error("invalid");
        }
        thresholds = parsed;
      } catch {
        message.error(t("netanal.thresholdsHelp"));
        return;
      }
    }
    setSubmitting(true);
    try {
      const run = await api.createRun({
        task_key: "netanal.glmy",
        params: {
          member: values.member,
          mode: values.mode,
          n_stages: values.n_stages,
          max_columns: values.max_columns,
          thresholds,
          export_vector: values.export_vector,
        },
        inputs: [
          selected.source === "dataset"
            ? { role: "netrecon_zip", ordinal: 0, dataset_id: selected.id }
            : { role: "netrecon_zip", ordinal: 0, artifact_id: selected.id },
        ],
      });
      message.success(t("netanal.runCreated"));
      navigate(`/runs/${run.id}`);
    } catch {
      message.error(t("netanal.uploadFailed"));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Space direction="vertical" size={18} style={{ width: "100%", maxWidth: 1120 }}>
      <div className="page-header">
        <Title level={2}>{t("netanal.title")}</Title>
        <Paragraph type="secondary">{t("netanal.desc")}</Paragraph>
      </div>

      <Card title={t("netanal.input.title")}>
        <div style={{ marginBottom: 16 }}>
          <Space>
            <Button
              type={source === "upload" ? "primary" : "default"}
              onClick={() => {
                setSource("upload");
                setSelected(null);
              }}
            >
              {t("netanal.input.upload")}
            </Button>
            <Button
              type={source === "history" ? "primary" : "default"}
              onClick={() => {
                setSource("history");
                setSelected(null);
              }}
            >
              {t("netanal.input.history")}
            </Button>
          </Space>
        </div>
        {source === "upload" ? (
          <Dragger
            accept=".zip"
            maxCount={1}
            showUploadList={false}
            beforeUpload={(file) => {
              void chooseDataset(file);
              return false;
            }}
            disabled={uploading}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">{t("netanal.input.uploadHint")}</p>
            <p className="ant-upload-hint">{t("netanal.input.uploadHelp")}</p>
          </Dragger>
        ) : (
          <Space direction="vertical" size={10} style={{ width: "100%" }}>
            <Select
              showSearch
              placeholder={t("netanal.input.chooseArtifact")}
              style={{ width: "100%" }}
              loading={artifactsQuery.isLoading}
              value={selected?.source === "artifact" ? selected.id : undefined}
              onChange={(value) => void chooseArtifact(value)}
              optionFilterProp="label"
              options={netreconArtifacts.map((item) => ({
                value: item.id,
                label: `${item.logical_name} · ${dayjs(item.created_at).format("YYYY-MM-DD HH:mm")}`,
              }))}
            />
            {!artifactsQuery.isLoading && !netreconArtifacts.length ? (
              <Alert type="info" showIcon message={t("netanal.input.noArtifacts")} />
            ) : null}
          </Space>
        )}
        {selected ? (
          <Space wrap style={{ marginTop: 14 }}>
            <Tag color="blue">{selected.name}</Tag>
            <Text type="secondary">
              {t("netanal.input.summary", { members: selected.members.length })}
            </Text>
          </Space>
        ) : null}
      </Card>

      <Card title={t("netanal.settingsTitle")}>
        <Form<GlmyValues>
          form={form}
          name="netanal-glmy"
          layout="vertical"
          initialValues={{
            member: "",
            mode: "all_edges_colored",
            n_stages: 4,
            max_columns: 5,
            thresholds: "",
            export_vector: false,
          }}
          onFinish={submit}
        >
          <Row gutter={12}>
            <Col span={12}>
              <Form.Item
                label={t("netanal.member")}
                name="member"
                rules={[{ required: true }]}
              >
                <Select
                  options={(selected?.members ?? []).map((member) => ({
                    value: member,
                    label: memberLabel(member),
                  }))}
                  placeholder={t("netanal.memberHint")}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label={t("netanal.mode")} name="mode">
                <Select
                  options={[
                    {
                      value: "all_edges_colored",
                      label: t("netanal.mode.all_edges_colored"),
                    },
                    {
                      value: "separate_signs",
                      label: t("netanal.mode.separate_signs"),
                    },
                  ]}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label={t("netanal.nStages")} name="n_stages">
                <InputNumber min={2} max={12} style={{ width: "100%" }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label={t("netanal.maxColumns")} name="max_columns">
                <InputNumber min={1} max={8} style={{ width: "100%" }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                label={t("netanal.exportVector")}
                name="export_vector"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>
            </Col>
            <Col span={24}>
              <Form.Item
                label={t("netanal.thresholds")}
                name="thresholds"
                extra={t("netanal.thresholdsHelp")}
              >
                <Input placeholder="0, 0.3, 0.6, 1.0" />
              </Form.Item>
            </Col>
          </Row>

          <Alert
            type="info"
            showIcon
            message={t("netanal.tip")}
            style={{ marginBottom: 14 }}
          />
          <Button
            type="primary"
            htmlType="submit"
            loading={submitting}
            icon={<PlayCircleOutlined />}
          >
            {t("netanal.submit")}
          </Button>
        </Form>
      </Card>
    </Space>
  );
}
