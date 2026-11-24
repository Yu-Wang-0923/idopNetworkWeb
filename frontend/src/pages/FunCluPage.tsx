import { InboxOutlined, PlayCircleOutlined } from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import {
  Alert,
  App as AntdApp,
  Button,
  Card,
  Col,
  Form,
  InputNumber,
  Row,
  Segmented,
  Select,
  Space,
  Tabs,
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
  conditions: string[];
  features: number | null;
}

interface KValues {
  n_components: number;
  max_iter: number;
  tol: number;
  random_state: number;
}

interface BicValues {
  k_min: number;
  k_max: number;
  step: number;
  n_init: number;
  max_iter: number;
  tol: number;
  random_state: number;
  min_cluster_size: number;
}

function inputPayload(input: SelectedInput) {
  return input.source === "dataset"
    ? [{ role: "curve_zip", ordinal: 0, dataset_id: input.id }]
    : [{ role: "curve_zip", ordinal: 0, artifact_id: input.id }];
}

function KForm({ input }: { input: SelectedInput | null }) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { message } = AntdApp.useApp();
  const [form] = Form.useForm<KValues>();
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const features = input?.features;
    if (!features) {
      return;
    }
    const current = form.getFieldValue("n_components");
    if (!current || current > features) {
      form.setFieldsValue({ n_components: Math.min(3, features) });
    }
  }, [input, form]);

  const submit = async (values: KValues) => {
    if (!input) {
      message.warning(t("funclu.needInput"));
      return;
    }
    setSubmitting(true);
    try {
      const run = await api.createRun({
        task_key: "funclu.k",
        params: values,
        inputs: inputPayload(input),
      });
      message.success(t("funclu.runCreated"));
      navigate(`/runs/${run.id}`);
    } catch {
      message.error(t("funclu.uploadFailed"));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Form<KValues>
      form={form}
      name="funclu-k"
      layout="vertical"
      initialValues={{
        n_components: 3,
        max_iter: 1000,
        tol: 0.0001,
        random_state: 42,
      }}
      onFinish={submit}
    >
      <Row gutter={12}>
        <Col span={12}>
          <Form.Item
            label={t("funclu.k.nComponents")}
            name="n_components"
            rules={[{ required: true }]}
          >
            <InputNumber
              min={2}
              max={input?.features ?? 10000}
              style={{ width: "100%" }}
            />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item label={t("funclu.k.maxIter")} name="max_iter">
            <InputNumber min={10} max={2000} step={10} style={{ width: "100%" }} />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item label={t("funclu.k.tol")} name="tol">
            <InputNumber min={1e-10} max={1} step={1e-4} style={{ width: "100%" }} />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item label={t("funclu.k.randomState")} name="random_state">
            <InputNumber min={0} max={9999} style={{ width: "100%" }} />
          </Form.Item>
        </Col>
      </Row>
      <Alert
        type="info"
        showIcon
        message={t("funclu.k.tip")}
        style={{ marginBottom: 14 }}
      />
      <Button
        type="primary"
        htmlType="submit"
        loading={submitting}
        icon={<PlayCircleOutlined />}
      >
        {t("funclu.k.submit")}
      </Button>
    </Form>
  );
}

function BicForm({ input }: { input: SelectedInput | null }) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { message } = AntdApp.useApp();
  const [form] = Form.useForm<BicValues>();
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const features = input?.features;
    if (!features) {
      return;
    }
    const kMin = Math.min(form.getFieldValue("k_min") ?? 2, features);
    const kMax = Math.max(
      kMin,
      Math.min(form.getFieldValue("k_max") ?? 20, features),
    );
    form.setFieldsValue({ k_min: kMin, k_max: kMax });
  }, [input, form]);

  const submit = async (values: BicValues) => {
    if (!input) {
      message.warning(t("funclu.needInput"));
      return;
    }
    setSubmitting(true);
    try {
      const run = await api.createRun({
        task_key: "funclu.bic",
        params: values,
        inputs: inputPayload(input),
      });
      message.success(t("funclu.runCreated"));
      navigate(`/runs/${run.id}`);
    } catch {
      message.error(t("funclu.uploadFailed"));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Form<BicValues>
      form={form}
      name="funclu-bic"
      layout="vertical"
      initialValues={{
        k_min: 2,
        k_max: 20,
        step: 1,
        n_init: 5,
        max_iter: 1000,
        tol: 0.0001,
        random_state: 42,
        min_cluster_size: 1,
      }}
      onFinish={submit}
    >
      <Row gutter={12}>
        <Col span={8}>
          <Form.Item label={t("funclu.bic.kMin")} name="k_min">
            <InputNumber min={1} max={10000} style={{ width: "100%" }} />
          </Form.Item>
        </Col>
        <Col span={8}>
          <Form.Item label={t("funclu.bic.kMax")} name="k_max">
            <InputNumber min={1} max={10000} style={{ width: "100%" }} />
          </Form.Item>
        </Col>
        <Col span={8}>
          <Form.Item label={t("funclu.bic.step")} name="step">
            <InputNumber min={1} max={100} style={{ width: "100%" }} />
          </Form.Item>
        </Col>
      </Row>

      <Row gutter={12}>
        <Col span={8}>
          <Form.Item label={t("funclu.bic.nInit")} name="n_init">
            <InputNumber min={1} max={10} style={{ width: "100%" }} />
          </Form.Item>
        </Col>
        <Col span={8}>
          <Form.Item label={t("funclu.bic.maxIter")} name="max_iter">
            <InputNumber min={10} max={2000} step={10} style={{ width: "100%" }} />
          </Form.Item>
        </Col>
        <Col span={8}>
          <Form.Item label={t("funclu.bic.tol")} name="tol">
            <InputNumber min={1e-10} max={1} step={1e-4} style={{ width: "100%" }} />
          </Form.Item>
        </Col>
        <Col span={8}>
          <Form.Item label={t("funclu.bic.randomState")} name="random_state">
            <InputNumber min={0} max={9999} style={{ width: "100%" }} />
          </Form.Item>
        </Col>
        <Col span={8}>
          <Form.Item label={t("funclu.bic.minClusterSize")} name="min_cluster_size">
            <InputNumber
              min={1}
              max={input?.features ?? 10000}
              style={{ width: "100%" }}
            />
          </Form.Item>
        </Col>
      </Row>
      <Alert
        type="info"
        showIcon
        message={t("funclu.bic.tip")}
        style={{ marginBottom: 14 }}
      />
      <Button
        type="primary"
        htmlType="submit"
        loading={submitting}
        icon={<PlayCircleOutlined />}
      >
        {t("funclu.bic.submit")}
      </Button>
    </Form>
  );
}

export default function FunCluPage() {
  const { t } = useTranslation();
  const { message } = AntdApp.useApp();
  const navigate = useNavigate();
  const [source, setSource] = useState<Source>("upload");
  const [selected, setSelected] = useState<SelectedInput | null>(null);
  const [uploading, setUploading] = useState(false);

  const artifactsQuery = useQuery({
    queryKey: ["artifacts", "curve_fitting_export"],
    queryFn: () => api.listArtifacts("curve_fitting_export"),
    enabled: source === "history",
  });

  const chooseDataset = async (file: File) => {
    setUploading(true);
    try {
      const dataset = await api.uploadDataset(file);
      if (dataset.detected_kind !== "curve_fitting_export") {
        message.error(t("funclu.input.invalid"));
        return;
      }
      const inspect = await api.inspectDataset(dataset.id);
      setSelected({
        source: "dataset",
        id: dataset.id,
        name: dataset.original_filename,
        conditions: inspect.conditions,
        features: inspect.features ?? null,
      });
      message.success(t("funclu.input.loaded", { name: dataset.original_filename }));
    } catch {
      message.error(t("funclu.uploadFailed"));
    } finally {
      setUploading(false);
    }
  };

  const chooseArtifact = async (artifactId: string) => {
    try {
      const inspect = await api.inspectArtifact(artifactId);
      const artifact = artifactsQuery.data?.find((item) => item.id === artifactId);
      setSelected({
        source: "artifact",
        id: artifactId,
        name: artifact?.logical_name ?? artifactId,
        conditions: inspect.conditions,
        features: inspect.features ?? null,
      });
    } catch {
      message.error(t("funclu.uploadFailed"));
    }
  };

  return (
    <Space direction="vertical" size={18} style={{ width: "100%", maxWidth: 1120 }}>
      <div className="page-header">
        <Title level={2}>{t("funclu.title")}</Title>
        <Paragraph type="secondary">{t("funclu.desc")}</Paragraph>
      </div>

      <Card title={t("funclu.input.title")}>
        <Segmented
          value={source}
          onChange={(value) => {
            setSource(value as Source);
            setSelected(null);
          }}
          options={[
            { label: t("funclu.input.upload"), value: "upload" },
            { label: t("funclu.input.history"), value: "history" },
          ]}
          style={{ marginBottom: 16 }}
        />

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
            <p className="ant-upload-text">{t("funclu.input.uploadHint")}</p>
            <p className="ant-upload-hint">{t("funclu.input.uploadHelp")}</p>
          </Dragger>
        ) : (
          <Space direction="vertical" size={10} style={{ width: "100%" }}>
            <Select
              showSearch
              placeholder={t("funclu.input.chooseArtifact")}
              style={{ width: "100%" }}
              loading={artifactsQuery.isLoading}
              value={selected?.source === "artifact" ? selected.id : undefined}
              onChange={(value) => void chooseArtifact(value)}
              optionFilterProp="label"
              options={(artifactsQuery.data ?? []).map((item) => ({
                value: item.id,
                label: `${item.logical_name} · ${dayjs(item.created_at).format("YYYY-MM-DD HH:mm")}`,
              }))}
            />
            {!artifactsQuery.isLoading && !(artifactsQuery.data ?? []).length ? (
              <Alert
                type="info"
                showIcon
                message={t("funclu.input.noArtifacts")}
                action={
                  <Button size="small" onClick={() => navigate("/modules/curve-fitting")}>
                    Curve Fitting
                  </Button>
                }
              />
            ) : null}
          </Space>
        )}

        {selected ? (
          <Space wrap style={{ marginTop: 14 }}>
            <Tag color="blue">{selected.name}</Tag>
            <Text type="secondary">
              {t("funclu.input.summary", {
                conditions: selected.conditions.length,
                features: selected.features ?? "—",
              })}
            </Text>
            {selected.conditions.length ? (
              <Text type="secondary">{selected.conditions.join(", ")}</Text>
            ) : null}
          </Space>
        ) : null}
      </Card>

      <Card>
        <Tabs
          items={[
            { key: "k", label: t("funclu.tab.k"), children: <KForm input={selected} /> },
            { key: "bic", label: t("funclu.tab.bic"), children: <BicForm input={selected} /> },
          ]}
        />
      </Card>
    </Space>
  );
}
