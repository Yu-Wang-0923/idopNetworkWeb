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
type Algorithm = "asgl" | "static_idop" | "dynamic_idop";

interface SelectedInput {
  source: "dataset" | "artifact";
  id: string;
  name: string;
  conditions: string[];
}

interface NetReconValues {
  algorithm: Algorithm;
  condition: string;
  max_order: number;
  nonneg_self: boolean;
  max_interactions: number;
  adjacency_aggregation: "mean" | "integral" | "energy_share";
  lasso_alpha: number;
  lasso_windows: number;
  lasso_threshold: number;
  n_fourier: number;
  dynamic_window: number;
  dynamic_fs: number;
  top_edges: number;
}

interface MultiSelectedInput extends SelectedInput {
  clusters: string[];
}

interface MultiValues
  extends Omit<NetReconValues, "condition"> {
  scope: "all" | "inter" | "intra";
  clusters: string[];
  member_cap: number;
}

export default function NetReconPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { message } = AntdApp.useApp();
  const [form] = Form.useForm<NetReconValues>();
  const [source, setSource] = useState<Source>("upload");
  const [mode, setMode] = useState<"single" | "multi">("single");
  const [selected, setSelected] = useState<SelectedInput | null>(null);
  const [uploading, setUploading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const algorithm = Form.useWatch("algorithm", form) ?? "asgl";
  const isDynamic = algorithm === "dynamic_idop";
  const isStatic = algorithm === "static_idop";

  const artifactsQuery = useQuery({
    queryKey: ["artifacts", "curve_fitting_export"],
    queryFn: () => api.listArtifacts("curve_fitting_export"),
    enabled: source === "history",
  });

  useEffect(() => {
    if (isDynamic) {
      form.setFieldsValue({
        max_order: 3,
        nonneg_self: false,
        max_interactions: 0,
        adjacency_aggregation: "energy_share",
        lasso_windows: 0,
        lasso_threshold: 0.9,
      });
    } else if (isStatic) {
      form.setFieldsValue({
        max_order: 1,
        nonneg_self: true,
        adjacency_aggregation: "mean",
        lasso_windows: 1,
        lasso_threshold: 0.5,
      });
    } else {
      form.setFieldsValue({
        max_order: 1,
        nonneg_self: true,
        adjacency_aggregation: "mean",
      });
    }
  }, [algorithm, isDynamic, isStatic, form]);

  useEffect(() => {
    if (selected && selected.conditions.length) {
      const current = form.getFieldValue("condition");
      if (!current || !selected.conditions.includes(current)) {
        form.setFieldsValue({ condition: selected.conditions[0] });
      }
    }
  }, [selected, form]);

  const chooseDataset = async (file: File) => {
    setUploading(true);
    try {
      const dataset = await api.uploadDataset(file);
      if (dataset.detected_kind !== "curve_fitting_export") {
        message.error(t("netrecon.input.invalid"));
        return;
      }
      const inspect = await api.inspectDataset(dataset.id);
      setSelected({
        source: "dataset",
        id: dataset.id,
        name: dataset.original_filename,
        conditions: inspect.conditions,
      });
    } catch {
      message.error(t("netrecon.uploadFailed"));
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
      });
    } catch {
      message.error(t("netrecon.uploadFailed"));
    }
  };

  const submit = async (values: NetReconValues) => {
    if (!selected) {
      message.warning(t("netrecon.needInput"));
      return;
    }
    setSubmitting(true);
    try {
      const run = await api.createRun({
        task_key: "netrecon.single",
        params: {
          ...values,
          adjacency_aggregation: isDynamic
            ? "energy_share"
            : values.adjacency_aggregation,
          nonneg_self: isDynamic ? false : values.nonneg_self,
          max_interactions: isDynamic ? 0 : values.max_interactions,
        },
        inputs: [
          selected.source === "dataset"
            ? { role: "curve_zip", ordinal: 0, dataset_id: selected.id }
            : { role: "curve_zip", ordinal: 0, artifact_id: selected.id },
        ],
      });
      message.success(t("netrecon.runCreated"));
      navigate(`/runs/${run.id}`);
    } catch {
      message.error(t("netrecon.uploadFailed"));
    } finally {
      setSubmitting(false);
    }
  };

  const conditionOptions = (selected?.conditions ?? []).map((condition) => ({
    value: condition,
    label: condition,
  }));

  return (
    <Space direction="vertical" size={18} style={{ width: "100%", maxWidth: 1120 }}>
      <div className="page-header">
        <Title level={2}>{t("netrecon.title")}</Title>
        <Paragraph type="secondary">{t("netrecon.desc")}</Paragraph>
      </div>

      <Segmented
        value={mode}
        onChange={(value) => setMode(value as "single" | "multi")}
        options={[
          { label: t("netrecon.mode.single"), value: "single" },
          { label: t("netrecon.mode.multi"), value: "multi" },
        ]}
      />

      {mode === "single" ? (
        <>
      <Card title={t("netrecon.input.title")}>
        <Segmented
          value={source}
          onChange={(value) => {
            setSource(value as Source);
            setSelected(null);
          }}
          options={[
            { label: t("netrecon.input.upload"), value: "upload" },
            { label: t("netrecon.input.history"), value: "history" },
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
            <p className="ant-upload-text">{t("netrecon.input.uploadHint")}</p>
            <p className="ant-upload-hint">{t("netrecon.input.uploadHelp")}</p>
          </Dragger>
        ) : (
          <Space direction="vertical" size={10} style={{ width: "100%" }}>
            <Select
              showSearch
              placeholder={t("netrecon.input.chooseArtifact")}
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
                message={t("netrecon.input.noArtifacts")}
                action={
                  <Button
                    size="small"
                    onClick={() => navigate("/modules/curve-fitting")}
                  >
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
              {t("netrecon.input.summary", {
                conditions: selected.conditions.length,
              })}
            </Text>
          </Space>
        ) : null}
      </Card>

      <Card title={t("netrecon.settingsTitle")}>
        <Form<NetReconValues>
          form={form}
          name="netrecon-single"
          layout="vertical"
          initialValues={{
            algorithm: "asgl",
            condition: "",
            max_order: 1,
            nonneg_self: true,
            max_interactions: 0,
            adjacency_aggregation: "mean",
            lasso_alpha: 0.1,
            lasso_windows: 1,
            lasso_threshold: 0.5,
            n_fourier: 0,
            dynamic_window: 0,
            dynamic_fs: 100,
            top_edges: 0,
          }}
          onFinish={submit}
        >
          <Row gutter={12}>
            <Col span={24}>
              <Form.Item label={t("netrecon.algorithm")} name="algorithm">
                <Segmented
                  block
                  options={[
                    { label: t("netrecon.algorithm.asgl"), value: "asgl" },
                    { label: t("netrecon.algorithm.static_idop"), value: "static_idop" },
                    { label: t("netrecon.algorithm.dynamic_idop"), value: "dynamic_idop" },
                  ]}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label={t("netrecon.condition")}
                name="condition"
                rules={[{ required: true }]}
              >
                <Select
                  options={conditionOptions}
                  placeholder={t("netrecon.conditionHint")}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label={t("netrecon.maxOrder")} name="max_order">
                <InputNumber
                  min={1}
                  max={100}
                  style={{ width: "100%" }}
                  disabled={false}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                label={t("netrecon.nonnegSelf")}
                name="nonneg_self"
                valuePropName="checked"
              >
                <Switch disabled={isDynamic} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label={t("netrecon.maxInteractions")} name="max_interactions">
                <InputNumber
                  min={0}
                  max={100}
                  style={{ width: "100%" }}
                  disabled={isDynamic}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label={t("netrecon.aggregation")} name="adjacency_aggregation">
                <Select
                  disabled={isDynamic}
                  options={[
                    { value: "mean", label: t("netrecon.aggregation.mean") },
                    { value: "integral", label: t("netrecon.aggregation.integral") },
                  ]}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label={t("netrecon.topEdges")} name="top_edges">
                <InputNumber min={0} max={100000} style={{ width: "100%" }} />
              </Form.Item>
            </Col>
          </Row>

          {isStatic || isDynamic ? (
            <Row gutter={12}>
              <Col span={8}>
                <Form.Item label={t("netrecon.lassoAlpha")} name="lasso_alpha">
                  <InputNumber
                    min={1e-6}
                    max={1}
                    step={0.01}
                    style={{ width: "100%" }}
                  />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item label={t("netrecon.lassoWindows")} name="lasso_windows">
                  <InputNumber
                  min={0}
                  max={50}
                  style={{ width: "100%" }}
                />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item label={t("netrecon.lassoThreshold")} name="lasso_threshold">
                  <InputNumber
                    min={0}
                    max={1}
                    step={0.05}
                    style={{ width: "100%" }}
                  />
                </Form.Item>
              </Col>
            </Row>
          ) : null}

          {isDynamic ? (
            <Row gutter={12}>
              <Col span={8}>
                <Form.Item label={t("netrecon.nFourier")} name="n_fourier">
                  <InputNumber min={0} max={2000} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item label={t("netrecon.dynamicWindow")} name="dynamic_window">
                  <InputNumber min={0} max={1000000} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item label={t("netrecon.dynamicFs")} name="dynamic_fs">
                  <InputNumber min={1} max={10000} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
            </Row>
          ) : null}

          <Alert
            type="info"
            showIcon
            message={
              isDynamic
                ? t("netrecon.tip.dynamic")
                : isStatic
                  ? t("netrecon.tip.static")
                  : t("netrecon.tip.asgl")
            }
            style={{ marginBottom: 14 }}
          />
          <Button
            type="primary"
            htmlType="submit"
            loading={submitting}
            icon={<PlayCircleOutlined />}
          >
            {t("netrecon.submit")}
          </Button>
        </Form>
      </Card>
        </>
      ) : (
        <MultiLayerSection />
      )}
    </Space>
  );
}

function MultiLayerSection() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { message } = AntdApp.useApp();
  const [form] = Form.useForm<MultiValues>();
  const [source, setSource] = useState<Source>("upload");
  const [selected, setSelected] = useState<MultiSelectedInput | null>(null);
  const [uploading, setUploading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const algorithm = Form.useWatch("algorithm", form) ?? "asgl";
  const isDynamic = algorithm === "dynamic_idop";
  const isStatic = algorithm === "static_idop";
  const scope = Form.useWatch("scope", form) ?? "all";

  const artifactsQuery = useQuery({
    queryKey: ["artifacts", "funclu_k_export"],
    queryFn: () => api.listArtifacts("funclu_k_export"),
    enabled: source === "history",
  });

  useEffect(() => {
    if (isDynamic) {
      form.setFieldsValue({
        max_order: 3,
        nonneg_self: false,
        max_interactions: 0,
        adjacency_aggregation: "energy_share",
        lasso_windows: 0,
        lasso_threshold: 0.9,
      });
    } else if (isStatic) {
      form.setFieldsValue({
        max_order: 1,
        nonneg_self: true,
        adjacency_aggregation: "mean",
        lasso_windows: 1,
        lasso_threshold: 0.5,
      });
    } else {
      form.setFieldsValue({
        max_order: 1,
        nonneg_self: true,
        adjacency_aggregation: "mean",
      });
    }
  }, [algorithm, isDynamic, isStatic, form]);

  useEffect(() => {
    if (selected) {
      form.setFieldsValue({ clusters: selected.clusters });
    }
  }, [selected, form]);

  const chooseDataset = async (file: File) => {
    setUploading(true);
    try {
      const dataset = await api.uploadDataset(file);
      if (dataset.detected_kind !== "funclu_k_export") {
        message.error(t("netrecon.multi.invalid"));
        return;
      }
      const inspect = await api.inspectDataset(dataset.id);
      setSelected({
        source: "dataset",
        id: dataset.id,
        name: dataset.original_filename,
        conditions: inspect.conditions,
        clusters: inspect.clusters ?? [],
      });
    } catch {
      message.error(t("netrecon.multi.uploadFailed"));
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
        clusters: inspect.clusters ?? [],
      });
    } catch {
      message.error(t("netrecon.multi.uploadFailed"));
    }
  };

  const submit = async (values: MultiValues) => {
    if (!selected) {
      message.warning(t("netrecon.multi.needInput"));
      return;
    }
    setSubmitting(true);
    try {
      const run = await api.createRun({
        task_key: "netrecon.multi",
        params: {
          ...values,
          adjacency_aggregation: isDynamic
            ? "energy_share"
            : values.adjacency_aggregation,
          nonneg_self: isDynamic ? false : values.nonneg_self,
          max_interactions: isDynamic ? 0 : values.max_interactions,
        },
        inputs: [
          selected.source === "dataset"
            ? { role: "funclu_zip", ordinal: 0, dataset_id: selected.id }
            : { role: "funclu_zip", ordinal: 0, artifact_id: selected.id },
        ],
      });
      message.success(t("netrecon.multi.runCreated"));
      navigate(`/runs/${run.id}`);
    } catch {
      message.error(t("netrecon.multi.uploadFailed"));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Space direction="vertical" size={18} style={{ width: "100%" }}>
      <Card title={t("netrecon.input.title")}>
        <Segmented
          value={source}
          onChange={(value) => {
            setSource(value as Source);
            setSelected(null);
          }}
          options={[
            { label: t("netrecon.input.upload"), value: "upload" },
            { label: t("netrecon.input.history"), value: "history" },
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
            <p className="ant-upload-text">{t("netrecon.multi.uploadHint")}</p>
            <p className="ant-upload-hint">{t("netrecon.multi.uploadHelp")}</p>
          </Dragger>
        ) : (
          <Space direction="vertical" size={10} style={{ width: "100%" }}>
            <Select
              showSearch
              placeholder={t("netrecon.multi.chooseArtifact")}
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
                message={t("netrecon.multi.noArtifacts")}
                action={
                  <Button
                    size="small"
                    onClick={() => navigate("/modules/funclu")}
                  >
                    FunClu
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
              {t("netrecon.multi.summary", {
                conditions: selected.conditions.length,
                clusters: selected.clusters.length,
              })}
            </Text>
          </Space>
        ) : null}
      </Card>

      <Card title={t("netrecon.settingsTitle")}>
        <Form<MultiValues>
          form={form}
          name="netrecon-multi"
          layout="vertical"
          initialValues={{
            algorithm: "asgl",
            max_order: 1,
            nonneg_self: true,
            max_interactions: 0,
            adjacency_aggregation: "mean",
            lasso_alpha: 0.1,
            lasso_windows: 1,
            lasso_threshold: 0.5,
            n_fourier: 0,
            dynamic_window: 0,
            dynamic_fs: 100,
            top_edges: 0,
            scope: "all",
            clusters: [],
            member_cap: 0,
          }}
          onFinish={submit}
        >
          <Row gutter={12}>
            <Col span={24}>
              <Form.Item label={t("netrecon.algorithm")} name="algorithm">
                <Segmented
                  block
                  options={[
                    { label: t("netrecon.algorithm.asgl"), value: "asgl" },
                    { label: t("netrecon.algorithm.static_idop"), value: "static_idop" },
                    { label: t("netrecon.algorithm.dynamic_idop"), value: "dynamic_idop" },
                  ]}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label={t("netrecon.multi.scope")} name="scope">
                <Segmented
                  block
                  options={[
                    { label: t("netrecon.multi.scope.all"), value: "all" },
                    { label: t("netrecon.multi.scope.inter"), value: "inter" },
                    { label: t("netrecon.multi.scope.intra"), value: "intra" },
                  ]}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label={t("netrecon.multi.memberCap")}
                name="member_cap"
                extra={t("netrecon.multi.memberCapHelp")}
              >
                <InputNumber min={0} max={50000} style={{ width: "100%" }} />
              </Form.Item>
            </Col>
            <Col span={24}>
              <Form.Item
                label={t("netrecon.multi.clusters")}
                name="clusters"
                extra={t("netrecon.multi.clustersHelp")}
              >
                <Select
                  mode="multiple"
                  allowClear
                  disabled={scope === "inter"}
                  options={(selected?.clusters ?? []).map((cluster) => ({
                    value: cluster,
                    label: cluster,
                  }))}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label={t("netrecon.maxOrder")} name="max_order">
                <InputNumber min={1} max={100} style={{ width: "100%" }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                label={t("netrecon.nonnegSelf")}
                name="nonneg_self"
                valuePropName="checked"
              >
                <Switch disabled={isDynamic} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label={t("netrecon.maxInteractions")} name="max_interactions">
                <InputNumber
                  min={0}
                  max={100}
                  style={{ width: "100%" }}
                  disabled={isDynamic}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label={t("netrecon.aggregation")} name="adjacency_aggregation">
                <Select
                  disabled={isDynamic}
                  options={[
                    { value: "mean", label: t("netrecon.aggregation.mean") },
                    { value: "integral", label: t("netrecon.aggregation.integral") },
                  ]}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label={t("netrecon.topEdges")} name="top_edges">
                <InputNumber min={0} max={100000} style={{ width: "100%" }} />
              </Form.Item>
            </Col>
          </Row>

          {isStatic || isDynamic ? (
            <Row gutter={12}>
              <Col span={8}>
                <Form.Item label={t("netrecon.lassoAlpha")} name="lasso_alpha">
                  <InputNumber min={1e-6} max={1} step={0.01} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item label={t("netrecon.lassoWindows")} name="lasso_windows">
                  <InputNumber min={0} max={50} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item label={t("netrecon.lassoThreshold")} name="lasso_threshold">
                  <InputNumber min={0} max={1} step={0.05} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
            </Row>
          ) : null}

          {isDynamic ? (
            <Row gutter={12}>
              <Col span={8}>
                <Form.Item label={t("netrecon.nFourier")} name="n_fourier">
                  <InputNumber min={0} max={2000} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item label={t("netrecon.dynamicWindow")} name="dynamic_window">
                  <InputNumber min={0} max={1000000} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item label={t("netrecon.dynamicFs")} name="dynamic_fs">
                  <InputNumber min={1} max={10000} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
            </Row>
          ) : null}

          <Alert
            type="info"
            showIcon
            message={
              isDynamic
                ? t("netrecon.multi.tip.dynamic")
                : isStatic
                  ? t("netrecon.multi.tip.static")
                  : t("netrecon.multi.tip.asgl")
            }
            style={{ marginBottom: 14 }}
          />
          <Button
            type="primary"
            htmlType="submit"
            loading={submitting}
            icon={<PlayCircleOutlined />}
          >
            {t("netrecon.multi.submit")}
          </Button>
        </Form>
      </Card>
    </Space>
  );
}
