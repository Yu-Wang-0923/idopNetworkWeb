import { InboxOutlined, PlayCircleOutlined } from "@ant-design/icons";
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
  Typography,
  Upload,
} from "antd";
import type { UploadFile } from "antd";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { Dataset } from "../types";

const { Title, Paragraph, Text, Link } = Typography;
const { Dragger } = Upload;

type Mode = "quasi_dynamic" | "dynamic";

interface FormValues {
  data_mode: Mode;
  first_transform: string;
  second_transform: string;
  n_samples: number;
  trim_percent: number;
  swt_wavelet: string;
  swt_level: number;
  swt_alpha: number;
  ecg_sampling_rate: number;
}

export default function CurveFittingPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { message } = AntdApp.useApp();
  const [form] = Form.useForm<FormValues>();
  const mode = Form.useWatch("data_mode", form) ?? "quasi_dynamic";
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const isDynamic = mode === "dynamic";

  const transformOptions = [
    { value: "None", label: t("fitting.transform.none") },
    { value: "Z_min_add1", label: t("fitting.transform.z_min_add1") },
    { value: "Log10_1p", label: t("fitting.transform.log10_1p") },
    { value: "Minmax_0_1", label: t("fitting.transform.minmax_0_1") },
  ];
  const waveletOptions = [
    { value: "rbio3.9", label: t("fitting.wavelet.rbio39") },
    { value: "db4", label: t("fitting.wavelet.db4") },
    { value: "sym8", label: t("fitting.wavelet.sym8") },
    { value: "coif3", label: t("fitting.wavelet.coif3") },
    { value: "bior3.9", label: t("fitting.wavelet.bior39") },
  ];

  const submit = async (values: FormValues) => {
    if (fileList.length === 0) {
      message.warning(t("fitting.needFiles"));
      return;
    }
    const names = fileList.map((file) => file.name);
    if (new Set(names).size !== names.length) {
      message.error(t("fitting.duplicate"));
      return;
    }
    setSubmitting(true);
    try {
      const datasets: Dataset[] = [];
      for (const file of fileList) {
        const raw = file.originFileObj as File | undefined;
        if (!raw) {
          throw new Error(`Missing file content: ${file.name}`);
        }
        datasets.push(await api.uploadDataset(raw));
      }
      const params = isDynamic
        ? {
            data_mode: mode,
            swt_wavelet: values.swt_wavelet,
            swt_level: values.swt_level,
            swt_alpha: values.swt_alpha,
            ecg_sampling_rate: values.ecg_sampling_rate,
          }
        : {
            data_mode: mode,
            first_transform: values.first_transform,
            second_transform: values.second_transform,
            n_samples: values.n_samples,
            trim_percent: values.trim_percent,
          };
      const run = await api.createRun({
        task_key: "curve_fitting.fit",
        params,
        inputs: datasets.map((dataset, index) => ({
          role: "csv",
          ordinal: index,
          dataset_id: dataset.id,
        })),
      });
      message.success(t("fitting.runCreated"));
      navigate(`/runs/${run.id}`);
    } catch {
      message.error(t("fitting.uploadFailed"));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Space direction="vertical" size={18} style={{ width: "100%", maxWidth: 1120 }}>
      <div className="page-header">
        <Title level={2}>{t("fitting.title")}</Title>
        <Paragraph type="secondary">{t("fitting.desc")}</Paragraph>
      </div>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={13}>
          <Card title={t("fitting.uploadTitle")}>
            <Dragger
              multiple
              accept=".csv"
              fileList={fileList}
              beforeUpload={() => false}
              onChange={({ fileList: next }) => setFileList(next)}
              onRemove={(file) => {
                setFileList((current) =>
                  current.filter((item) => item.uid !== file.uid),
                );
                return true;
              }}
            >
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">{t("fitting.uploadHint")}</p>
              <p className="ant-upload-hint">{t("fitting.uploadHelp")}</p>
            </Dragger>
            <div style={{ marginTop: 14 }}>
              <Text type="secondary">{t("fitting.samplesTitle")}: </Text>
              <Space wrap>
                <Link href="/samples/math_scores.csv" download>
                  {t("fitting.sampleMath")}
                </Link>
                <Link href="/samples/physics_scores.csv" download>
                  {t("fitting.samplePhysics")}
                </Link>
                <Link href="/samples/ecg_100hz.csv" download>
                  {t("fitting.sampleEcg")}
                </Link>
                <Link href="/samples/education_example.zip" download>
                  {t("fitting.sampleEducation")}
                </Link>
              </Space>
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={11}>
          <Card title={t("fitting.settingsTitle")}>
            <Form<FormValues>
              form={form}
              layout="vertical"
              initialValues={{
                data_mode: "quasi_dynamic",
                first_transform: "Z_min_add1",
                second_transform: "Log10_1p",
                n_samples: 30,
                trim_percent: 1.0,
                swt_wavelet: "rbio3.9",
                swt_level: 2,
                swt_alpha: 0.5,
                ecg_sampling_rate: 100,
              }}
              onFinish={submit}
            >
              <Form.Item label={t("fitting.dataMode")} name="data_mode">
                <Segmented
                  block
                  options={[
                    { label: t("fitting.mode.quasi"), value: "quasi_dynamic" },
                    { label: t("fitting.mode.dynamic"), value: "dynamic" },
                  ]}
                />
              </Form.Item>
              <Text type="secondary" style={{ display: "block", marginBottom: 12 }}>
                {isDynamic ? t("fitting.modeHint.dynamic") : t("fitting.modeHint.quasi")}
              </Text>

              {isDynamic ? (
                <Row gutter={12}>
                  <Col span={12}>
                    <Form.Item label={t("fitting.swtWavelet")} name="swt_wavelet">
                      <Select options={waveletOptions} />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item label={t("fitting.swtLevel")} name="swt_level">
                      <InputNumber min={1} max={10} style={{ width: "100%" }} />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item label={t("fitting.swtAlpha")} name="swt_alpha">
                      <InputNumber min={0} max={5} step={0.05} style={{ width: "100%" }} />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item label={t("fitting.samplingRate")} name="ecg_sampling_rate">
                      <InputNumber min={1} max={10000} style={{ width: "100%" }} />
                    </Form.Item>
                  </Col>
                </Row>
              ) : (
                <Row gutter={12}>
                  <Col span={12}>
                    <Form.Item label={t("fitting.firstTransform")} name="first_transform">
                      <Select options={transformOptions} />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item label={t("fitting.secondTransform")} name="second_transform">
                      <Select options={transformOptions} />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item label={t("fitting.nSamples")} name="n_samples">
                      <InputNumber min={2} max={10000} style={{ width: "100%" }} />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item label={t("fitting.trimPercent")} name="trim_percent">
                      <InputNumber min={0} max={99} step={0.5} style={{ width: "100%" }} />
                    </Form.Item>
                  </Col>
                </Row>
              )}

              <Alert
                type="info"
                showIcon
                message={t("fitting.tip")}
                style={{ marginBottom: 14 }}
              />
              <Button
                type="primary"
                htmlType="submit"
                loading={submitting}
                icon={<PlayCircleOutlined />}
              >
                {submitting ? t("fitting.uploading") : t("fitting.submit")}
              </Button>
            </Form>
          </Card>
        </Col>
      </Row>
    </Space>
  );
}
