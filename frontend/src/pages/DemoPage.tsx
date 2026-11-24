import { App as AntdApp, Button, Card, Form, Input, InputNumber, Space, Typography } from "antd";
import { useMutation } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";

const { Title, Paragraph } = Typography;

interface DemoForm {
  message: string;
  steps: number;
  delay_ms: number;
}

export default function DemoPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { message } = AntdApp.useApp();

  const mutation = useMutation({
    mutationFn: (values: DemoForm) =>
      api.createRun({
        task_key: "demo.echo",
        params: {
          message: values.message,
          steps: values.steps,
          delay_ms: values.delay_ms,
        },
        inputs: [],
      }),
    onSuccess: (run) => {
      message.success(t("demo.started"));
      navigate(`/runs/${run.id}`);
    },
    onError: () => message.error(t("common.error")),
  });

  return (
    <Space direction="vertical" size={20} style={{ width: "100%", maxWidth: 720 }}>
      <div className="page-header">
        <Title level={2}>{t("demo.title")}</Title>
        <Paragraph type="secondary">{t("demo.desc")}</Paragraph>
      </div>
      <Card>
        <Form<DemoForm>
          layout="vertical"
          initialValues={{ message: "Hello idopNetwork Web", steps: 5, delay_ms: 400 }}
          onFinish={(values) => mutation.mutate(values)}
        >
          <Form.Item
            label={t("demo.message")}
            name="message"
            rules={[{ required: true, max: 200 }]}
          >
            <Input />
          </Form.Item>
          <Form.Item label={t("demo.steps")} name="steps">
            <InputNumber min={1} max={20} style={{ width: 160 }} />
          </Form.Item>
          <Form.Item label={t("demo.delay")} name="delay_ms">
            <InputNumber min={0} max={5000} step={100} style={{ width: 160 }} />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={mutation.isPending}>
            {t("demo.submit")}
          </Button>
        </Form>
      </Card>
    </Space>
  );
}
