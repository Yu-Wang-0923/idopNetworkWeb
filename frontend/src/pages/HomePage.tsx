import { ArrowRightOutlined, ExperimentOutlined } from "@ant-design/icons";
import { Button, Card, Col, Row, Space, Tag, Typography } from "antd";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

const { Title, Paragraph, Text } = Typography;

const MODULES = [
  { key: "curve_fitting", path: "/modules/curve-fitting", phase: 1 },
  { key: "funclu", path: "/modules/funclu", phase: 2 },
  { key: "netrecon", path: "/modules/netrecon", phase: 3 },
  { key: "netanal", path: "/modules/netanal", phase: 5 },
];

export default function HomePage() {
  const { t } = useTranslation();
  const navigate = useNavigate();

  return (
    <Space direction="vertical" size={24} style={{ width: "100%" }}>
      <Card className="hero-card" variant="borderless">
        <Title level={1} style={{ marginTop: 0 }}>
          {t("home.title")}
        </Title>
        <Paragraph>{t("home.intro")}</Paragraph>
        <Space wrap>
          <Tag color="blue">{t("home.badge.informative")}</Tag>
          <Tag color="geekblue">{t("home.badge.dynamic")}</Tag>
          <Tag color="cyan">{t("home.badge.omnidirectional")}</Tag>
          <Tag color="purple">{t("home.badge.personalized")}</Tag>
        </Space>
        <div style={{ marginTop: 20 }}>
          <Text type="secondary">{t("home.workspace")}</Text>
        </div>
      </Card>

      <div>
        <Title level={3}>{t("home.modules")}</Title>
        <Row gutter={[16, 16]}>
          {MODULES.map((module) => (
            <Col xs={24} md={12} xl={6} key={module.key}>
              <Card
                className="module-card"
                title={t(`module.${module.key}.title`)}
                extra={<Tag>{t("home.comingSoon", { phase: module.phase })}</Tag>}
                actions={[
                  <Button
                    key="open"
                    type="link"
                    icon={<ArrowRightOutlined />}
                    onClick={() => navigate(module.path)}
                  >
                    {t("home.open")}
                  </Button>,
                ]}
              >
                <Paragraph type="secondary" style={{ marginBottom: 0 }}>
                  {t(`module.${module.key}.desc`)}
                </Paragraph>
              </Card>
            </Col>
          ))}
        </Row>
      </div>

      <Card
        title={
          <Space>
            <ExperimentOutlined />
            {t("tasks.demo.title")}
          </Space>
        }
        extra={<Tag color="green">{t("home.available")}</Tag>}
        actions={[
          <Button key="demo" type="primary" onClick={() => navigate("/demo")}>
            {t("home.demoCta")}
          </Button>,
        ]}
      >
        <Paragraph type="secondary" style={{ marginBottom: 0 }}>
          {t("tasks.demo.description")}
        </Paragraph>
      </Card>
    </Space>
  );
}
