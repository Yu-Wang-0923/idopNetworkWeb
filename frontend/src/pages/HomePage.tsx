import { ArrowRightOutlined, ExperimentOutlined } from "@ant-design/icons";
import { Button, Card, Col, Row, Space, Tag, Tooltip, Typography } from "antd";
import { Trans, useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import ReferencesCard from "../components/ReferencesCard";

const { Title, Paragraph, Text } = Typography;

const BADGES = ["informative", "dynamic", "omnidirectional", "personalized"] as const;

const BADGE_COLORS: Record<(typeof BADGES)[number], string> = {
  informative: "blue",
  dynamic: "geekblue",
  omnidirectional: "cyan",
  personalized: "purple",
};

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
      <Card className="home-hero" variant="borderless">
        <div className="home-hero-grid">
          <div className="home-hero-main">
            <div className="home-hero-head">
              <h1 className="home-title">
                idopNetwork
                <span className="home-title-cn">{t("home.hero.suffix")}</span>
              </h1>
            </div>
            <div className="home-hero-body">
              <figure className="home-hero-figure">
                <img src="/images/idopn.png" alt={t("home.hero.figure.alt")} />
              </figure>
              <div className="home-hero-copy">
                <Paragraph className="home-intro">
                  <Trans
                    i18nKey="home.hero.intro"
                    components={{ b: <strong /> }}
                  />
                </Paragraph>
                <div className="home-badges">
                  {BADGES.map((key) => (
                    <Tooltip key={key} title={t(`home.badge.${key}.tip`)}>
                      <Tag color={BADGE_COLORS[key]}>{t(`home.badge.${key}`)}</Tag>
                    </Tooltip>
                  ))}
                </div>
                <Space wrap align="center" style={{ marginTop: 18 }}>
                  <Button
                    type="primary"
                    icon={<ExperimentOutlined />}
                    onClick={() => navigate("/demo")}
                  >
                    {t("home.demoCta")}
                  </Button>
                  <Text type="secondary">{t("home.workspace")}</Text>
                </Space>
              </div>
            </div>
          </div>
          <aside className="home-profile">
            <img
              className="home-profile__photo"
              src="/images/wu.jpg"
              alt={t("home.profile.photo.alt")}
            />
            <div className="home-profile__name">{t("home.profile.name")}</div>
            <div className="home-profile__role">{t("home.profile.role")}</div>
            <a
              className="home-profile__link"
              href="https://www.bimsa.cn/zh-CN/detail/ronglingwu.html"
              target="_blank"
              rel="noreferrer"
            >
              {t("home.profile.link")} →
            </a>
          </aside>
        </div>
      </Card>

      <section>
        <Title level={3} className="home-section-title">
          {t("home.section.modules")}
        </Title>
        <Row gutter={[16, 16]}>
          {MODULES.map((module) => (
            <Col xs={24} md={12} xl={6} key={module.key}>
              <Card
                className="module-card"
                title={t(`home.module.${module.key}.title`)}
                actions={[
                  <Button
                    key="open"
                    type="link"
                    icon={<ArrowRightOutlined />}
                    onClick={() => navigate(module.path)}
                  >
                    {t("home.module.entry")}
                  </Button>,
                ]}
              >
                <Paragraph type="secondary" style={{ marginBottom: 0 }}>
                  {t(`home.module.${module.key}.desc`)}
                </Paragraph>
              </Card>
            </Col>
          ))}
        </Row>
      </section>

      <ReferencesCard />
    </Space>
  );
}
