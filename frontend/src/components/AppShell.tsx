import {
  ExperimentOutlined,
  HistoryOutlined,
  HomeOutlined,
} from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import { Alert, Layout, Menu, Space, Tag, Typography } from "antd";
import { useTranslation } from "react-i18next";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import LanguageSwitch from "./LanguageSwitch";

const { Sider, Header, Content, Footer } = Layout;

export default function AppShell() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const session = useQuery({ queryKey: ["session"], queryFn: api.session });
  const tasksQuery = useQuery({
    queryKey: ["tasks"],
    queryFn: api.tasks,
    staleTime: 60_000,
  });

  const moduleRoutes: Record<string, string> = {
    curve_fitting: "/modules/curve-fitting",
    funclu: "/modules/funclu",
    netrecon: "/modules/netrecon",
    netanal: "/modules/netanal",
  };
  const moduleBeta = new Map<string, boolean>();
  for (const task of tasksQuery.data ?? []) {
    if (!(task.module in moduleRoutes)) continue;
    moduleBeta.set(
      task.module,
      (moduleBeta.get(task.module) ?? false) || task.maturity === "beta",
    );
  }
  const moduleItems = Object.entries(moduleRoutes)
    .filter(([module]) => (tasksQuery.isSuccess ? moduleBeta.has(module) : true))
    .map(([module, path]) => ({
      key: path,
      label: (
        <Space size={6}>
          <span>{t(`module.${module}.title`)}</span>
          {moduleBeta.get(module) ? (
            <Tag color="gold" bordered={false}>
              beta
            </Tag>
          ) : null}
        </Space>
      ),
    }));

  const selectedKey = location.pathname.startsWith("/runs")
    ? "/runs"
    : location.pathname === "/"
      ? "/"
      : location.pathname;

  const items = [
    { key: "/", icon: <HomeOutlined />, label: t("nav.home") },
    { key: "/runs", icon: <HistoryOutlined />, label: t("nav.runs") },
    { key: "/demo", icon: <ExperimentOutlined />, label: t("nav.demo") },
    {
      type: "group" as const,
      label: t("nav.modules"),
      children: moduleItems,
    },
  ];

  return (
    <Layout className="app-shell">
      <Sider
        className="app-sider"
        theme="light"
        width={236}
        breakpoint="lg"
        collapsedWidth="0"
      >
        <div className="app-brand">
          <img
            className="app-brand__logo"
            src="/images/TSA.png"
            alt={t("sidebar.logo.alt")}
          />
          <span className="app-brand__name">{t("app.name")}</span>
          <span className="app-brand__subtitle">{t("app.tagline")}</span>
        </div>
        <Menu
          mode="inline"
          items={items}
          selectedKeys={[selectedKey]}
          onClick={({ key }) => navigate(key)}
          style={{ borderInlineEnd: 0 }}
        />
      </Sider>
      <Layout>
        <Header className="app-header">
          <Typography.Text type="secondary">
            {t("nav.retention", { days: session.data?.retention_days ?? 30 })}
          </Typography.Text>
          <Space>
            <Tag bordered={false}>{t("nav.anonymous")}</Tag>
            <LanguageSwitch />
          </Space>
        </Header>
        <Content className="app-content">
          {session.isError ? (
            <Alert
              type="error"
              showIcon
              message={t("common.error")}
              description="API is not reachable. Start the backend with `python -m uvicorn app.main:app --port 8000`."
              style={{ marginBottom: 16 }}
            />
          ) : null}
          <Outlet />
        </Content>
        <Footer className="app-footer">{t("home.footer")}</Footer>
      </Layout>
    </Layout>
  );
}
