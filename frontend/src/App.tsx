import { App as AntdApp, ConfigProvider } from "antd";
import enUS from "antd/locale/en_US";
import zhCN from "antd/locale/zh_CN";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { BrowserRouter } from "react-router-dom";
import { AppRouter } from "./router";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

export default function App() {
  const { i18n } = useTranslation();
  const isEnglish = i18n.language === "en";

  return (
    <QueryClientProvider client={queryClient}>
      <ConfigProvider
        locale={isEnglish ? enUS : zhCN}
        theme={{
          token: {
            colorPrimary: "#245db4",
            colorInfo: "#245db4",
            colorBgLayout: "#f4f7fb",
            borderRadius: 8,
            fontFamily:
              "'Segoe UI', 'Microsoft YaHei', system-ui, -apple-system, sans-serif",
          },
        }}
      >
        <AntdApp>
          <BrowserRouter>
            <AppRouter />
          </BrowserRouter>
        </AntdApp>
      </ConfigProvider>
    </QueryClientProvider>
  );
}
