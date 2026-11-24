import { ArrowLeftOutlined } from "@ant-design/icons";
import { Button, Card, Space, Tag, Typography } from "antd";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

const { Title, Paragraph } = Typography;

interface Props {
  moduleKey: string;
  phase: number;
}

export default function ModulePlaceholder({ moduleKey, phase }: Props) {
  const { t } = useTranslation();
  const navigate = useNavigate();

  return (
    <Space direction="vertical" size={16} style={{ width: "100%", maxWidth: 760 }}>
      <Button icon={<ArrowLeftOutlined />} onClick={() => navigate("/")}>
        {t("nav.home")}
      </Button>
      <Card>
        <Space direction="vertical" size={10}>
          <Space>
            <Title level={2} style={{ margin: 0 }}>
              {t(`module.${moduleKey}.title`)}
            </Title>
            <Tag color="blue">{t("home.comingSoon", { phase })}</Tag>
          </Space>
          <Paragraph type="secondary" style={{ margin: 0 }}>
            {t(`module.${moduleKey}.desc`)}
          </Paragraph>
          <Paragraph style={{ margin: 0 }}>
            {t("module.placeholder", { phase })}
          </Paragraph>
        </Space>
      </Card>
    </Space>
  );
}
