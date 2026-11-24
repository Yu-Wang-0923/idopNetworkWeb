import { ReloadOutlined } from "@ant-design/icons";
import { Button, Space, Table, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useQuery } from "@tanstack/react-query";
import dayjs from "dayjs";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import StatusTag from "../components/StatusTag";
import type { RunSummary } from "../types";

const { Title, Paragraph } = Typography;

export default function RunsPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const query = useQuery({ queryKey: ["runs"], queryFn: () => api.listRuns() });

  const columns: ColumnsType<RunSummary> = [
    {
      title: t("column.created"),
      dataIndex: "created_at",
      width: 180,
      render: (value: string) => dayjs(value).format("YYYY-MM-DD HH:mm:ss"),
    },
    {
      title: t("column.task"),
      dataIndex: "task_key",
      render: (_value, record) => `${record.module} / ${record.task_key}`,
    },
    {
      title: t("column.status"),
      dataIndex: "status",
      width: 110,
      render: (status) => <StatusTag status={status} />,
    },
    {
      title: t("column.progress"),
      dataIndex: "progress",
      width: 100,
      render: (value: number) => `${value}%`,
    },
    {
      title: t("column.actions"),
      key: "actions",
      width: 100,
      render: (_value, record) => (
        <Button type="link" onClick={() => navigate(`/runs/${record.id}`)}>
          {t("column.view")}
        </Button>
      ),
    },
  ];

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <div className="page-header">
        <Title level={2}>{t("runs.title")}</Title>
        <Paragraph type="secondary">{t("runs.desc")}</Paragraph>
      </div>
      <Space>
        <Button icon={<ReloadOutlined />} onClick={() => query.refetch()} loading={query.isFetching}>
          {t("runs.refresh")}
        </Button>
      </Space>
      <Table<RunSummary>
        rowKey="id"
        dataSource={query.data ?? []}
        columns={columns}
        loading={query.isLoading}
        pagination={false}
        locale={{ emptyText: t("runs.empty") }}
        scroll={{ x: 720 }}
      />
    </Space>
  );
}
