import { Segmented } from "antd";
import { useTranslation } from "react-i18next";
import { setLanguage } from "../i18n";

export default function LanguageSwitch() {
  const { i18n } = useTranslation();
  const value = i18n.language.startsWith("en") ? "en" : "zh";
  return (
    <Segmented
      size="small"
      value={value}
      options={[
        { label: "中文", value: "zh" },
        { label: "English", value: "en" },
      ]}
      onChange={(next) => setLanguage(next as "zh" | "en")}
    />
  );
}
