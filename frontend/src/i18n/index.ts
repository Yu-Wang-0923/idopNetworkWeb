import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import jobMessages from "@shared/job_messages.json";
import en from "./locales/en.json";
import zh from "./locales/zh.json";

type JobMessages = Record<string, { zh: string; en: string }>;

function withJobMessages(
  base: Record<string, string>,
  language: "zh" | "en",
): Record<string, string> {
  const merged: Record<string, string> = { ...base };
  for (const [key, value] of Object.entries(jobMessages as JobMessages)) {
    merged[key] = value[language];
  }
  return merged;
}

const stored = localStorage.getItem("idop_lang");
const initial = stored === "en" ? "en" : "zh";

void i18n.use(initReactI18next).init({
  resources: {
    zh: { translation: withJobMessages(zh, "zh") },
    en: { translation: withJobMessages(en, "en") },
  },
  lng: initial,
  fallbackLng: "zh",
  keySeparator: false,
  nsSeparator: false,
  interpolation: { escapeValue: false },
});

document.documentElement.lang = initial === "en" ? "en" : "zh-CN";

export function setLanguage(language: "zh" | "en"): void {
  localStorage.setItem("idop_lang", language);
  document.documentElement.lang = language === "en" ? "en" : "zh-CN";
  void i18n.changeLanguage(language);
}

export default i18n;
