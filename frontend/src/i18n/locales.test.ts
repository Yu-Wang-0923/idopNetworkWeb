import { describe, expect, it } from "vitest";
import en from "./locales/en.json";
import zh from "./locales/zh.json";

function flatten(value: unknown, prefix = ""): Record<string, string> {
  if (value === null || typeof value !== "object") {
    return { [prefix]: String(value) };
  }
  return Object.entries(value as Record<string, unknown>).reduce(
    (acc, [key, child]) =>
      Object.assign(acc, flatten(child, prefix ? `${prefix}.${key}` : key)),
    {} as Record<string, string>,
  );
}

describe("i18n locales", () => {
  it("zh and en expose the same keys", () => {
    expect(Object.keys(flatten(zh)).sort()).toEqual(
      Object.keys(flatten(en)).sort(),
    );
  });

  it("has no blank translations", () => {
    for (const locale of [zh, en]) {
      for (const [key, value] of Object.entries(flatten(locale))) {
        expect(value.trim(), key).not.toBe("");
      }
    }
  });
});
