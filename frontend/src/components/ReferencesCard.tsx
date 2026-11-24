import { Card, Typography } from "antd";
import { useTranslation } from "react-i18next";
import references from "../data/references.json";

const { Link } = Typography;

interface Reference {
  year: number;
  title: string;
  authors: string;
  journal: string;
  doi: string;
  url: string;
}

const YEARS = Array.from(
  new Set((references as Reference[]).map((item) => item.year)),
).sort((a, b) => b - a);

export default function ReferencesCard() {
  const { t } = useTranslation();
  return (
    <Card title={t("home.section.papers")} className="references-card">
      <div
        className="references-scroll"
        role="region"
        aria-label={t("home.references.regionLabel")}
        tabIndex={0}
      >
        {YEARS.map((year) => (
          <section key={year} className="references-year">
            <h3>{year}</h3>
            {(references as Reference[])
              .filter((item) => item.year === year)
              .map((item, index) => (
                <div className="reference-item" key={`${year}-${index}`}>
                  <div className="reference-title">{item.title}</div>
                  <div className="reference-meta">
                    {item.authors} ({item.year}). <i>{item.journal}</i>.
                    {item.url ? (
                      <>
                        {" "}
                        <Link href={item.url} target="_blank" rel="noreferrer">
                          {item.doi || item.url}
                        </Link>
                      </>
                    ) : null}
                  </div>
                </div>
              ))}
          </section>
        ))}
      </div>
    </Card>
  );
}
