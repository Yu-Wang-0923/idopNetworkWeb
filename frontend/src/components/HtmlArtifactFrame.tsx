import { api } from "../api/client";

interface Props {
  artifactId: string;
  height?: number;
  title?: string;
}

export default function HtmlArtifactFrame({ artifactId, height = 720, title }: Props) {
  return (
    <iframe
      className="artifact-frame"
      title={title ?? "idopNetwork artifact"}
      src={api.artifactUrl(artifactId, "content")}
      sandbox="allow-scripts allow-downloads"
      referrerPolicy="no-referrer"
      style={{ height }}
    />
  );
}
