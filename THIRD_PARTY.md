# Third-party components

- **vis-network 9.1.9** — bundled at `backend/app/static/vis-network.min.js` so
  interactive network figures work offline. Upstream:
  <https://github.com/visjs/vis-network>. The bundle is distributed under the
  MIT license (with Apache-2.0 parts); the license texts are included in the
  file header.
- **fast-funclu** — pinned in `backend/pyproject.toml` at commit
  `583cebbeb8416678f06d979e03b7461701d1c74d` and used as the FunClu engine.
  The project author authorized its use in idopNetwork Web; the upstream
  repository did not contain a LICENSE file when this was added.
