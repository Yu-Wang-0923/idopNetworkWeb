"""HTML/SVG renderers for NetRecon effect decomposition and adjacency heatmaps.

The layouts follow the old Streamlit figures (observed points, predicted line,
red self effect, colored cross effects; blue-white-red off-diagonal heatmap with
a grayscale diagonal) but are rendered as self-contained HTML pages.
"""
from __future__ import annotations

import math
from html import escape
from typing import Any

import numpy as np
import pandas as pd

OBSERVED_COLOR = "#F59E0B"
PREDICTED_COLOR = "#3B6FA1"
SELF_COLOR = "#C0504D"
NONSELF_COLOR = "#7FA88A"
CROSS_COLORS = (
    "#2CA02C",
    "#9467BD",
    "#17BECF",
    "#8C564B",
    "#7F7F7F",
    "#BCBD22",
    "#1B9E77",
    "#66A61E",
    "#6A3D9A",
    "#A6761D",
    "#4D908E",
    "#577590",
)

PANEL_W = 250.0
PANEL_H = 190.0
PANEL_LEFT = 36.0
PANEL_RIGHT = 10.0
PANEL_TOP = 34.0
PANEL_BOTTOM = 26.0
COLUMNS = 4
PAGE_ROWS = 4
PAGE_MARGIN = 20.0
PAGE_GAP_X = 16.0
PAGE_GAP_Y = 20.0
PAGE_HEADER = 46.0
PAGE_FOOTER = 58.0

PAGE_STYLE = """
  *{box-sizing:border-box}
  html,body{margin:0;background:#fff;color:#1a1a1a}
  body{font:12px/1.4 Helvetica,Arial,"DejaVu Sans",sans-serif}
  .page{max-width:1120px;margin:0 auto;padding:8px 14px 24px}
  .head h1{font-size:16px;margin:2px 0 4px}
  .head p{margin:0 0 10px;color:#555;font-size:12px}
  .toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;
           margin:0 0 10px;padding:4px 2px 8px;border-bottom:1px solid #e5e8ee}
  .toolbar .spacer{flex:1}
  .toolbar button{border:1px solid #c9d3e0;background:#fff;border-radius:4px;
                  padding:3px 11px;font-size:12px;color:#245db4;cursor:pointer}
  .toolbar button:hover:not(:disabled){background:#eef4fc}
  .toolbar button:disabled{opacity:.35;cursor:default}
  .toolbar select{border:1px solid #c9d3e0;border-radius:4px;padding:2px 4px;
                  font-size:12px;background:#fff}
  .toolbar .counter{color:#666;font-size:12px;font-variant-numeric:tabular-nums}
  .effect-page[hidden]{display:none}
  .effect-page svg{display:block;width:100%;height:auto;background:#fff}
  .page > svg{display:block;width:auto;max-width:100%;height:auto;margin:0 auto}
  @media print{
    .toolbar{display:none}
    .page{max-width:none;padding:0}
    .effect-page{break-after:page}
    .effect-page:last-child{break-after:auto}
  }
"""


def _nice_ticks(low: float, high: float, count: int) -> list[float]:
    if not high > low:
        return [low]
    raw = (high - low) / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = magnitude
    for multiplier in (1, 2, 2.5, 5, 10):
        step = multiplier * magnitude
        if (high - low) / step <= count + 0.5:
            break
    ticks = []
    value = math.ceil(low / step - 1e-9) * step
    while value <= high + step * 1e-9:
        ticks.append(value)
        value += step
    return ticks


def _decimals(step: float) -> int:
    text = f"{step:.6f}".rstrip("0").rstrip(".")
    return len(text.split(".")[1]) if "." in text else 0


def _fmt(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    return "0" if text in {"-0", "-0.0", "-0.00"} else text


def _short_label(name: str, limit: int = 18) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _wrap_label(name: str, limit: int = 22) -> list[str]:
    if len(name) <= limit:
        return [name]
    middle = len(name) // 2
    cuts = [index for index, char in enumerate(name) if char in "_ "]
    if cuts:
        cut = min(cuts, key=lambda index: abs(index - middle))
        first, second = name[: cut + 1], name[cut + 1 :]
    else:
        first, second = name[:limit], name[limit:]
    if len(second) > limit:
        second = second[: limit - 1] + "…"
    return [first, second]


def _effect_panel_content(
    target: str,
    predicted: pd.DataFrame,
    effect: pd.DataFrame,
    intercept: float,
) -> str:
    x = np.asarray(effect.index, dtype=float)
    self_series = effect[target].to_numpy(dtype=float) + intercept
    cross_columns = [column for column in effect.columns if str(column) != target]
    nonself_series = (
        effect.loc[:, cross_columns].sum(axis=1).to_numpy(dtype=float)
        if cross_columns
        else np.zeros(len(effect), dtype=float)
    )
    total_series = predicted[target].reindex(effect.index).to_numpy(dtype=float)
    if not np.isfinite(total_series).all():
        total_series = self_series + nonself_series
    values = np.concatenate([self_series, nonself_series, total_series])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"No finite effect series for {target}")
    y_low = float(np.min(finite))
    y_high = float(np.max(finite))
    # Every panel keeps zero in view so the dashed zero reference is visible.
    y_low = min(y_low, 0.0)
    y_high = max(y_high, 0.0)
    span = y_high - y_low
    if span <= 0:
        span = max(abs(y_high), 1.0)
    y_low -= span * 0.08
    y_high += span * 0.08
    x_low = float(np.min(x))
    x_high = float(np.max(x))
    if x_high <= x_low:
        x_high = x_low + 1.0

    width = PANEL_W - PANEL_LEFT - PANEL_RIGHT
    height = PANEL_H - PANEL_TOP - PANEL_BOTTOM

    def sx(value: float) -> float:
        return PANEL_LEFT + (value - x_low) / (x_high - x_low) * width

    def sy(value: float) -> float:
        return PANEL_TOP + (y_high - value) / (y_high - y_low) * height

    x_ticks = _nice_ticks(x_low, x_high, 3)
    y_ticks = _nice_ticks(y_low, y_high, 3)
    x_dec = _decimals(x_ticks[1] - x_ticks[0]) if len(x_ticks) > 1 else 1
    y_dec = _decimals(y_ticks[1] - y_ticks[0]) if len(y_ticks) > 1 else 2
    bottom = PANEL_TOP + height
    right = PANEL_LEFT + width

    def path(values_series: np.ndarray, color: str, width_value: float, dash: str) -> str:
        points = " ".join(
            f'{"L" if index else "M"}{sx(float(x_value)):.1f},{sy(float(y_value)):.1f}'
            for index, (x_value, y_value) in enumerate(zip(x, values_series))
            if np.isfinite(y_value)
        )
        return (
            f'<path d="{points}" fill="none" stroke="{color}" '
            f'stroke-width="{width_value}" stroke-dasharray="{dash}"/>'
        )

    parts = [
        f'<line x1="{PANEL_LEFT}" y1="{bottom:.1f}" x2="{right:.1f}" '
        f'y2="{bottom:.1f}" stroke="#333" stroke-width="0.8"/>',
        f'<line x1="{PANEL_LEFT}" y1="{PANEL_TOP:.1f}" x2="{PANEL_LEFT}" '
        f'y2="{bottom:.1f}" stroke="#333" stroke-width="0.8"/>',
    ]
    for tick in x_ticks:
        px = sx(tick)
        parts.append(
            f'<line x1="{px:.1f}" y1="{bottom:.1f}" x2="{px:.1f}" '
            f'y2="{bottom + 3:.1f}" stroke="#333" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{px:.1f}" y="{bottom + 13:.1f}" text-anchor="middle" '
            f'font-size="7.5">{_fmt(tick, x_dec)}</text>'
        )
    for tick in y_ticks:
        py = sy(tick)
        parts.append(
            f'<line x1="{PANEL_LEFT - 3:.1f}" y1="{py:.1f}" x2="{PANEL_LEFT:.1f}" '
            f'y2="{py:.1f}" stroke="#333" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{PANEL_LEFT - 4:.1f}" y="{py + 2.6:.1f}" text-anchor="end" '
            f'font-size="7.5">{_fmt(tick, y_dec)}</text>'
        )
    clip_id = "clip-" + str(abs(hash((target, x_low, x_high))) % 10**8)
    parts.append(
        f'<clipPath id="{clip_id}"><rect x="{PANEL_LEFT}" y="{PANEL_TOP}" '
        f'width="{width}" height="{height}"/></clipPath>'
    )
    if y_low <= 0.0 <= y_high:
        zero_y = sy(0.0)
        parts.append(
            f'<line x1="{PANEL_LEFT}" y1="{zero_y:.1f}" x2="{right:.1f}" '
            f'y2="{zero_y:.1f}" stroke="#8A8A8A" stroke-width="0.7" '
            f'stroke-dasharray="3 3"/>'
        )
    parts.append(f'<g clip-path="url(#{clip_id})">')
    parts.append(path(nonself_series, NONSELF_COLOR, 1.2, "2 3"))
    parts.append(path(self_series, SELF_COLOR, 1.4, "6 4"))
    parts.append(path(total_series, PREDICTED_COLOR, 1.8, "none"))
    parts.append("</g>")
    lines = _wrap_label(target)
    for index, line in enumerate(lines):
        parts.append(
            f'<text x="{PANEL_LEFT + width / 2:.1f}" '
            f'y="{10 + index * 11:.1f}" text-anchor="middle" '
            f'font-size="9" font-weight="600">{escape(line)}</text>'
        )
    return "".join(parts)


def _legend_svg(x: float, y: float) -> str:
    items = (
        (SELF_COLOR, "6 4", "Self-effect + initial value", 150.0),
        (NONSELF_COLOR, "2 3", "Summed non-self effects", 142.0),
        (PREDICTED_COLOR, "none", "Model total", 72.0),
    )
    parts = []
    cursor = x
    for color, dash, label, text_width in items:
        parts.append(
            f'<line x1="{cursor:.1f}" y1="{y:.1f}" x2="{cursor + 24:.1f}" '
            f'y2="{y:.1f}" stroke="{color}" stroke-width="1.6" '
            f'stroke-dasharray="{dash}"/>'
        )
        parts.append(
            f'<text x="{cursor + 30:.1f}" y="{y + 3:.1f}" font-size="9">'
            f"{escape(label)}</text>"
        )
        cursor += 30 + text_width + 18
    return "".join(parts)


def _page_svg(
    page_index: int,
    page_targets: list[str],
    predicted: pd.DataFrame,
    effect_map: dict[str, pd.DataFrame],
    intercept_map: dict[str, float],
    *,
    title: str,
    subtitle: str,
    page_count: int,
) -> str:
    rows = max(1, int(math.ceil(len(page_targets) / COLUMNS)))
    width = 2 * PAGE_MARGIN + COLUMNS * PANEL_W + (COLUMNS - 1) * PAGE_GAP_X
    height = PAGE_HEADER + rows * PANEL_H + (rows - 1) * PAGE_GAP_Y + PAGE_FOOTER
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}pt" height="{height:.0f}pt" role="img" '
        f'font-family="Helvetica, Arial, sans-serif" aria-label="{escape(title)}">',
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
        f'<text x="{PAGE_MARGIN:.1f}" y="20" font-size="12" font-weight="700">'
        f"{escape(title)}</text>",
        f'<text x="{PAGE_MARGIN:.1f}" y="34" font-size="9" fill="#444">'
        f"{escape(subtitle)}</text>",
        f'<text x="{width - PAGE_MARGIN:.1f}" y="20" text-anchor="end" font-size="9" '
        f'fill="#444">Page {page_index + 1} / {page_count}</text>',
    ]
    for position, target in enumerate(page_targets):
        column = position % COLUMNS
        row = position // COLUMNS
        x = PAGE_MARGIN + column * (PANEL_W + PAGE_GAP_X)
        y = PAGE_HEADER + row * (PANEL_H + PAGE_GAP_Y)
        effect = effect_map[target]
        parts.append(f'<g transform="translate({x:.1f} {y:.1f})">')
        parts.append(
            _effect_panel_content(
                target,
                predicted,
                effect,
                float(intercept_map.get(target, 0.0)),
            )
        )
        parts.append("</g>")
    parts.append(
        f'<text x="{width / 2:.1f}" y="{height - PAGE_FOOTER + 18:.1f}" '
        f'text-anchor="middle" font-size="9">Quasi-time index (dimensionless)</text>'
    )
    legend_width = 30 + 150 + 18 + 30 + 142 + 18 + 30 + 72
    parts.append(_legend_svg((width - legend_width) / 2, height - 20))
    parts.append("</svg>")
    return "".join(parts)


PAGE_SCRIPT = """
(function () {
  var pages = Array.prototype.slice.call(document.querySelectorAll(".effect-page"));
  var select = document.getElementById("effect-select");
  var counter = document.getElementById("effect-counter");
  var previous = document.getElementById("effect-prev");
  var next = document.getElementById("effect-next");
  var current = 0;
  function currentSvg() {
    return pages[current].querySelector("svg");
  }
  function show(index) {
    current = Math.max(0, Math.min(pages.length - 1, index));
    pages.forEach(function (page, position) { page.hidden = position !== current; });
    select.value = String(current);
    counter.textContent = (current + 1) + " / " + pages.length;
    previous.disabled = current === 0;
    next.disabled = current === pages.length - 1;
    window.scrollTo(0, 0);
  }
  function download(blob, name) {
    var url = URL.createObjectURL(blob);
    var anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }
  function serialized() {
    var svg = currentSvg().cloneNode(true);
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    return new XMLSerializer().serializeToString(svg);
  }
  document.getElementById("effect-svg").addEventListener("click", function () {
    download(new Blob([serialized()], {type: "image/svg+xml"}), "effect_decomposition_" + (current + 1) + ".svg");
  });
  document.getElementById("effect-png").addEventListener("click", function () {
    var text = serialized();
    var image = new Image();
    image.onload = function () {
      var viewBox = currentSvg().getAttribute("viewBox").split(/\\s+/).map(Number);
      var dpi = Number(document.getElementById("effect-dpi").value);
      var canvas = document.createElement("canvas");
      canvas.width = Math.round(viewBox[2] / 72 * dpi);
      canvas.height = Math.round(viewBox[3] / 72 * dpi);
      var context = canvas.getContext("2d");
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(function (blob) {
        download(blob, "effect_decomposition_" + (current + 1) + "_" + dpi + "dpi.png");
      }, "image/png");
    };
    image.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(text);
  });
  select.addEventListener("change", function () { show(Number(select.value)); });
  previous.addEventListener("click", function () { show(current - 1); });
  next.addEventListener("click", function () { show(current + 1); });
  if (pages.length < 2) { select.disabled = true; }
  show(0);
})();
"""


def render_effects(
    predicted: pd.DataFrame,
    effects: list[pd.DataFrame],
    intercept: pd.Series | dict[str, float] | None,
    *,
    title: str = "Effect decomposition",
    subtitle: str = "",
) -> str:
    """Render reference-style effect panels as publication SVG pages in HTML."""
    targets = [str(column) for column in predicted.columns]
    effect_map = {str(target): effect for target, effect in zip(targets, effects)}
    intercept_map = (
        intercept.to_dict() if isinstance(intercept, pd.Series) else (intercept or {})
    )
    targets = [
        target
        for target in targets
        if target in effect_map and target in effect_map[target].columns
    ]
    if not targets:
        raise ValueError("No target has an effect table to plot")
    per_page = COLUMNS * PAGE_ROWS
    chunks = [
        targets[index : index + per_page]
        for index in range(0, len(targets), per_page)
    ]
    pages = [
        _page_svg(
            index,
            chunk,
            predicted,
            effect_map,
            intercept_map,
            title=title,
            subtitle=subtitle,
            page_count=len(chunks),
        )
        for index, chunk in enumerate(chunks)
    ]
    options = "".join(
        f'<option value="{index}">Page {index + 1} · {escape(chunk[0])}</option>'
        for index, chunk in enumerate(chunks)
    )
    sections = "".join(
        f'<section class="effect-page" data-page="{index}">{svg}</section>'
        for index, svg in enumerate(pages)
    )
    return (
        "<!DOCTYPE html>"
        '<html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{PAGE_STYLE}</style></head><body>"
        '<main class="page">'
        '<div class="toolbar">'
        '<button type="button" id="effect-prev" aria-label="Previous page">‹</button>'
        f'<select id="effect-select" aria-label="Page">{options}</select>'
        '<span class="counter" id="effect-counter"></span>'
        '<button type="button" id="effect-next" aria-label="Next page">›</button>'
        '<span class="spacer"></span>'
        '<select id="effect-dpi" aria-label="DPI"><option value="300" selected>300 dpi</option><option value="600">600 dpi</option></select>'
        '<button type="button" id="effect-svg">Export SVG</button>'
        '<button type="button" id="effect-png">Export PNG</button>'
        "</div>"
        f"{sections}"
        "</main>"
        f"<script>{PAGE_SCRIPT}</script></body></html>"
    )


def _color_scale(value: float, low_color: tuple[int, int, int], high_color: tuple[int, int, int]) -> str:
    value = max(0.0, min(1.0, float(value)))
    rgb = tuple(
        int(round(low + (high - low) * value))
        for low, high in zip(low_color, high_color)
    )
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def render_heatmap(
    adjacency: pd.DataFrame,
    *,
    title: str = "Adjusted Matrix Heatmap",
    subtitle: str = "",
) -> str:
    """Render the adjacency heatmap as an inline SVG page."""
    labels = [str(value) for value in adjacency.index]
    size = len(labels)
    raw = adjacency.to_numpy(dtype=float)
    off = raw.copy()
    np.fill_diagonal(off, np.nan)
    off[np.abs(off) < 1e-12] = np.nan
    diag = np.abs(np.diag(raw))
    diag[diag < 1e-12] = np.nan
    off_max = float(np.nanmax(np.abs(off))) if np.isfinite(off).any() else 1.0
    diag_max = float(np.nanmax(diag)) if np.isfinite(diag).any() else 1.0
    off_max = off_max if off_max > 1e-12 else 1.0
    diag_max = diag_max if diag_max > 1e-12 else 1.0

    cell = max(7.0, min(15.0, 900.0 / max(size, 1)))
    left = 150.0
    top = 120.0
    width = left + size * cell + 26.0
    height = top + size * cell + 60.0
    top_every = max(1, int(math.ceil(size / 15)))
    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="{escape(title)}">',
        f'<title>{escape(title)}</title>',
    ]
    blue = (33, 102, 172)
    white = (255, 255, 255)
    red = (178, 24, 43)
    for row in range(size):
        for column in range(size):
            x = left + column * cell
            y = top + row * cell
            value = raw[row, column]
            if row == column:
                if not np.isfinite(diag[row]):
                    color = "#ffffff"
                else:
                    color = _color_scale(diag[row] / diag_max, white, (20, 20, 20))
                weight_text = f"self |{value:.4g}|"
            else:
                if not np.isfinite(off[row, column]):
                    color = "#ffffff"
                elif value >= 0:
                    color = _color_scale(abs(value) / off_max, white, red)
                else:
                    color = _color_scale(abs(value) / off_max, white, blue)
                weight_text = f"{value:.4g}"
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
                f'fill="{color}" stroke="#f0f0f0" stroke-width="0.2">'
                f"<title>{escape(labels[row])} → {escape(labels[column])}: "
                f"{weight_text}</title></rect>"
            )
    for index, label in enumerate(labels):
        y = top + index * cell + cell * 0.7
        parts.append(
            f'<text x="{left - 5:.1f}" y="{y:.1f}" text-anchor="end" '
            f'font-size="{min(9.0, max(6.0, cell * 0.68)):.1f}">{escape(label)}</text>'
        )
        if index % top_every:
            continue
        x = left + index * cell + cell * 0.7
        parts.append(
            f'<text x="{x:.1f}" y="{top - 6:.1f}" text-anchor="end" '
            f'font-size="{min(10.0, max(6.5, cell * 0.8)):.1f}" '
            f'transform="rotate(-60 {x:.1f} {top - 6:.1f})">{escape(label)}</text>'
        )
    parts.append(
        f'<text x="{left:.1f}" y="{top + size * cell + 22:.1f}" font-size="11">'
        f"rows: source · columns: target · off-diagonal blue/red = signed weight "
        f"(normalized), diagonal gray = |self effect|</text>"
    )
    legend_y = top + size * cell + 36.0
    steps = 60
    for index in range(steps):
        fraction = index / (steps - 1)
        if fraction < 0.5:
            color = _color_scale(fraction * 2, blue, white)
        else:
            color = _color_scale((fraction - 0.5) * 2, white, red)
        parts.append(
            f'<rect x="{left + index * 3:.1f}" y="{legend_y:.1f}" width="3" '
            f'height="10" fill="{color}"/>'
        )
    parts.append(
        f'<text x="{left:.1f}" y="{legend_y + 22:.1f}" font-size="10">'
        f"−{off_max:.3g}</text>"
    )
    parts.append(
        f'<text x="{left + steps * 3:.1f}" y="{legend_y + 22:.1f}" '
        f'text-anchor="end" font-size="10">{off_max:.3g}</text>'
    )
    parts.append(
        f'<text x="{left + steps * 1.5:.1f}" y="{legend_y + 22:.1f}" '
        f'text-anchor="middle" font-size="10">signed edge weight</text>'
    )
    parts.append("</svg>")
    note = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    return (
        "<!DOCTYPE html>\n<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{PAGE_STYLE}</style></head><body>"
        '<main class="page"><div class="head">'
        f"<h1>{escape(title)}</h1>{note}</div>"
        f"{''.join(parts)}</main></body></html>"
    )


OVERVIEW_PANEL_W = 210.0
OVERVIEW_PANEL_H = 160.0
OVERVIEW_ROW_LABEL = 92.0
OVERVIEW_HEADER = 34.0
OVERVIEW_MARGIN = 18.0
OVERVIEW_GAP = 8.0

OVERVIEW_SCRIPT = """
(function () {
  var svg = document.getElementById("ml-overview");
  function download(blob, name) {
    var url = URL.createObjectURL(blob);
    var anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }
  function serialized() {
    var clone = svg.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    return new XMLSerializer().serializeToString(clone);
  }
  document.getElementById("ml-svg").addEventListener("click", function () {
    download(new Blob([serialized()], {type: "image/svg+xml"}), "multilayer_overview.svg");
  });
  document.getElementById("ml-png").addEventListener("click", function () {
    var text = serialized();
    var image = new Image();
    image.onload = function () {
      var viewBox = svg.getAttribute("viewBox").split(/\\s+/).map(Number);
      var dpi = Number(document.getElementById("ml-dpi").value);
      var canvas = document.createElement("canvas");
      canvas.width = Math.round(viewBox[2] / 72 * dpi);
      canvas.height = Math.round(viewBox[3] / 72 * dpi);
      var context = canvas.getContext("2d");
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(function (blob) {
        download(blob, "multilayer_overview_" + dpi + "dpi.png");
      }, "image/png");
    };
    image.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(text);
  });
})();
"""


def _overview_panel(adjacency: pd.DataFrame, *, max_labels: int = 12) -> str:
    nodes = [str(node) for node in adjacency.index]
    size = len(nodes)
    if size == 0:
        return ""
    values = adjacency.to_numpy(dtype=float)
    positions = {}
    radius = min(OVERVIEW_PANEL_W, OVERVIEW_PANEL_H) * 0.34
    center_x = OVERVIEW_PANEL_W / 2
    center_y = OVERVIEW_PANEL_H / 2
    for index, node in enumerate(nodes):
        angle = 2.0 * math.pi * index / size - math.pi / 2
        positions[node] = (
            center_x + radius * math.cos(angle),
            center_y + radius * math.sin(angle),
        )
    offdiag = values.copy()
    np.fill_diagonal(offdiag, 0.0)
    max_abs = float(np.max(np.abs(offdiag))) if np.isfinite(offdiag).any() else 1.0
    max_abs = max_abs if max_abs > 1e-12 else 1.0
    parts = [
        f'<rect x="0" y="0" width="{OVERVIEW_PANEL_W:.0f}" '
        f'height="{OVERVIEW_PANEL_H:.0f}" fill="#ffffff" stroke="#e3e8ef" '
        f'stroke-width="0.7"/>'
    ]
    edge_count = 0
    for row, source in enumerate(nodes):
        for column, target in enumerate(nodes):
            if row == column:
                continue
            weight = float(values[row, column])
            if not np.isfinite(weight) or abs(weight) < 1e-12:
                continue
            edge_count += 1
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            ratio = min(1.0, abs(weight) / max_abs)
            width = 0.45 + 1.15 * math.sqrt(ratio)
            color = "#B2182B" if weight > 0 else "#2166AC"
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-width="{width:.2f}" stroke-opacity="0.62"/>'
            )
    for node in nodes:
        x, y = positions[node]
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="#ffffff" '
            f'stroke="#4B5563" stroke-width="0.7"><title>{escape(node)}</title></circle>'
        )
        if size <= max_labels:
            anchor = "start" if x >= center_x else "end"
            dx = 3.5 if x >= center_x else -3.5
            parts.append(
                f'<text x="{x + dx:.1f}" y="{y + 2.2:.1f}" '
                f'text-anchor="{anchor}" font-size="6" fill="#333">'
                f"{escape(_short_label(node, 18))}</text>"
            )
    parts.append(
        f'<text x="4" y="{OVERVIEW_PANEL_H - 4:.0f}" font-size="6" fill="#666">'
        f"{size} nodes · {edge_count} edges</text>"
    )
    return "".join(parts)


def render_multilayer_overview(
    inter_cluster: dict[str, pd.DataFrame],
    intra_cluster: dict[str, dict[str, pd.DataFrame]],
    *,
    conditions: list[str] | None = None,
    modules: list[str] | None = None,
    title: str = "Multilayer IdopNetwork overview",
    subtitle: str = "",
) -> str:
    """Publication matrix of inter-module and intra-module networks."""
    condition_names = list(
        conditions
        if conditions is not None
        else dict.fromkeys([*inter_cluster, *intra_cluster])
    )
    module_names = list(
        modules
        if modules is not None
        else sorted({module for cluster_map in intra_cluster.values() for module in cluster_map})
    )
    if not condition_names:
        raise ValueError("No condition has a fitted network to plot")
    rows = [None, *module_names]
    columns = len(condition_names)
    width = (
        2 * OVERVIEW_MARGIN
        + OVERVIEW_ROW_LABEL
        + columns * OVERVIEW_PANEL_W
        + (columns - 1) * OVERVIEW_GAP
    )
    height = (
        OVERVIEW_MARGIN
        + 40
        + OVERVIEW_HEADER
        + len(rows) * OVERVIEW_PANEL_H
        + (len(rows) - 1) * OVERVIEW_GAP
        + 64
    )
    parts = [
        f'<svg id="ml-overview" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}pt" '
        f'height="{height:.0f}pt" font-family="Helvetica, Arial, sans-serif" '
        f'role="img" aria-label="{escape(title)}">',
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
        f'<text x="{OVERVIEW_MARGIN:.1f}" y="20" font-size="13" font-weight="700">'
        f"{escape(title)}</text>",
        f'<text x="{OVERVIEW_MARGIN:.1f}" y="34" font-size="9" fill="#444">'
        f"{escape(subtitle)}</text>",
    ]
    grid_left = OVERVIEW_MARGIN + OVERVIEW_ROW_LABEL
    grid_top = OVERVIEW_MARGIN + 40
    for column, condition in enumerate(condition_names):
        x = grid_left + column * (OVERVIEW_PANEL_W + OVERVIEW_GAP)
        parts.append(
            f'<rect x="{x:.1f}" y="{grid_top:.1f}" width="{OVERVIEW_PANEL_W:.0f}" '
            f'height="{OVERVIEW_HEADER:.0f}" fill="#F3F6FA" stroke="#e3e8ef" '
            f'stroke-width="0.7"/>'
        )
        parts.append(
            f'<text x="{x + OVERVIEW_PANEL_W / 2:.1f}" '
            f'y="{grid_top + OVERVIEW_HEADER / 2 + 3:.1f}" text-anchor="middle" '
            f'font-size="8.5" font-weight="600">{escape(condition)}</text>'
        )
    for row, module in enumerate(rows):
        y = grid_top + OVERVIEW_HEADER + row * (OVERVIEW_PANEL_H + OVERVIEW_GAP)
        label = "Inter-module" if module is None else module
        parts.append(
            f'<text x="{grid_left - 8:.1f}" y="{y + OVERVIEW_PANEL_H / 2:.1f}" '
            f'text-anchor="end" font-size="9" font-weight="600">{escape(label)}</text>'
        )
        for column, condition in enumerate(condition_names):
            x = grid_left + column * (OVERVIEW_PANEL_W + OVERVIEW_GAP)
            if module is None:
                frame = inter_cluster.get(condition)
            else:
                frame = intra_cluster.get(condition, {}).get(module)
            if frame is None or frame.empty:
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{OVERVIEW_PANEL_W:.0f}" '
                    f'height="{OVERVIEW_PANEL_H:.0f}" fill="#FAFAFA" '
                    f'stroke="#e3e8ef" stroke-width="0.7"/>'
                )
                parts.append(
                    f'<text x="{x + OVERVIEW_PANEL_W / 2:.1f}" '
                    f'y="{y + OVERVIEW_PANEL_H / 2:.1f}" text-anchor="middle" '
                    f'font-size="8" fill="#999">—</text>'
                )
                continue
            parts.append(f'<g transform="translate({x:.1f} {y:.1f})">')
            parts.append(_overview_panel(frame))
            parts.append("</g>")
    legend_y = height - 24
    parts.append(
        f'<line x1="{OVERVIEW_MARGIN}" y1="{legend_y}" x2="{OVERVIEW_MARGIN + 26}" '
        f'y2="{legend_y}" stroke="#B2182B" stroke-width="1.4"/>'
    )
    parts.append(
        f'<text x="{OVERVIEW_MARGIN + 32}" y="{legend_y + 3}" font-size="8.5">'
        f"positive edge</text>"
    )
    parts.append(
        f'<line x1="{OVERVIEW_MARGIN + 120}" y1="{legend_y}" '
        f'x2="{OVERVIEW_MARGIN + 146}" y2="{legend_y}" stroke="#2166AC" '
        f'stroke-width="1.4"/>'
    )
    parts.append(
        f'<text x="{OVERVIEW_MARGIN + 152}" y="{legend_y + 3}" font-size="8.5">'
        f"negative edge · circular layout · labels for ≤12 nodes</text>"
    )
    parts.append("</svg>")
    return (
        "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{PAGE_STYLE}</style></head><body>"
        '<main class="page"><div class="toolbar">'
        '<span class="spacer"></span>'
        '<select id="ml-dpi" aria-label="DPI"><option value="300" selected>300 dpi</option>'
        '<option value="600">600 dpi</option></select>'
        '<button type="button" id="ml-svg">Export SVG</button>'
        '<button type="button" id="ml-png">Export PNG</button>'
        "</div>"
        f"{''.join(parts)}</main><script>{OVERVIEW_SCRIPT}</script></body></html>"
    )
