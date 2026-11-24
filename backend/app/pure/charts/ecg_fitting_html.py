"""HTML/SVG figures for dynamic ECG fitting.

Mirrors the upstream matplotlib scheme in
``idopnetwork.curve_fitting.ecg_plot.plot_fitting`` (one figure per lead, the
full record above the first 250 samples) but emits a self-contained HTML page
so the browser can show and download it without matplotlib images.
"""
from __future__ import annotations

import math
import re
from html import escape

import numpy as np
import pandas as pd

# Figure geometry in PostScript points (11 x 7 in, like the upstream figsize).
WIDTH = 792.0
HEIGHT = 504.0
PAD_LEFT = 64.0
PAD_RIGHT = 16.0
PAD_TOP = 44.0
PAD_BOTTOM = 42.0
PANEL_GAP = 72.0
AXES_WIDTH = WIDTH - PAD_LEFT - PAD_RIGHT
PANEL_HEIGHT = (HEIGHT - PAD_TOP - PAD_BOTTOM - PANEL_GAP) / 2

OBSERVED_COLOR = "#4d4d4d"
FIT_COLOR = "#d73027"
AXIS_COLOR = "#333333"
SHADE_COLOR = "#ebebeb"
FONT_FAMILY = 'Helvetica, Arial, "DejaVu Sans", sans-serif'
TITLE_SIZE = 16.0
LABEL_SIZE = 13.0
TICK_SIZE = 12.0
LEGEND_SIZE = 13.0
MARKER_RADIUS = 2.4
MARKER_WIDTH = 0.8
LINE_WIDTH = 1.4

MAX_FULL_MARKERS = 2000
MAX_CURVE_POINTS = 4000


def _time_axis(
    observed: pd.DataFrame, sampling_rate: float
) -> tuple[np.ndarray, list[str]]:
    if "time" in observed.columns:
        return (
            observed["time"].to_numpy(dtype=float),
            [str(column) for column in observed.columns if column != "time"],
        )
    if str(observed.index.name).lower() == "time":
        return observed.index.to_numpy(dtype=float), [str(c) for c in observed.columns]
    return np.arange(len(observed), dtype=float) / float(sampling_rate), [
        str(column) for column in observed.columns
    ]


def available_leads(observed: pd.DataFrame, fitted: pd.DataFrame) -> list[str]:
    """Leads whose observed and fitted series are fully finite."""
    leads = [
        str(column)
        for column in observed.columns
        if str(column) != "time" and column in fitted.columns
    ]
    return [
        lead
        for lead in leads
        if np.isfinite(observed[lead].to_numpy(dtype=float)).all()
        and np.isfinite(fitted[lead].to_numpy(dtype=float)).all()
    ]


def lead_slug(lead: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", lead).strip("-")
    return slug or "lead"


def _nice_ticks(lo: float, hi: float, count: int) -> list[float]:
    if not hi > lo:
        return [lo]
    raw = (hi - lo) / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = magnitude
    for multiplier in (1, 2, 2.5, 5, 10):
        step = multiplier * magnitude
        if (hi - lo) / step <= count + 0.5:
            break
    ticks = []
    value = math.ceil(lo / step - 1e-9) * step
    while value <= hi + step * 1e-9:
        ticks.append(value)
        value += step
    return ticks


def _tick_decimals(step: float) -> int:
    text = f"{step:.6f}".rstrip("0").rstrip(".")
    return len(text.split(".")[1]) if "." in text else 0


def _format_tick(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    return "0" if text in {"-0", "-0.0", "-0.00"} else text


def _decimate(length: int, target: int) -> list[int]:
    if length <= target:
        return list(range(length))
    stride = max(1, math.ceil(length / target))
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def _span(values: np.ndarray) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    margin = (high - low) * 0.12
    if margin <= 0:
        margin = max(abs(high) * 0.1, 1.0)
    return low - margin, high + margin


def _text_width(text: str, size: float) -> float:
    return 0.55 * size * len(text)


def _panel_svg(
    *,
    lead: str,
    time: np.ndarray,
    observed: np.ndarray,
    fitted: np.ndarray,
    top: float,
    letter: str,
    title: str,
    x_low: float,
    x_high: float,
    show_shade: float | None,
) -> str:
    left = PAD_LEFT
    bottom = top + PANEL_HEIGHT
    width = AXES_WIDTH
    height = PANEL_HEIGHT
    y_low, y_high = _span(np.concatenate([observed, fitted]))
    x_ticks = _nice_ticks(x_low, x_high, 6)
    y_ticks = _nice_ticks(y_low, y_high, 5)
    x_step = x_ticks[1] - x_ticks[0] if len(x_ticks) > 1 else 1.0
    y_step = y_ticks[1] - y_ticks[0] if len(y_ticks) > 1 else 1.0
    x_decimals = _tick_decimals(x_step)
    y_decimals = _tick_decimals(y_step)

    def sx(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * width

    def sy(value: float) -> float:
        return top + (y_high - value) / (y_high - y_low) * height

    parts: list[str] = []
    clip_id = f"clip-{letter}-{lead_slug(lead)}"

    if show_shade is not None:
        x0 = sx(x_low)
        x1 = sx(show_shade)
        parts.append(
            f'<rect x="{x0:.2f}" y="{top:.2f}" width="{x1 - x0:.2f}" '
            f'height="{height:.2f}" fill="{SHADE_COLOR}"/>'
        )

    parts.append(
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" '
        f'height="{height:.2f}" fill="none" stroke="{AXIS_COLOR}" '
        f'stroke-width="0.8"/>'
    )

    marker_indices = _decimate(len(observed), MAX_FULL_MARKERS)
    markers = "".join(
        f'<circle cx="{sx(time[index]):.2f}" '
        f'cy="{sy(observed[index]):.2f}" r="{MARKER_RADIUS}" fill="none" '
        f'stroke="{OBSERVED_COLOR}" stroke-width="{MARKER_WIDTH}"/>'
        for index in marker_indices
    )
    curve_indices = _decimate(len(fitted), MAX_CURVE_POINTS)
    path = " ".join(
        f'{"L" if position else "M"}{sx(time[index]):.2f},{sy(fitted[index]):.2f}'
        for position, index in enumerate(curve_indices)
    )
    parts.append(f'<clipPath id="{clip_id}"><rect x="{left:.2f}" y="{top:.2f}" '
                 f'width="{width:.2f}" height="{height:.2f}"/></clipPath>')
    parts.append(f'<g clip-path="url(#{clip_id})">{markers}'
                 f'<path d="{path}" fill="none" stroke="{FIT_COLOR}" '
                 f'stroke-width="{LINE_WIDTH}" stroke-linejoin="round" '
                 f'stroke-linecap="round"/></g>')

    tick_marks = []
    tick_labels = []
    for value in x_ticks:
        px = sx(value)
        tick_marks.append(
            f'<line x1="{px:.2f}" y1="{bottom:.2f}" x2="{px:.2f}" '
            f'y2="{bottom + 3:.2f}" stroke="{AXIS_COLOR}" stroke-width="0.8"/>'
        )
        tick_labels.append(
            f'<text x="{px:.2f}" y="{bottom + 16:.2f}" text-anchor="middle" '
            f'font-size="{TICK_SIZE}">{_format_tick(value, x_decimals)}</text>'
        )
    for value in y_ticks:
        py = sy(value)
        tick_marks.append(
            f'<line x1="{left - 3:.2f}" y1="{py:.2f}" x2="{left:.2f}" '
            f'y2="{py:.2f}" stroke="{AXIS_COLOR}" stroke-width="0.8"/>'
        )
        tick_labels.append(
            f'<text x="{left - 6:.2f}" y="{py + 4:.2f}" text-anchor="end" '
            f'font-size="{TICK_SIZE}">{_format_tick(value, y_decimals)}</text>'
        )
    parts.extend(tick_marks)
    parts.extend(tick_labels)

    parts.append(
        f'<text x="{left + width / 2:.2f}" y="{bottom + 34:.2f}" '
        f'text-anchor="middle" font-size="{LABEL_SIZE}">Time (s)</text>'
    )
    center_y = top + height / 2
    parts.append(
        f'<text x="18" y="{center_y:.2f}" text-anchor="middle" '
        f'font-size="{LABEL_SIZE}" transform="rotate(-90 18 {center_y:.2f})">'
        f"Signal (stored units)</text>"
    )
    parts.append(
        f'<text x="{left:.2f}" y="{top - 12:.2f}" font-size="{TITLE_SIZE}">'
        f"{escape(letter)}  {escape(title)}</text>"
    )
    return "".join(parts)


def _legend(right: float, baseline: float) -> str:
    gap = 6.0
    line_length = 22.0
    observed_text = "Observed"
    fit_text = "SWT fit"
    width = (
        7.0
        + gap
        + _text_width(observed_text, LEGEND_SIZE)
        + 2 * gap
        + line_length
        + gap
        + _text_width(fit_text, LEGEND_SIZE)
    )
    x = right - width
    center_y = baseline - 4.5
    parts = [
        f'<circle cx="{x + 3.5:.2f}" cy="{center_y:.2f}" r="3.2" fill="none" '
        f'stroke="{OBSERVED_COLOR}" stroke-width="{MARKER_WIDTH}"/>',
        f'<text x="{x + 7 + gap:.2f}" y="{baseline:.2f}" font-size="{LEGEND_SIZE}">'
        f"{observed_text}</text>",
    ]
    line_x = x + 7 + gap + _text_width(observed_text, LEGEND_SIZE) + 2 * gap
    parts.append(
        f'<line x1="{line_x:.2f}" y1="{center_y:.2f}" x2="{line_x + line_length:.2f}" '
        f'y2="{center_y:.2f}" stroke="{FIT_COLOR}" stroke-width="{LINE_WIDTH}"/>'
    )
    parts.append(
        f'<text x="{line_x + line_length + gap:.2f}" y="{baseline:.2f}" '
        f'font-size="{LEGEND_SIZE}">{fit_text}</text>'
    )
    return "".join(parts)


def _lead_section(
    *,
    lead: str,
    time: np.ndarray,
    observed: np.ndarray,
    fitted: np.ndarray,
    zoom: int,
) -> str:
    full_top = PAD_TOP
    zoom_top = PAD_TOP + PANEL_HEIGHT + PANEL_GAP
    zoom_time = float(time[zoom - 1])
    body = [
        _panel_svg(
            lead=lead,
            time=time,
            observed=observed,
            fitted=fitted,
            top=full_top,
            letter="a",
            title=f"Lead {lead} · Full record",
            x_low=float(time[0]),
            x_high=float(time[-1]),
            show_shade=zoom_time,
        ),
        _legend(PAD_LEFT + AXES_WIDTH, full_top - 12),
        _panel_svg(
            lead=lead,
            time=time[:zoom],
            observed=observed[:zoom],
            fitted=fitted[:zoom],
            top=zoom_top,
            letter="b",
            title=f"Lead {lead} · First {zoom} samples",
            x_low=float(time[0]),
            x_high=zoom_time,
            show_shade=None,
        ),
    ]
    return (
        f'<section class="lead" id="lead-{escape(lead_slug(lead))}">'
        f'<svg viewBox="0 0 {WIDTH:g} {HEIGHT:g}" role="img" '
        f'aria-label="{escape(f"Lead {lead} ECG fitting")}" '
        f'font-family=\'{FONT_FAMILY}\'>'
        f"<title>{escape(f'Lead {lead} — full record and first {zoom} samples')}</title>"
        f"{''.join(body)}</svg></section>"
    )


PAGE_STYLE = """
  *{box-sizing:border-box}
  html,body{margin:0;background:#fff;color:#111}
  body{font:13px/1.4 Helvetica,Arial,"DejaVu Sans",sans-serif}
  .page{max-width:1120px;margin:0 auto;padding:6px 14px 24px}
  .pager{display:flex;align-items:center;gap:10px;margin:0 0 8px;padding:4px 2px 8px;
         border-bottom:1px solid #e5e8ee}
  .pager button{border:1px solid #c9d3e0;background:#fff;border-radius:4px;
                padding:3px 11px;font-size:13px;line-height:1.4;color:#245db4;cursor:pointer}
  .pager button:hover:not(:disabled){background:#eef4fc}
  .pager button:disabled{opacity:.35;cursor:default}
  .pager select{border:1px solid #c9d3e0;border-radius:4px;padding:2px 4px;
                font-size:12px;color:#111;background:#fff}
  .pager .counter{color:#666;font-size:12px;font-variant-numeric:tabular-nums}
  section.lead{margin:0 0 6px;scroll-margin-top:8px}
  section.lead[hidden]{display:none}
  svg{display:block;width:100%;height:auto}
  @media print{section.lead{break-inside:avoid}}
"""

PAGE_SCRIPT = """
(function () {
  var sections = Array.prototype.slice.call(document.querySelectorAll("section.lead"));
  var pager = document.getElementById("lead-pager");
  var select = document.getElementById("lead-select");
  var counter = document.getElementById("lead-counter");
  var previous = document.getElementById("lead-prev");
  var next = document.getElementById("lead-next");
  if (!sections.length || !select || !pager) return;
  if (sections.length < 2) { pager.hidden = true; return; }
  var current = 0;
  function show(index) {
    current = Math.max(0, Math.min(sections.length - 1, index));
    sections.forEach(function (section, position) {
      section.hidden = position !== current;
    });
    select.value = String(current);
    counter.textContent = (current + 1) + " / " + sections.length;
    previous.disabled = current === 0;
    next.disabled = current === sections.length - 1;
    window.scrollTo(0, 0);
  }
  select.addEventListener("change", function () { show(Number(select.value)); });
  previous.addEventListener("click", function () { show(current - 1); });
  next.addEventListener("click", function () { show(current + 1); });
  show(0);
})();
"""


def render(
    observed: pd.DataFrame,
    fitted: pd.DataFrame,
    *,
    sampling_rate: float = 100.0,
    zoom_samples: int = 250,
    title: str | None = None,
) -> str:
    """Render every finite lead as an HTML page in the upstream plot style."""
    if len(observed) != len(fitted):
        raise ValueError("Observed and fitted ECG frames must have the same length")
    if observed.empty:
        raise ValueError("No ECG sample is available for plotting")
    time, _ = _time_axis(observed, sampling_rate)
    leads = available_leads(observed, fitted)
    if not leads:
        raise ValueError("No finite ECG lead is available for plotting")
    zoom = max(1, min(int(zoom_samples), len(time)))
    sections = [
        _lead_section(
            lead=lead,
            time=time,
            observed=observed[lead].to_numpy(dtype=float),
            fitted=fitted[lead].to_numpy(dtype=float),
            zoom=zoom,
        )
        for lead in leads
    ]
    options = "".join(
        f'<option value="{position}">Lead {escape(lead)}</option>'
        for position, lead in enumerate(leads)
    )
    document_title = escape(title or "ECG Fitting")
    return (
        "<!DOCTYPE html>\n"
        '<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{document_title}</title>\n<style>{PAGE_STYLE}</style>\n"
        "</head>\n<body>\n<main class=\"page\">\n"
        '<nav class="pager" id="lead-pager" aria-label="Lead pages">'
        '<button type="button" id="lead-prev" aria-label="Previous lead">‹</button>'
        f'<select id="lead-select" aria-label="Lead">{options}</select>'
        '<span class="counter" id="lead-counter"></span>'
        '<button type="button" id="lead-next" aria-label="Next lead">›</button>'
        "</nav>\n"
        f"{''.join(sections)}\n"
        f"</main>\n<script>{PAGE_SCRIPT}</script>\n</body>\n</html>\n"
    )
