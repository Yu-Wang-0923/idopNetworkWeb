"""Runtime bootstrap: non-interactive matplotlib and CJK fonts."""
from __future__ import annotations

import os
from pathlib import Path

_initialized = False


def _find_cjk_font() -> Path | None:
    candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    return next((path for path in candidates if path.exists()), None)


def setup_runtime() -> None:
    global _initialized
    if _initialized:
        return
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["axes.unicode_minus"] = False
        font_path = _find_cjk_font()
        if font_path is not None:
            from matplotlib import font_manager

            font_manager.fontManager.addfont(str(font_path))
            family = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
    except ImportError:
        pass
    _initialized = True
