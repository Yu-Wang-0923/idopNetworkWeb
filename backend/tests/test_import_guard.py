from __future__ import annotations

import sys


def test_backend_does_not_import_streamlit():
    import app.api.artifacts  # noqa: F401
    import app.api.datasets  # noqa: F401
    import app.api.runs  # noqa: F401
    import app.job_runner  # noqa: F401
    import app.pure.charts.curve_fitting_html  # noqa: F401
    import app.pure.charts.ecg_fitting_html  # noqa: F401
    import app.pure.charts.funclu_html  # noqa: F401
    import app.pure.curve_fitting  # noqa: F401
    import app.pure.fast_funclu_power  # noqa: F401
    import app.pure.funclu  # noqa: F401
    import app.pure.netrecon  # noqa: F401
    import app.pure.charts.netrecon_html  # noqa: F401
    import app.tasks.curve_fitting  # noqa: F401
    import app.tasks.demo  # noqa: F401
    import app.tasks.funclu  # noqa: F401
    import app.tasks.netanal  # noqa: F401
    import app.tasks.netrecon  # noqa: F401
    import app.worker  # noqa: F401

    assert "streamlit" not in sys.modules
