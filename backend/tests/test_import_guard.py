from __future__ import annotations

import sys


def test_backend_does_not_import_streamlit():
    import app.api.artifacts  # noqa: F401
    import app.api.datasets  # noqa: F401
    import app.api.runs  # noqa: F401
    import app.job_runner  # noqa: F401
    import app.tasks.demo  # noqa: F401
    import app.worker  # noqa: F401

    assert "streamlit" not in sys.modules
