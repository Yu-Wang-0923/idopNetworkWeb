"""Task modules register themselves on import."""
from app.tasks import (  # noqa: F401
    curve_fitting,
    demo,
    funclu,
    netanal,
    netrecon,
    netrecon_multi,
)
