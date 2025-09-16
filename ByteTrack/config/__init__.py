"""Config package initializer.

Re-exports symbols from `config.py` so that `import config` exposes the
expected constants (MODEL_BODY_PATH, SOURCE_VIDEO_PATH, etc.). Without this
file Python created a namespace package and the inner module wasn't loaded,
causing AttributeError when accessing constants directly from `config`.
"""

from .config import *  # noqa: F401,F403
