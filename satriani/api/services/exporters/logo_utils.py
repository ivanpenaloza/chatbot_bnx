"""Shared logo utility for exporters.

The original banamex.png is RGB (no alpha channel) with a solid white
background.  This module uses Pillow to strip white/near-white pixels,
saves the result to a temp PNG (once, on first use), and returns its path.

Result is module-level cached — Pillow runs exactly once per process.
"""
from __future__ import annotations

import atexit
import os
import tempfile
from typing import Optional

_API_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOGO_PATH = os.path.join(_API_DIR, 'images', 'banamex.png')

_cached_path: Optional[str] = None
_temp_file: Optional[str] = None


def _cleanup_temp() -> None:
    if _temp_file and os.path.exists(_temp_file):
        try:
            os.unlink(_temp_file)
        except OSError:
            pass


atexit.register(_cleanup_temp)


def get_logo_path() -> Optional[str]:
    """Return a path to the Banamex logo with its white background made
    transparent (RGBA PNG).

    The processed image is cached in a temp file so Pillow runs only on
    the first call.  Returns None when the source logo does not exist.
    Falls back to the original path if Pillow is unavailable.
    """
    global _cached_path, _temp_file

    if not os.path.isfile(LOGO_PATH):
        return None

    if _cached_path is not None:
        return _cached_path

    try:
        from PIL import Image

        img = Image.open(LOGO_PATH).convert("RGBA")
        pixels = img.load()
        width, height = img.size

        # The Banamex logo uses only red and dark navy blue — no logo
        # element has all RGB channels above 200.  Any pixel where all
        # channels exceed the threshold is therefore pure background.
        # A binary cutoff (no gradient) gives the cleanest result.
        threshold = 210

        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if r > threshold and g > threshold and b > threshold:
                    pixels[x, y] = (r, g, b, 0)

        fd, _temp_file = tempfile.mkstemp(suffix='.png', prefix='bnx_logo_')
        os.close(fd)
        img.save(_temp_file, 'PNG')
        _cached_path = _temp_file
        return _cached_path

    except Exception:
        # Pillow unavailable or image error — fall back to original
        _cached_path = LOGO_PATH
        return _cached_path
