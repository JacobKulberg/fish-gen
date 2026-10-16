"""fishgen - procedural pixel-art sea creatures, rendered from realistic 3D models, with seamless swim and turn-around animations."""

__version__ = "0.1.0"

from .animation import Animation, FishSprite, generate  # noqa: E402
from .genome import FishGenome  # noqa: E402

__all__ = ["Animation", "FishGenome", "FishSprite", "generate", "__version__"]
