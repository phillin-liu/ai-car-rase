"""AI Car Race - a 3D two-player racing game with AI/LLM drivers.

Modules
-------
config      : dataclass based configuration + persistence
mathutil    : small numpy matrix/vector helpers
core_types  : shared Action / Observation dataclasses
track       : random track map generation and queries
cars        : car physics, wall + obstacle interaction
items       : item definitions and cooldown controller
ai          : the vision-LLM driver and its factory
game        : race engine (one match)
analysis    : post match scoring + suggestions
project     : project management / result persistence
render      : OpenGL renderer
runner      : multi-match session runner (windowed / headless)
"""

__version__ = "1.0.0"
