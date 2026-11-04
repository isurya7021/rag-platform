from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    score: float | None = None