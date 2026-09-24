from dataclasses import dataclass


class ActionError(RuntimeError):
    pass


@dataclass
class AssistantResponse:
    spoken: str
    display: str | None = None

    def __post_init__(self) -> None:
        if self.display is None:
            self.display = self.spoken
