from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


Message = dict[str, str]


@dataclass
class ChatSession:
    system_prompt: Optional[str] = None
    messages: list[Message] = field(default_factory=list)

    def reset(self) -> None:
        self.messages.clear()

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def remove_last_message(self) -> None:
        if self.messages:
            self.messages.pop()

    def prompt_messages(
        self,
        max_context_messages: Optional[int] = None,
    ) -> list[Message]:
        prompt: list[Message] = []
        if self.system_prompt:
            prompt.append({"role": "system", "content": self.system_prompt})

        if max_context_messages is None or max_context_messages <= 0:
            prompt.extend(self.messages)
            return prompt

        prompt.extend(self.messages[-max_context_messages:])
        return prompt

    def turn_count(self) -> int:
        user_turns = sum(1 for message in self.messages if message["role"] == "user")
        return user_turns
