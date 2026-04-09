from __future__ import annotations

from typing import Optional

import requests

from models import EmailAction, EpisodeState, ResetResponse, StepResponse


class EmailTriageEnvClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, task_name: str = "easy", email_id: Optional[str] = None) -> ResetResponse:
        payload = {"task_name": task_name, "email_id": email_id}
        response = self.session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        response.raise_for_status()
        return ResetResponse.model_validate(response.json())

    def step(self, action: EmailAction) -> StepResponse:
        response = self.session.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump(exclude_none=True)},
            timeout=30,
        )
        response.raise_for_status()
        return StepResponse.model_validate(response.json())

    def state(self) -> EpisodeState:
        response = self.session.get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return EpisodeState.model_validate(response.json())

    def close(self) -> None:
        self.session.close()
