from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


ActionType = Literal["classify", "prioritize", "respond"]


class EmailAction(BaseModel):
    action_type: ActionType
    label: Optional[Literal["spam", "important"]] = None
    priority: Optional[Literal["low", "normal", "high"]] = None
    response_text: Optional[str] = None

    @model_validator(mode="after")
    def validate_payload(self) -> "EmailAction":
        if self.action_type == "classify" and self.label is None:
            raise ValueError("Classify actions must provide a label.")
        if self.action_type == "prioritize" and self.priority is None:
            raise ValueError("Prioritize actions must provide a priority.")
        if self.action_type == "respond" and (self.response_text is None or not self.response_text.strip()):
            raise ValueError("Respond actions must provide response_text.")
        return self


class ActionOption(BaseModel):
    action_type: ActionType
    label: Optional[str] = None
    priority: Optional[str] = None
    response_text: Optional[str] = None


class EmailRecord(BaseModel):
    id: str
    sender: str
    sender_name: str
    subject: str
    body: str
    received_at: str
    summary: str


class RewardSignal(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    delta: float = Field(ge=-0.2, le=0.4)
    score: float = Field(ge=0.0, le=1.0)


class HistoryEntry(BaseModel):
    step: int = Field(ge=1)
    expected_stage: str
    action: Dict[str, Any]
    delta_reward: float
    raw_reward: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    result: str
    error: Optional[str] = None
    response_quality_details: Optional[Dict[str, Any]] = None


class EmailObservation(BaseModel):
    env_name: str
    task_name: str
    objective: str
    stage: str
    step_count: int = Field(ge=0)
    remaining_steps: int = Field(ge=0)
    done: bool
    email: EmailRecord
    allowed_actions: List[ActionOption]
    history: List[HistoryEntry]


class StepInfo(BaseModel):
    task_name: str
    current_stage: str
    completed_stages: List[str]
    reward_signal: RewardSignal
    result: str
    error: Optional[str] = None
    success: bool
    termination_reason: Optional[str] = None
    response_quality_details: Optional[Dict[str, Any]] = None
    ground_truth: Optional[Dict[str, Any]] = None


class EpisodeState(BaseModel):
    env_name: str
    initialized: bool
    task_name: Optional[str] = None
    objective: Optional[str] = None
    email_id: Optional[str] = None
    done: bool
    step_count: int = Field(default=0, ge=0)
    current_stage: Optional[str] = None
    completed_stages: List[str] = Field(default_factory=list)
    history: List[HistoryEntry] = Field(default_factory=list)
    raw_reward: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_balance: float = 0.0
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    success: bool = False
    termination_reason: Optional[str] = None
    reward_signal: Optional[RewardSignal] = None
    ground_truth: Optional[Dict[str, Any]] = None


class ResetRequest(BaseModel):
    task_name: str = Field(default="easy")
    email_id: Optional[str] = None


class StepRequest(BaseModel):
    action: EmailAction


class ResetResponse(BaseModel):
    observation: EmailObservation


class StepResponse(BaseModel):
    observation: EmailObservation
    reward: float
    done: bool
    info: StepInfo


class HealthResponse(BaseModel):
    status: str
    env_name: str


class RootResponse(BaseModel):
    message: str
    env_name: str
    endpoints: List[str]


class TaskSummary(BaseModel):
    objective: str
    stages: List[str]
    max_steps: int
    max_reward: float


class TaskCatalog(BaseModel):
    env_name: str
    tasks: Dict[str, TaskSummary]
