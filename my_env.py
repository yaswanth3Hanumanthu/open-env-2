from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from models import (
    ActionOption,
    EmailAction,
    EmailObservation,
    EpisodeState,
    HealthResponse,
    HistoryEntry,
    RewardSignal,
    ResetRequest,
    ResetResponse,
    RootResponse,
    StepInfo,
    StepRequest,
    StepResponse,
    TaskCatalog,
    TaskSummary,
)


ENV_NAME = "email-triage-smart-response-system"


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def normalize_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    objective: str
    stages: List[str]
    max_steps: int
    max_reward: float
    success_threshold: float


@dataclass(frozen=True)
class EmailExample:
    email_id: str
    sender: str
    sender_name: str
    subject: str
    body: str
    received_at: str
    classification: str
    priority: str
    response_requirements: List[List[str]]
    response_style: str
    summary: str


TASK_DEFINITIONS: Dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        name="easy",
        objective="Classify the email as spam or important.",
        stages=["classify"],
        max_steps=2,
        max_reward=0.3,
        success_threshold=0.6,
    ),
    "medium": TaskDefinition(
        name="medium",
        objective="Classify the email and assign the correct priority.",
        stages=["classify", "prioritize"],
        max_steps=4,
        max_reward=0.6,
        success_threshold=0.6,
    ),
    "hard": TaskDefinition(
        name="hard",
        objective="Classify the email, assign priority, and draft a smart response.",
        stages=["classify", "prioritize", "respond"],
        max_steps=5,
        max_reward=1.0,
        success_threshold=0.6,
    ),
}


EMAIL_EXAMPLES: Dict[str, EmailExample] = {
    "phishing_payroll": EmailExample(
        email_id="phishing_payroll",
        sender="security-alert@paypaI-support-mail.com",
        sender_name="Payroll Security Team",
        subject="Immediate action required: payroll account verification",
        body=(
            "We detected unusual payroll activity on your profile. Confirm your password "
            "within 30 minutes or your account will be suspended. Use the secure link below "
            "to avoid disruption: http://paypaI-secure-check.example"
        ),
        received_at="2026-04-07T08:15:00Z",
        classification="spam",
        priority="low",
        response_requirements=[
            ["do not reply", "will not respond", "won't respond"],
            ["spam", "phishing", "malicious"],
            ["block", "report", "quarantine"],
        ],
        response_style="Do not engage. Mark as phishing and route to spam quarantine.",
        summary="Credential phishing email pretending to be a payroll security notice.",
    ),
    "gift_card_fraud": EmailExample(
        email_id="gift_card_fraud",
        sender="ceo-office@urgent-board-mail.example",
        sender_name="Executive Office",
        subject="Need 12 gift cards in the next hour",
        body=(
            "I am stepping into a board meeting and need you to purchase 12 gift cards "
            "for client appreciation right away. Reply only when completed and send the codes "
            "back immediately. Keep this confidential."
        ),
        received_at="2026-04-07T10:40:00Z",
        classification="spam",
        priority="low",
        response_requirements=[
            ["do not reply", "will not respond", "won't respond"],
            ["spam", "fraud", "scam"],
            ["report", "security", "block"],
        ],
        response_style="Treat as business-email compromise and escalate to security, not the sender.",
        summary="Classic executive impersonation scam asking for gift cards.",
    ),
    "candidate_reschedule": EmailExample(
        email_id="candidate_reschedule",
        sender="mia.patel@gmail.com",
        sender_name="Mia Patel",
        subject="Request to reschedule tomorrow's interview",
        body=(
            "Hi team, my inbound flight was delayed by the weather and I will not land in time "
            "for tomorrow morning's interview. Could we move the conversation to tomorrow afternoon "
            "or Friday morning instead? I am still very interested in the role. Thanks, Mia"
        ),
        received_at="2026-04-07T13:20:00Z",
        classification="important",
        priority="normal",
        response_requirements=[
            ["thank you", "thanks"],
            ["reschedule", "move", "adjust"],
            ["tomorrow afternoon", "friday morning", "calendar invite", "availability"],
            ["confirm", "let me know"],
        ],
        response_style="Acknowledge the delay, confirm rescheduling is fine, and offer next steps.",
        summary="Legitimate recruiting email asking to move an interview slot.",
    ),
    "finance_followup": EmailExample(
        email_id="finance_followup",
        sender="ap@northstarpackaging.com",
        sender_name="Northstar Packaging AP",
        subject="Invoice 4471 approval needed this week",
        body=(
            "Hello, we are following up on invoice 4471 for the March packing materials shipment. "
            "The payment team flagged that the PO match is still pending on your side. Could someone "
            "review the approval status before Friday so we can close the month cleanly?"
        ),
        received_at="2026-04-07T15:05:00Z",
        classification="important",
        priority="normal",
        response_requirements=[
            ["thank you", "thanks"],
            ["review", "approve", "finance", "accounts payable"],
            ["before friday", "this week", "timeline", "update"],
            ["confirm", "follow up", "let you know"],
        ],
        response_style="Acknowledge the request, confirm review with finance, and provide an update window.",
        summary="Vendor payment follow-up that needs attention but is not an emergency.",
    ),
    "customer_api_outage": EmailExample(
        email_id="customer_api_outage",
        sender="maya.chen@brightledger.io",
        sender_name="Maya Chen",
        subject="Urgent: production API errors blocking our launch",
        body=(
            "Our team started seeing repeated 500 errors from your reconciliation API after today's "
            "deploy. This is blocking a customer launch scheduled for this afternoon and finance cannot "
            "complete settlements. Please let us know what is happening and when we can expect an update."
        ),
        received_at="2026-04-08T06:55:00Z",
        classification="important",
        priority="high",
        response_requirements=[
            ["sorry", "apologize"],
            ["investigating", "reviewing", "engineering", "incident"],
            ["within the next hour", "today", "eta", "update"],
            ["call", "walk through", "meeting", "next steps"],
        ],
        response_style="Acknowledge the outage, apologize, share an immediate investigation plan, and offer a live follow-up.",
        summary="High-priority customer incident with production impact and time pressure.",
    ),
    "enterprise_renewal": EmailExample(
        email_id="enterprise_renewal",
        sender="nora.ross@fabrikamhealth.com",
        sender_name="Nora Ross",
        subject="Security review blocker for renewal approval today",
        body=(
            "Hi account team, our procurement approval is ready to go but legal still needs the updated "
            "security addendum that was mentioned last week. The contract renewal has to be signed today "
            "to avoid a service gap for our clinicians. Can you coordinate the remaining review items "
            "as soon as possible?"
        ),
        received_at="2026-04-08T07:40:00Z",
        classification="important",
        priority="high",
        response_requirements=[
            ["thank you", "appreciate"],
            ["security", "legal", "addendum", "review"],
            ["today", "this afternoon", "within the next hour", "update"],
            ["coordinate", "call", "follow up", "next steps"],
        ],
        response_style="Acknowledge urgency, confirm coordination with legal/security, and commit to a same-day update.",
        summary="Renewal blocker that threatens a production service gap for an enterprise customer.",
    ),
}


TASK_EMAIL_POOL: Dict[str, List[str]] = {
    "easy": ["phishing_payroll", "gift_card_fraud"],
    "medium": ["candidate_reschedule", "finance_followup"],
    "hard": ["customer_api_outage", "enterprise_renewal"],
}


class EmailTriageEnv:
    def __init__(self) -> None:
        self.env_name = ENV_NAME
        self._task_offsets = {task_name: 0 for task_name in TASK_DEFINITIONS}
        self._episode: Optional[Dict[str, Any]] = None

    def reset(self, task_name: str = "easy", email_id: Optional[str] = None) -> EmailObservation:
        if task_name not in TASK_DEFINITIONS:
            raise ValueError(f"Unknown task_name '{task_name}'. Expected one of {list(TASK_DEFINITIONS)}.")

        task = TASK_DEFINITIONS[task_name]
        email = self._select_email(task_name=task_name, email_id=email_id)
        self._episode = {
            "task": task,
            "email": email,
            "current_stage_index": 0,
            "step_count": 0,
            "done": False,
            "reward_balance": 0.0,
            "raw_reward": 0.0,
            "score": 0.0,
            "success": False,
            "history": [],
            "completed_stages": [],
            "last_quality_details": None,
            "last_error": None,
            "termination_reason": None,
        }
        return self._build_observation()

    def step(self, action: EmailAction | Dict[str, Any]) -> Tuple[EmailObservation, float, bool, StepInfo]:
        if self._episode is None:
            raise RuntimeError("Environment has not been reset. Call reset() before step().")

        if self._episode["done"]:
            observation = self._build_observation()
            info = self._build_info(delta_reward=0.0, error="Episode already completed.", result="noop")
            return observation, round(self._episode["raw_reward"], 2), True, info

        self._episode["step_count"] += 1
        expected_stage = self._current_stage()
        delta_reward = 0.0
        error: Optional[str] = None
        result = "incorrect"
        stage_completed = False
        quality_details_for_info: Optional[Dict[str, Any]] = None

        action_payload: Dict[str, Any]
        action_model: Optional[EmailAction] = None

        if isinstance(action, EmailAction):
            action_model = action
            action_payload = action.model_dump(exclude_none=True)
        elif isinstance(action, dict):
            action_payload = dict(action)
            try:
                action_model = EmailAction.model_validate(action)
                action_payload = action_model.model_dump(exclude_none=True)
            except ValidationError as exc:
                first_error = exc.errors()[0].get("msg", "Invalid action payload.")
                error = f"Invalid action payload: {first_error}"
                delta_reward = -0.2
        else:
            action_payload = {"invalid_action": str(action)}
            error = "Action must be a dictionary."
            delta_reward = -0.2

        if action_model is not None:
            action_type = action_model.action_type
            if action_type != expected_stage:
                error = f"Expected action_type '{expected_stage}' but received '{action_type}'."
                delta_reward = -0.2
            elif expected_stage == "classify":
                predicted = str(action_model.label or "").strip().lower()
                if predicted == self._episode["email"].classification:
                    delta_reward = 0.3
                    result = "correct"
                    stage_completed = True
                else:
                    error = f"Classification '{predicted or 'missing'}' is incorrect for this email."
                    delta_reward = -0.2
            elif expected_stage == "prioritize":
                predicted = str(action_model.priority or "").strip().lower()
                if predicted == self._episode["email"].priority:
                    delta_reward = 0.3
                    result = "correct"
                    stage_completed = True
                else:
                    error = f"Priority '{predicted or 'missing'}' is incorrect for this email."
                    delta_reward = -0.2
            elif expected_stage == "respond":
                response_text = str(action_model.response_text or "").strip()
                quality, quality_details = self._score_response(response_text)
                quality_details_for_info = quality_details
                if quality <= 0.0:
                    error = "Response quality was too low or empty."
                    delta_reward = -0.2
                else:
                    delta_reward = round(0.4 * quality, 2)
                    result = "partial" if quality < 0.85 else "correct"
                    stage_completed = True

        self._episode["reward_balance"] += delta_reward
        self._episode["raw_reward"] = clamp(self._episode["reward_balance"], 0.0, 1.0)
        self._episode["score"] = self._normalized_score()
        self._episode["last_error"] = error
        self._episode["last_quality_details"] = quality_details_for_info

        if stage_completed:
            self._episode["completed_stages"].append(expected_stage)
            self._episode["current_stage_index"] += 1

        history_item = {
            "step": self._episode["step_count"],
            "expected_stage": expected_stage,
            "action": action_payload,
            "delta_reward": round(delta_reward, 2),
            "raw_reward": round(self._episode["raw_reward"], 2),
            "score": round(self._episode["score"], 2),
            "result": result,
            "error": error,
            "response_quality_details": quality_details_for_info,
        }
        self._episode["history"].append(history_item)

        if self._episode["current_stage_index"] >= len(self._episode["task"].stages):
            self._episode["done"] = True
            self._episode["termination_reason"] = "completed"
        elif self._episode["step_count"] >= self._episode["task"].max_steps:
            self._episode["done"] = True
            self._episode["termination_reason"] = "max_steps_reached"

        self._episode["success"] = bool(
            self._episode["done"]
            and self._episode["termination_reason"] == "completed"
            and self._episode["score"] >= self._episode["task"].success_threshold
        )

        observation = self._build_observation()
        info = self._build_info(delta_reward=delta_reward, error=error, result=result)
        return observation, round(self._episode["raw_reward"], 2), self._episode["done"], info

    def state(self) -> EpisodeState:
        if self._episode is None:
            return EpisodeState(
                env_name=self.env_name,
                initialized=False,
                done=False,
                score=0.0,
                raw_reward=0.0,
                reward_balance=0.0,
                history=[],
            )

        email = self._episode["email"]
        return EpisodeState(
            env_name=self.env_name,
            initialized=True,
            task_name=self._episode["task"].name,
            objective=self._episode["task"].objective,
            email_id=email.email_id,
            done=self._episode["done"],
            step_count=self._episode["step_count"],
            current_stage=self._current_stage(),
            completed_stages=list(self._episode["completed_stages"]),
            history=[HistoryEntry.model_validate(item) for item in self._episode["history"]],
            raw_reward=round(self._episode["raw_reward"], 2),
            reward_balance=round(self._episode["reward_balance"], 2),
            score=round(self._episode["score"], 2),
            success=self._episode["success"],
            termination_reason=self._episode["termination_reason"],
            reward_signal=RewardSignal(
                value=round(self._episode["raw_reward"], 2),
                delta=0.0,
                score=round(self._episode["score"], 2),
            ),
            ground_truth={
                "classification": email.classification,
                "priority": email.priority,
                "response_style": email.response_style,
            },
        )

    def _select_email(self, task_name: str, email_id: Optional[str]) -> EmailExample:
        if email_id is not None:
            if email_id not in EMAIL_EXAMPLES:
                raise ValueError(f"Unknown email_id '{email_id}'.")
            return EMAIL_EXAMPLES[email_id]

        pool = TASK_EMAIL_POOL[task_name]
        current_index = self._task_offsets[task_name] % len(pool)
        self._task_offsets[task_name] += 1
        return EMAIL_EXAMPLES[pool[current_index]]

    def _current_stage(self) -> str:
        if self._episode is None:
            return "uninitialized"
        if self._episode["done"]:
            return "completed"
        task = self._episode["task"]
        return task.stages[self._episode["current_stage_index"]]

    def _normalized_score(self) -> float:
        if self._episode is None:
            return 0.0
        task = self._episode["task"]
        if task.max_reward <= 0:
            return 0.0
        return round(clamp(self._episode["raw_reward"] / task.max_reward, 0.0, 1.0), 2)

    def _build_observation(self) -> EmailObservation:
        if self._episode is None:
            raise RuntimeError("Environment has not been reset. Call reset() before requesting observation.")

        email = self._episode["email"]
        task = self._episode["task"]
        stage = self._current_stage()
        return EmailObservation(
            env_name=self.env_name,
            task_name=task.name,
            objective=task.objective,
            stage=stage,
            step_count=self._episode["step_count"],
            remaining_steps=max(0, task.max_steps - self._episode["step_count"]),
            done=self._episode["done"],
            email={
                "id": email.email_id,
                "sender": email.sender,
                "sender_name": email.sender_name,
                "subject": email.subject,
                "body": email.body,
                "received_at": email.received_at,
                "summary": email.summary,
            },
            allowed_actions=self._allowed_actions_for_stage(stage),
            history=[HistoryEntry.model_validate(item) for item in self._episode["history"]],
        )

    def _allowed_actions_for_stage(self, stage: str) -> List[ActionOption]:
        if stage == "classify":
            return [
                ActionOption(action_type="classify", label="spam"),
                ActionOption(action_type="classify", label="important"),
            ]
        if stage == "prioritize":
            return [
                ActionOption(action_type="prioritize", priority="low"),
                ActionOption(action_type="prioritize", priority="normal"),
                ActionOption(action_type="prioritize", priority="high"),
            ]
        if stage == "respond":
            return [
                ActionOption(
                    action_type="respond",
                    response_text="A concise, professional reply that acknowledges the email and proposes next steps.",
                )
            ]
        return []

    def _score_response(self, response_text: str) -> Tuple[float, Dict[str, Any]]:
        if self._episode is None:
            return 0.0, {"error": "Environment is not initialized."}

        normalized = normalize_text(response_text)
        if len(normalized.split()) < 6:
            return 0.0, {"reason": "Response must contain at least six words."}

        email = self._episode["email"]
        generic_checks = {
            "greeting": any(token in normalized for token in ["hello", "hi ", "good morning", "good afternoon"]),
            "acknowledgement": any(token in normalized for token in ["thank", "sorry", "appreciate"]),
            "next_step": any(
                token in normalized
                for token in [
                    "review",
                    "investig",
                    "coordinate",
                    "follow up",
                    "update",
                    "confirm",
                    "schedule",
                    "call",
                    "send",
                ]
            ),
            "closing": any(token in normalized for token in ["best", "regards", "sincerely"]),
        }
        generic_component = sum(1 for passed in generic_checks.values() if passed) / len(generic_checks)

        matched_groups = 0
        matched_keywords: List[List[str]] = []
        for keyword_group in email.response_requirements:
            if any(keyword in normalized for keyword in keyword_group):
                matched_groups += 1
                matched_keywords.append(keyword_group)
        content_component = matched_groups / len(email.response_requirements)

        quality = round(clamp((0.4 * generic_component) + (0.6 * content_component), 0.0, 1.0), 2)
        return quality, {
            "generic_checks": generic_checks,
            "matched_requirement_groups": matched_keywords,
            "content_component": round(content_component, 2),
            "generic_component": round(generic_component, 2),
            "quality": quality,
        }

    def _build_info(self, delta_reward: float, error: Optional[str], result: str) -> StepInfo:
        if self._episode is None:
            raise RuntimeError("Environment has not been reset. Call reset() before requesting info.")

        email = self._episode["email"]
        return StepInfo(
            task_name=self._episode["task"].name,
            current_stage=self._current_stage(),
            completed_stages=list(self._episode["completed_stages"]),
            reward_signal=RewardSignal(
                value=round(self._episode["raw_reward"], 2),
                delta=round(delta_reward, 2),
                score=round(self._episode["score"], 2),
            ),
            result=result,
            error=error,
            success=self._episode["success"],
            termination_reason=self._episode["termination_reason"],
            response_quality_details=self._episode["last_quality_details"],
            ground_truth={
                "classification": email.classification,
                "priority": email.priority,
                "response_style": email.response_style,
            }
            if self._episode["done"]
            else None,
        )


app = FastAPI(
    title="Email Triage and Smart Response System",
    version="1.0.0",
    description=(
        "OpenEnv-compatible environment for simulating email classification, prioritization, "
        "and response drafting."
    ),
)

ENV = EmailTriageEnv()


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(
        message="Email Triage and Smart Response System is running.",
        env_name=ENV.env_name,
        endpoints=["/", "/health", "/tasks", "/reset", "/step", "/state"],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", env_name=ENV.env_name)


@app.get("/tasks", response_model=TaskCatalog)
def list_tasks() -> TaskCatalog:
    return TaskCatalog(
        env_name=ENV.env_name,
        tasks={
            name: TaskSummary(
                objective=definition.objective,
                stages=definition.stages,
                max_steps=definition.max_steps,
                max_reward=definition.max_reward,
            )
            for name, definition in TASK_DEFINITIONS.items()
        },
    )


@app.get("/reset", response_model=ResetResponse)
def reset_env_get(task_name: str = "easy", email_id: Optional[str] = None) -> ResetResponse:
    try:
        observation = ENV.reset(task_name=task_name, email_id=email_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ResetResponse(observation=observation)


@app.post("/reset", response_model=ResetResponse)
def reset_env(request: ResetRequest) -> ResetResponse:
    try:
        observation = ENV.reset(task_name=request.task_name, email_id=request.email_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ResetResponse(observation=observation)


@app.post("/step", response_model=StepResponse)
def step_env(request: StepRequest) -> StepResponse:
    try:
        observation, reward, done, info = ENV.step(request.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", response_model=EpisodeState)
def state_env() -> EpisodeState:
    return ENV.state()
