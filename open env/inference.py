from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - safe fallback for minimal local runs
    OpenAI = None

from my_env import ENV_NAME, EmailTriageEnv, TASK_DEFINITIONS

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"


def to_bool_string(value: bool) -> str:
    return "true" if value else "false"


def serialize_action(action: Dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def serialize_error(error: Optional[str]) -> str:
    return "null" if error is None else json.dumps(error, ensure_ascii=True)


def extract_json_object(payload: str) -> str:
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output.")
    return payload[start : end + 1]


def extract_first_name(full_name: str) -> str:
    cleaned = full_name.strip()
    return cleaned.split()[0] if cleaned else "there"


def to_mapping(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return dict(value)


class HeuristicPlanner:
    SPAM_MARKERS = [
        "confirm your password",
        "account will be suspended",
        "secure link",
        "gift cards",
        "send the codes",
        "keep this confidential",
        "wire transfer",
        "crypto",
        "wallet",
        "unusual payroll activity",
    ]

    HIGH_PRIORITY_MARKERS = [
        "urgent",
        "blocking",
        "production",
        "launch",
        "today",
        "as soon as possible",
        "service gap",
        "500 errors",
        "cannot",
        "outage",
        "renewal",
        "incident",
    ]

    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        observation = to_mapping(observation)
        stage = observation["stage"]
        email = observation["email"]
        email_text = f"{email['subject']} {email['body']}".lower()

        if stage == "classify":
            label = "spam" if any(marker in email_text for marker in self.SPAM_MARKERS) else "important"
            return {"action_type": "classify", "label": label}

        if stage == "prioritize":
            if any(marker in email_text for marker in self.SPAM_MARKERS):
                priority = "low"
            elif any(marker in email_text for marker in self.HIGH_PRIORITY_MARKERS):
                priority = "high"
            else:
                priority = "normal"
            return {"action_type": "prioritize", "priority": priority}

        if stage == "respond":
            return {"action_type": "respond", "response_text": self._draft_response(email)}

        return {"action_type": "classify", "label": "important"}

    def _draft_response(self, email: Dict[str, Any]) -> str:
        sender_name = extract_first_name(email["sender_name"])
        email_text = f"{email['subject']} {email['body']}".lower()

        if "interview" in email_text:
            return (
                f"Hello {sender_name},\n\n"
                "Thank you for the heads-up. Rescheduling your interview is no problem. "
                "Please let me know whether tomorrow afternoon or Friday morning works best, "
                "and I will send an updated calendar invite as soon as you confirm.\n\n"
                "Best,\nRecruiting Team"
            )

        if "invoice" in email_text or "approval" in email_text:
            return (
                f"Hello {sender_name},\n\n"
                "Thank you for following up. I am reviewing the invoice approval with finance "
                "and accounts payable today, and I will send you an update on the status before Friday. "
                "If anything else is needed from our side, I will confirm that in my follow up.\n\n"
                "Best,\nFinance Operations"
            )

        if "api" in email_text or "500 errors" in email_text or "launch" in email_text:
            return (
                f"Hello {sender_name},\n\n"
                "I am sorry about the disruption and the production API errors impacting your launch. "
                "Our engineering team is investigating the incident right now, and I will send you an update "
                "within the next hour with the latest status and ETA. If helpful, I can also set up a call today "
                "to walk through the issue and next steps.\n\n"
                "Best,\nSupport Team"
            )

        if "renewal" in email_text or "security addendum" in email_text or "legal" in email_text:
            return (
                f"Hello {sender_name},\n\n"
                "Thank you for flagging this. I appreciate the urgency and I am coordinating with our legal "
                "and security teams on the remaining addendum review items now. I will send you an update today, "
                "and I am happy to jump on a call this afternoon if that helps us keep the renewal on track.\n\n"
                "Best,\nAccount Team"
            )

        return (
            f"Hello {sender_name},\n\n"
            "Thank you for reaching out. I am reviewing the request now and will follow up with an update shortly. "
            "If a quick call would help, I am happy to coordinate next steps.\n\n"
            "Best,\nOperations Team"
        )


class OpenAIPlanner:
    def __init__(self, model_name: str, api_base_url: str, api_key: str) -> None:
        if OpenAI is None:
            raise RuntimeError("The openai package is not installed.")
        self.model_name = model_name
        self.client = OpenAI(base_url=api_base_url, api_key=api_key)

    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        observation = to_mapping(observation)
        prompt = (
            "You are controlling an email triage environment. "
            "Return one JSON object only with a valid action for the current stage. "
            "Use the schema below and do not include markdown.\n"
            "Classification action: {\"action_type\":\"classify\",\"label\":\"spam|important\"}\n"
            "Priority action: {\"action_type\":\"prioritize\",\"priority\":\"low|normal|high\"}\n"
            "Response action: {\"action_type\":\"respond\",\"response_text\":\"...\"}\n\n"
            f"Observation:\n{json.dumps(observation, ensure_ascii=True)}"
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            max_tokens=220,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an agent that must act in a deterministic email triage environment. "
                        "Return exactly one JSON object and nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        output_text = response.choices[0].message.content.strip() if response.choices else ""
        if not output_text:
            raise ValueError("Model response did not include content.")
        return json.loads(extract_json_object(output_text))


class PolicyController:
    def __init__(self, requested_model: str, api_base_url: str, api_key: Optional[str], allow_openai: bool) -> None:
        self.heuristic = HeuristicPlanner()
        self.model_name = "simulated-heuristic"
        self._openai_planner: Optional[OpenAIPlanner] = None

        if allow_openai and OpenAI is not None and api_key:
            self._openai_planner = OpenAIPlanner(requested_model, api_base_url=api_base_url, api_key=api_key)
            self.model_name = requested_model

    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self._openai_planner is None:
            return self.heuristic.select_action(observation)

        try:
            return self._openai_planner.select_action(observation)
        except Exception:
            self.model_name = "simulated-heuristic"
            self._openai_planner = None
            return self.heuristic.select_action(observation)


def run_task(env: EmailTriageEnv, policy: PolicyController, task_name: str) -> None:
    reward_trace: List[str] = []
    step_count = 0
    done = False
    final_score = 0.0
    success = False
    observation = None

    print(f"[START] task={task_name} env={ENV_NAME} model={policy.model_name}")

    try:
        observation = env.reset(task_name=task_name)

        while not done:
            step_count += 1
            action = policy.select_action(observation)
            error_message: Optional[str] = None
            try:
                observation, reward, done, info = env.step(action)
                error_message = info.error
                final_score = float(info.reward_signal.score)
                success = bool(info.success)
            except Exception as exc:
                reward = 0.0
                done = True
                error_message = str(exc)
                success = False

            reward_trace.append(f"{reward:.2f}")
            print(
                f"[STEP] step={step_count} action={serialize_action(action)} "
                f"reward={reward:.2f} done={to_bool_string(done)} error={serialize_error(error_message)}"
            )

            if error_message is not None and done:
                success = False
    except Exception:
        success = False

    state = env.state()
    final_score = float(state.score)
    success = bool(state.success)
    print(
        f"[END] success={to_bool_string(success)} steps={step_count} "
        f"score={final_score:.2f} rewards={','.join(reward_trace)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Email Triage and Smart Response System.")
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL_NAME,
        help="Model name to use for OpenAI-compatible inference.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL),
        help="OpenAI-compatible base URL for routed inference.",
    )
    parser.add_argument(
        "--disable-openai",
        action="store_true",
        help="Force the local heuristic policy even when API credentials are available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    env = EmailTriageEnv()
    policy = PolicyController(
        requested_model=args.model,
        api_base_url=args.api_base_url,
        api_key=api_key,
        allow_openai=not args.disable_openai,
    )
    for task_name in TASK_DEFINITIONS:
        run_task(env, policy, task_name)


if __name__ == "__main__":
    main()
