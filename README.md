---
title: Email Triage Smart Response System
sdk: docker
app_port: 8000
tags:
  - openenv
  - fastapi
  - reinforcement-learning
  - agents
---

# Email Triage and Smart Response System

## Problem Description

This repository implements a production-ready OpenEnv environment for a realistic AI operations task: triaging inbound email. The agent must inspect incoming messages, identify whether they are spam or legitimate business email, assign a business priority, and draft a professional response when appropriate.

The environment is intentionally grounded in workflows people actually perform in support, recruiting, finance, and account-management teams. It is designed for deterministic evaluation, lightweight inference, and straightforward deployment to Hugging Face Spaces.

## Why This Environment Is Useful

This benchmark covers a real gap between toy classification tasks and practical enterprise agent workflows:

- email classification requires intent recognition and threat awareness
- prioritization requires business context rather than simple spam detection
- response drafting requires tone, action planning, and time-sensitive communication

That makes it useful for both RL-style agent learning and baseline agent evaluation.

## Repository Structure

- `my_env.py`: environment logic plus FastAPI endpoints
- `models.py`: typed Pydantic models for observations, actions, rewards, and API responses
- `inference.py`: strict-format baseline runner using the OpenAI client
- `openenv.yaml`: OpenEnv environment metadata
- `Dockerfile`: HF Space and local container build definition
- `requirements.txt`: runtime dependencies
- `.dockerignore`: build cleanup

## OpenEnv API

The core environment class is `EmailTriageEnv` and exposes the standard API:

- `reset(task_name="easy", email_id=None) -> EmailObservation`
- `step(action) -> observation, reward, done, info`
- `state() -> EpisodeState`

The FastAPI service exposes:

- `GET /`
- `GET /health`
- `GET /tasks`
- `GET /reset`
- `POST /reset`
- `POST /step`
- `GET /state`

## Typed Models

The project defines typed Pydantic models in `models.py`, including:

- `EmailAction`
- `EmailObservation`
- `RewardSignal`
- `StepInfo`
- `EpisodeState`
- `ResetResponse`
- `StepResponse`

These models make the environment validator-friendly and keep request and response contracts explicit.

## Action Space

Actions are structured dictionaries validated by `EmailAction`.

### Classification

```python
{"action_type": "classify", "label": "spam"}
{"action_type": "classify", "label": "important"}
```

### Priority Assignment

```python
{"action_type": "prioritize", "priority": "low"}
{"action_type": "prioritize", "priority": "normal"}
{"action_type": "prioritize", "priority": "high"}
```

### Response Generation

```python
{
  "action_type": "respond",
  "response_text": "Hello Maya, I am sorry for the disruption. Our engineering team is investigating and I will send an update within the next hour."
}
```

## Observation Space

Observations are structured dictionaries backed by the `EmailObservation` model.

```python
{
  "env_name": "email-triage-smart-response-system",
  "task_name": "hard",
  "objective": "Classify the email, assign priority, and draft a smart response.",
  "stage": "respond",
  "step_count": 2,
  "remaining_steps": 3,
  "done": False,
  "email": {
    "id": "customer_api_outage",
    "sender": "maya.chen@brightledger.io",
    "sender_name": "Maya Chen",
    "subject": "Urgent: production API errors blocking our launch",
    "body": "...",
    "received_at": "2026-04-08T06:55:00Z",
    "summary": "High-priority customer incident with production impact and time pressure."
  },
  "allowed_actions": [...],
  "history": [...]
}
```

## Tasks

The benchmark contains three deterministic tasks with increasing difficulty.

### Easy

- Objective: classify the email as `spam` or `important`
- Stages: `classify`
- Max steps: `2`
- Score range: `0.0` to `1.0`

### Medium

- Objective: classify the email and assign a business priority
- Stages: `classify`, `prioritize`
- Max steps: `4`
- Score range: `0.0` to `1.0`

### Hard

- Objective: run the full triage pipeline, including response generation
- Stages: `classify`, `prioritize`, `respond`
- Max steps: `5`
- Score range: `0.0` to `1.0`

## Task Grading and Reward Logic

The environment uses deterministic programmatic grading with partial progress signals.

- correct classification contributes `+0.3`
- correct priority contributes `+0.3`
- response quality contributes up to `+0.4`
- invalid, incorrect, or out-of-order actions apply a `-0.2` penalty to the internal reward balance
- cumulative reward is clamped into `0.0` to `1.0`
- final task `score` is normalized strictly inside `(0, 1)` for validator compatibility

### Response Quality Rubric

Responses are graded on:

- greeting and professional tone
- acknowledgement of the request or issue
- relevant next-step language
- closing language
- task-specific coverage such as urgency, timeline, incident handling, or scheduling details

This provides non-binary reward shaping and deterministic final scores.

## Realistic Email Scenarios

The environment includes multiple realistic examples:

- phishing payroll verification scam
- executive gift-card fraud request
- interview reschedule request
- vendor invoice approval follow-up
- customer production API outage escalation
- enterprise renewal blocker tied to legal and security review

## Baseline Inference

`inference.py` runs all three tasks and emits strict stdout logs in the required format:

```text
[START] task=<task_name> env=<env_name> model=<model_name>
[STEP] step=<n> action=<action> reward=<0.00> done=<true/false> error=<msg/null>
[END] success=<true/false> steps=<n> score=<score> rewards=<r1,r2,...>
```

### Environment Variables

The baseline runner supports the required Round 1 variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

It also accepts `API_KEY` or `OPENAI_API_KEY` as fallbacks for local testing.

### Expected Baseline Scores

Using the built-in heuristic fallback with `--disable-openai`, the expected deterministic baseline is:

| Task | Score | Reward Trace |
|---|---:|---|
| `easy` | `0.99` | `0.30` |
| `medium` | `0.99` | `0.30,0.60` |
| `hard` | `0.99` | `0.30,0.60,1.00` |

## Local Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the baseline locally without external model calls

```bash
python inference.py --disable-openai
```

### 3. Run with an OpenAI-compatible endpoint

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hf_token"
python inference.py
```

### 4. Start the API locally

```bash
uvicorn my_env:app --host 0.0.0.0 --port 8000
```

## Docker

### Build

```bash
docker build -t email-triage-openenv .
```

### Run the API server

```bash
docker run --rm -p 8000:8000 email-triage-openenv
```

### Run the baseline script inside the image

```bash
docker run --rm email-triage-openenv python inference.py --disable-openai
```

## Hugging Face Space Deployment

This repository is configured for a Docker Space. The YAML frontmatter at the top of this README sets:

- `sdk: docker`
- `app_port: 8000`
- tag `openenv`

When you create the Space:

1. Choose `Docker` as the Space SDK.
2. Connect the Space to this repository.
3. Add the environment variables if you want model-backed inference:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
4. Wait for the Space to build.
5. Verify the Space root URL returns HTTP `200`.

## OpenEnv Metadata

`openenv.yaml` points to the FastAPI application:

```yaml
spec_version: 1
name: email-triage-smart-response-system
description: OpenEnv environment for triaging inbound emails, assigning priority, and generating smart responses.
runtime: fastapi
app: my_env:app
port: 8000
```

## Validation Checklist

Before submission, run:

```bash
pip install openenv-core
openenv validate
python inference.py --disable-openai
docker build -t email-triage-openenv .
docker run --rm -p 8000:8000 email-triage-openenv
```

Then verify:

- the API root returns `200`
- `/reset` returns an observation
- `/step` returns `observation`, `reward`, `done`, and `info`
- all three tasks complete with score in `0.0` to `1.0`
- the inference logs preserve exact `[START]`, `[STEP]`, `[END]` formatting

## Performance Notes

- CPU only
- no heavy local model dependencies
- deterministic fallback baseline
- suitable for `vcpu=2`, `memory=8gb`
- end-to-end runtime is expected to be well under 20 minutes
