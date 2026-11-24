# Google ADK Multi-Agent Systems (GENAI106)

Local implementation of the **Google Agent Development Kit (ADK)** multi-agent orchestration lab (**GENAI106**).

## Project Structure

```text
adk_multiagent_systems/
├── requirements.txt
├── parent_and_subagents/
│   ├── __init__.py
│   ├── .env
│   └── agent.py          # Hierarchical agents, peer transfers, and session state
└── workflow_agents/
    ├── __init__.py
    ├── .env
    ├── callback_logging.py
    ├── agent.py          # SequentialAgent, LoopAgent, and ParallelAgent pipeline
    └── movie_pitches/    # Generated pitch documents (.txt)
```

## Prerequisites

- Python 3.12 (managed via `uv`)
- A Gemini API key from Google AI Studio (or a GCP project with Vertex AI enabled)

## Local Setup

1. Create and activate the virtual environment, then install dependencies:

   ```zsh
   uv venv --python 3.12 .venv
   source .venv/bin/activate
   uv pip install "google-adk==1.30.0" -r adk_multiagent_systems/requirements.txt
   ```

2. Configure `.env` in both `adk_multiagent_systems/parent_and_subagents/.env` and `adk_multiagent_systems/workflow_agents/.env`:

   - **Using Gemini API Key (recommended for local):**
     ```env
     GOOGLE_GENAI_USE_VERTEXAI=FALSE
     GOOGLE_API_KEY=your_gemini_api_key_here
     MODEL=gemini-3.1-flash-lite
     ```
   - **Using Vertex AI:**
     ```env
     GOOGLE_GENAI_USE_VERTEXAI=TRUE
     GOOGLE_CLOUD_PROJECT=your_gcp_project_id
     GOOGLE_CLOUD_LOCATION=global
     MODEL=gemini-3.1-flash-lite
     ```
     *(Requires `gcloud auth application-default login`.)*

## Running the Agents

Activate the environment and move into `adk_multiagent_systems`:

```zsh
source .venv/bin/activate
cd adk_multiagent_systems
```

### 1. Parent, Sub-Agents & Session State (`parent_and_subagents`)

- **CLI:**
  ```zsh
  adk run parent_and_subagents
  ```
- **Web UI:**
  ```zsh
  adk web --reload_agents
  ```
  Open `http://127.0.0.1:8000` and select `parent_and_subagents`.

### 2. Workflow Agents (`workflow_agents`)

Runs a movie-pitch pipeline using:
- `greeter` (`Agent`) → collects the historical figure or concept into `PROMPT` state.
- `film_concept_team` (`SequentialAgent`):
  1. `writers_room` (`LoopAgent`, `max_iterations=5`): `researcher` (Wikipedia tool) → `screenwriter` → `critic` (`exit_loop`).
  2. `preproduction_team` (`ParallelAgent`): `box_office_researcher` + `casting_agent`.
  3. `file_writer` (`Agent`): compiles the outline, box office report, and casting report into `movie_pitches/<title>.txt`.

Start the Web UI:

```zsh
adk web --reload_agents
```

Open `http://127.0.0.1:8000`, select `workflow_agents`, and start with `hello`.
