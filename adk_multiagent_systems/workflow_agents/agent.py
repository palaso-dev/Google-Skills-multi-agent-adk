import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.models import Gemini
from google.adk.tools import exit_loop
from google.adk.tools.langchain_tool import LangchainTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from .callback_logging import log_query_to_model, log_model_response

load_dotenv()

model_name = os.getenv("MODEL", "gemini-3.1-flash-lite")
RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, attempts=6)

# Tools


def append_to_state(
    tool_context: ToolContext, field: str, response: str
) -> dict[str, str]:
    """Append new output to an existing state key.

    Args:
        field (str): a field name to append to
        response (str): a string to append to the field

    Returns:
        dict[str, str]: {"status": "success"}
    """
    existing_state = tool_context.state.get(field, [])
    tool_context.state[field] = existing_state + [response]
    logging.info(f"[Added to {field}] {response}")
    return {"status": "success"}


def write_file(
    tool_context: ToolContext,
    directory: str,
    filename: str,
    content: str,
) -> dict[str, str]:
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    if not safe_name.endswith(".txt"):
        safe_name = f"{safe_name}.txt"
    target_file = target_dir / safe_name
    target_file.write_text(content, encoding="utf-8")
    logging.info(f"[Wrote file] {target_file}")
    return {"status": "success", "path": str(target_file)}


# Agents

box_office_researcher = Agent(
    name="box_office_researcher",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Considers the box office potential of this film",
    instruction="""
    PLOT_OUTLINE:
    { PLOT_OUTLINE? }

    INSTRUCTIONS:
    Write a report on the box office potential of a movie like that described in PLOT_OUTLINE based on the reported box office performance of other recent films.
    """,
    output_key="box_office_report",
)

casting_agent = Agent(
    name="casting_agent",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Generates casting ideas for this film",
    instruction="""
    PLOT_OUTLINE:
    { PLOT_OUTLINE? }

    INSTRUCTIONS:
    Generate ideas for casting for the characters described in PLOT_OUTLINE by suggesting actors who have received positive feedback from critics and/or fans when they have played similar roles.
    """,
    output_key="casting_report",
)

preproduction_team = ParallelAgent(
    name="preproduction_team",
    sub_agents=[box_office_researcher, casting_agent],
)

critic = Agent(
    name="critic",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Reviews the outline so that it can be improved.",
    instruction="""
    INSTRUCTIONS:
    Consider these questions about the PLOT_OUTLINE:
    - Does it meet a satisfying three-act cinematic structure?
    - Do the characters' struggles seem engaging?
    - Does it feel grounded in a real time period in history?
    - Does it sufficiently incorporate historical details from the RESEARCH?

    If the PLOT_OUTLINE does a good job with these questions, exit the writing loop with your 'exit_loop' tool.
    If significant improvements can be made, use the 'append_to_state' tool to add your feedback to the field 'CRITICAL_FEEDBACK'.
    Explain your decision and briefly summarize the feedback you have provided.

    PLOT_OUTLINE:
    { PLOT_OUTLINE? }

    RESEARCH:
    { research? }
    """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    tools=[append_to_state, exit_loop],
)

file_writer = Agent(
    name="file_writer",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Creates marketing details and saves a pitch document.",
    instruction="""
    INSTRUCTIONS:
    - Create a marketable, contemporary movie title suggestion for the movie described in the PLOT_OUTLINE. If a title has been suggested in PLOT_OUTLINE, you can use it, or replace it with a better one.
    - Use your 'write_file' tool to create a new txt file with the following arguments:
        - for a filename, use the movie title
        - Write to the 'movie_pitches' directory.
        - For the 'content' to write, include:
            - The PLOT_OUTLINE
            - The BOX_OFFICE_REPORT
            - The CASTING_REPORT

    PLOT_OUTLINE:
    { PLOT_OUTLINE? }

    BOX_OFFICE_REPORT:
    { box_office_report? }

    CASTING_REPORT:
    { casting_report? }
    """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    tools=[write_file],
)

screenwriter = Agent(
    name="screenwriter",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="As a screenwriter, write a logline and plot outline for a biopic about a historical character.",
    instruction="""
    INSTRUCTIONS:
    Your goal is to write a logline and three-act plot outline for an inspiring movie about the historical character(s) described by the PROMPT using the provided RESEARCH.

    If there is CRITICAL_FEEDBACK, use those thoughts to improve upon the outline.
    If there is RESEARCH provided, feel free to use details from it, but you are not required to use it all.

    Use the 'append_to_state' tool to write your logline and three-act plot outline to the field 'PLOT_OUTLINE'.
    Summarize what you focused on in this pass.

    PROMPT:
    { PROMPT? }

    RESEARCH:
    { research? }

    CRITICAL_FEEDBACK:
    { CRITICAL_FEEDBACK? }
    """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    tools=[append_to_state],
)

researcher = Agent(
    name="researcher",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Answer research questions using Wikipedia.",
    instruction="""
    PROMPT:
    { PROMPT? }

    CRITICAL_FEEDBACK:
    { CRITICAL_FEEDBACK? }

    INSTRUCTIONS:
    - If there is a CRITICAL_FEEDBACK, use your wikipedia tool to do research to solve those suggestions.
    - If there is no CRITICAL_FEEDBACK, use your wikipedia tool to do research on the subject in PROMPT.
    - Use the 'append_to_state' tool to add your research to the field 'research'.
    - Summarize what you have learned.
    Now, use your wikipedia tool to do research.
    """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state,
    ],
)

writers_room = LoopAgent(
    name="writers_room",
    description="Iterates through research and writing to improve a movie plot outline.",
    sub_agents=[researcher, screenwriter, critic],
    max_iterations=5,
)

film_concept_team = SequentialAgent(
    name="film_concept_team",
    description="Write a film plot outline and save it as a text file.",
    sub_agents=[writers_room, preproduction_team, file_writer],
)

root_agent = Agent(
    name="greeter",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Guides the user in defining a movie pitch.",
    instruction="""
    - Let the user know you will help them write a pitch for a hit movie. Ask them for:
        - A historical figure to base the movie on, or
        - A general type of historical character they'd like to explore.
    - When they respond, use the 'append_to_state' tool to store the user's response
      in the 'PROMPT' state key and transfer to the 'film_concept_team' agent.
    """,
    tools=[append_to_state],
    sub_agents=[film_concept_team],
)
