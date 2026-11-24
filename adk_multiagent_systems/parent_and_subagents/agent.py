import os
import logging
from typing import List

from dotenv import load_dotenv
from google.adk import Agent
from google.adk.models import Gemini
from google.adk.tools.tool_context import ToolContext
from google.genai import types

load_dotenv()

model_name = os.getenv("MODEL", "gemini-3.1-flash-lite")
RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, attempts=6)

# Tools


def save_attractions_to_state(
    tool_context: ToolContext,
    attractions: List[str],
) -> dict[str, str]:
    """Saves the list of attractions to state["attractions"].

    Args:
        attractions (List[str]): a list of strings to add to the list of attractions

    Returns:
        dict[str, str]: {"status": "success"}
    """
    existing_attractions = tool_context.state.get("attractions", [])
    tool_context.state["attractions"] = existing_attractions + attractions
    return {"status": "success"}


# Agents

travel_brainstormer = Agent(
    name="travel_brainstormer",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Helps a user decide what country to visit.",
    instruction="""
    Provide a few suggestions of popular countries for travelers.

    Help a user identify their primary goals of travel:
    adventure, leisure, learning, shopping, or viewing art.

    Identify countries that would make great destinations
    based on their priorities.
    """,
)

attractions_planner = Agent(
    name="attractions_planner",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Build a list of attractions to visit in a country.",
    instruction="""
    Provide the user options for attractions to visit within their selected country.
    - When they reply, use your tool to save their selected attraction and then provide more possible attractions.
    - If they ask to view the list, provide a bulleted list of { attractions? } and then suggest some more.
    """,
    tools=[save_attractions_to_state],
)

root_agent = Agent(
    name="steering",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Start a user on a travel adventure.",
    instruction="""
    Ask the user if they know where they'd like to travel
    or if they need some help deciding.
    If they need help deciding, send them to 'travel_brainstormer'.
    If they know what country they'd like to visit, send them to the 'attractions_planner'.
    """,
    sub_agents=[travel_brainstormer, attractions_planner],
)
