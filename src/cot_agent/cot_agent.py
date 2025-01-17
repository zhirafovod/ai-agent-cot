import re

import requests
import subprocess
from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from dataclasses import dataclass, field
from typing_extensions import TypedDict, Annotated

from bs4 import BeautifulSoup

# Note: Ollama is hard to troubleshoot because it doesn't return errors, only empty strings.
# orchestrator_llm = ChatOllama(model=Configuration.local_orchestrator_llm, temperature=0)

max_solution_attempts: int = 3   # Maximum times to regenerate code

orchestrator_llm: ChatOpenAI = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    model="llama-3.2-3b-instruct",
    temperature=0,
    api_key="not-needed"
)

coder_llm: ChatOpenAI = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    model="qwen2.5-coder-14b-instruct",
    temperature=0,
    api_key="not-needed"
)
# orchestrator_llm_json_mode = ChatOpenAI(model="llama-3.2-3b-instruct", temperature=0, format="json")

#### State

@dataclass(kw_only=True)
class AOCState:
    # New fields to track AoC puzzle info, solution, validation, etc.
    advent_of_code_day: int = field(default=1)  # The day from the user
    problem_description_html: str = field(default="")  # Raw HTML
    problem_description: str = field(default="")  #parsed text
    error: str = field(default="")  # error
    analysis_summary: str = field(default="")

    solution_code: str = field(default="")
    validation_output: str = field(default="")
    validation_success: bool = field(default=False)

    # Attempt counters to limit retries
    fetch_attempt_count: int = field(default=0)
    solution_attempt_count: int = field(default=0)


@dataclass(kw_only=True)
class AOCStateInput(TypedDict):
    advent_of_code_day: int


@dataclass(kw_only=True)
class AOCStateOutput(TypedDict):
    final_summary: str




# fetch the problem description
def fetch_aoc_problem_description(state: AOCState):
    """
    Fetch 2024 Advent of Code puzzle text from the official website for a given day.
    """
    day = state.advent_of_code_day
    url = f"https://adventofcode.com/2024/day/{day}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        puzzle_html = response.text

        # For demonstration, we just store raw HTML.
        # In a real scenario, you'd parse the HTML to extract the puzzle statement.
        return {
            "problem_description": puzzle_html,
            "fetch_attempt_count": state.fetch_attempt_count + 1
        }
    except Exception as e:
        print(f"Error fetching AoC problem: {e}")
        # You could retry here or store error in state
        # For simplicity, just store empty string if fetch fails
        return {
            "error": state(e),
            "fetch_attempt_count": state.fetch_attempt_count + 1
        }

# parse description from HTML
def parse_problem(state: AOCState):
    """
    parse problem description from HTML
    """
    print(state)

    try:
        # result = orchestrator_llm.invoke([SystemMessage(content=parse_prompt)])
        soup = BeautifulSoup(state.problem_description, "html.parser")

        # Extract the main puzzle description (within <article> tags)
        puzzle_parts = soup.find_all("article", class_="day-desc")
        cleaned_text = ""
        for part in puzzle_parts:
            cleaned_text += part.get_text(separator="\n").strip() + "\n\n"

        # Return the cleaned description
        return {"problem_description": cleaned_text.strip()}
    except Exception as e:
        print("Error parsing AoC problem: {e}")
        return {"error": str(e)}


# ANALYZE THE PROBLEM
def analyze_problem(state: AOCState):
    """
    Have the LLM look at the problem description and generate a short plan or reflection
    about how to approach the solution.
    """
    if not state.problem_description:
        return {"analysis_summary": "No problem description available."}

    analysis_prompt = f"""
    Below is the Advent of Code problem description:
    {state.problem_description}
    Think through solution strategy, edge cases and how to solve. NO CODE NEEDED"""

    result = orchestrator_llm.invoke([SystemMessage(content=analysis_prompt)])

    return {"analysis_summary": result.content}


############################################
# 3) GENERATE CODE
############################################
def generate_code_solution(state: AOCState):
    """
    Use the LLM to generate a Python solution
    """
    code_solver_instructions = f"""
        RETURN ONLY CODE, NO QUOTES OR EXPLANATION NEEDED.
        Problem description: 
        {state.problem_description}
        Hints to solve the problem:
        {state.analysis_summary}
        Write a python code to solve the problem provided by the user. 
        """
    if state.validation_output: # means we tryed to validate and it failed
        code_solver_instructions = f"""
            RETURN ONLY CODE, NO QUOTES OR EXPLANATION NEEDED.
            Problem description: 
            {state.problem_description}
            Hints to solve the problem:
            {state.analysis_summary}
            Correct the program below:
            {state.solution_code}
            Last error trying to run the code:
            {state.validation_output} 
            """

    result = coder_llm.invoke(
        [SystemMessage(content=code_solver_instructions)]
    )

    solution_code = result.content.strip()

    pattern = r"```(?:python)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, result.content)

    if match:
        # Extract the code inside the backticks
        solution_code = match.group(1).strip()

    return {"solution_code": solution_code, "solution_attempt_count": state.solution_attempt_count + 1}


# validate the code by running it :)
def validate_code_solution(state: AOCState):
    """
    Validate the generated code by either:
      - Running it locally and checking the return code
      - Or using an external "tool" / sandbox environment
    Here we attempt a local run in a subprocess with Python.

    If the code fails, we'll set success=False, so we can conditionally
    re-generate or finalize after a certain number of attempts.
    """
    solution_code = state.solution_code

    if not solution_code:
        return {"validation_success": False}

    # Write code to a temp file
    with open("aoc_solution.py", "w", encoding="utf-8") as f:
        f.write(solution_code)

    # Attempt to run it in a subprocess
    try:
        process = subprocess.run(
            ["python", "aoc_solution.py"], capture_output=True, text=True, timeout=10
        )
        if process.returncode == 0:
            # Assume success if code ran with zero return code
            return {"validation_success": True, "validation_output": process.stdout}
        else:
            # Non-zero exit code means failure
            print("Error output:\n", process.stderr)
            return {"validation_success": False, "validation_output": process.stderr}
    except Exception as e:
        print("Validation error:\n", e)
        return {"validation_success": False, "validation_output": str(e)}


############################################
# 5) ROUTE BASED ON VALIDATION
############################################
def route_solution(state: AOCState) -> Literal["generate_code_solution", "finalize_solution"]:
    """
    If validation fails and we haven't exceeded the maximum attempts,
    we return to code generation. Otherwise, finalize.
    """
    if not state.validation_success and state.solution_attempt_count < max_solution_attempts:
        return "generate_code_solution"
    else:
        return "finalize_solution"


############################################
# 6) FINALIZE
############################################
def finalize_solution(state: AOCState):
    """
    Prepare the final output: the validated code, or an error if no success.
    """
    if state.validation_success:
        final = "## Final Advent of Code Solution\n\n"
        final += f"**Day**: {state.advent_of_code_day}\n\n"
        final += "### Code:\n\n"
        final += f"```python\n{state.solution_code}\n```\n\n"
        final += "### Validation Output:\n\n"
        final += f"```\n{state.validation_output}\n```\n"
        return {"final_summary": final}
    else:
        return {"final_summary": "Validation failed. Exceeded the maximum attempt limit."}


############################################
# BUILD THE GRAPH
############################################
builder = StateGraph(AOCState, input=AOCStateInput, output=AOCStateOutput)

builder.add_node("fetch_aoc_problem_description", fetch_aoc_problem_description)
builder.add_node("parse_problem", parse_problem)
builder.add_node("analyze_problem", analyze_problem)
builder.add_node("generate_code_solution", generate_code_solution)
builder.add_node("validate_code_solution", validate_code_solution)
builder.add_node("finalize_solution", finalize_solution)

# Flow:
# START -> fetch problem -> analyze -> generate code -> validate code -> route -> ...
builder.add_edge(START, "fetch_aoc_problem_description")
builder.add_edge("fetch_aoc_problem_description", "parse_problem")
builder.add_edge("parse_problem", "analyze_problem")
builder.add_edge("analyze_problem", "generate_code_solution")
builder.add_edge("generate_code_solution", "validate_code_solution")

# route either back to generate code, or finalize
builder.add_conditional_edges("validate_code_solution", route_solution)

builder.add_edge("finalize_solution", END)

graph = builder.compile()
