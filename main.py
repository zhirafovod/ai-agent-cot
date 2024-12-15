import os

import openai
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

openai.api_base = "http://localhost:1234/v1"
openai.api_key = "dummy_key"  # Dummy key for compatibility

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
os.environ["OPENAI_API_BASE"] = LM_STUDIO_BASE_URL
os.environ["OPENAI_API_KEY"] = "dummy-key"  # Required by OpenAI clients but not used by LM Studio

# Define LLM
llm = OpenAI(temperature=0.5, model="qwen2.5-coder-14b-instruct")

# Step 1: Problem Breakdown Tool
problem_breakdown_prompt = PromptTemplate(
    input_variables=["problem"],
    template="Break down the following problem into steps: {problem}",
)
problem_breakdown_chain = LLMChain(llm=llm, prompt=problem_breakdown_prompt)

def problem_breakdown_tool(problem):
    return problem_breakdown_chain.run(problem)

# Step 2: Task Execution Tool
task_execution_prompt = PromptTemplate(
    input_variables=["task"],
    template="Solve the following task: {task}",
)
task_execution_chain = LLMChain(llm=llm, prompt=task_execution_prompt)

def task_execution_tool(task):
    return task_execution_chain.run(task)

# Step 3: Result Evaluation Tool
evaluation_prompt = PromptTemplate(
    input_variables=["results"],
    template="Evaluate the following results and check if they sufficiently solve the problem: {results}. If not, suggest next steps.",
)
evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)

def result_evaluation_tool(results):
    return evaluation_chain.run(results)

# Memory for Agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Helper to query human-in-the-loop
def ask_human_to_continue():
    while True:
        decision = input("Do you want to continue to the next iteration? (yes/no): ").strip().lower()
        if decision in ["yes", "no"]:
            return decision == "yes"
        print("Invalid input. Please enter 'yes' or 'no'.")

# Main Agent Logic
def solve_problem_with_human_in_loop(problem):
    current_problem = problem
    all_steps = []
    final_results = []

    while True:
        # Step 1: Break down the problem into tasks
        steps = problem_breakdown_tool(current_problem)
        print(f"Steps identified:\n{steps}")
        all_steps.append(steps)

        # Step 2: Execute each step
        step_results = []
        for step in steps.split("\n"):
            if step.strip():
                result = task_execution_tool(step)
                print(f"Result for step '{step}': {result}")
                step_results.append(result)

        # Step 3: Evaluate the results
        evaluation = result_evaluation_tool(step_results)
        print(f"Evaluation:\n{evaluation}")
        final_results.append(step_results)

        # Ask human whether to continue or stop
        if not ask_human_to_continue():
            print("Human decided to stop the process.")
            break

        # Check if the problem is solved
        if "problem solved" in evaluation.lower():
            print("Problem is solved based on evaluation.")
            break
        else:
            current_problem = evaluation  # Set next problem based on evaluation

    return final_results

# Example Usage
if __name__ == "__main__":
    problem = open("tests/input", 'r').read()
    solution = solve_problem_with_human_in_loop(problem)
    print("Final Solution:")
    print(solution)
