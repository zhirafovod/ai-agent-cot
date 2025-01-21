# CoT Agent Demonstration

This project demonstrates a Chain-of-Thought (CoT) approach to problem-solving using the [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) library and Langgraph Studio. The agent combines two models to iteratively analyze, generate, and validate solutions to computational problems, inspired by the Advent of Code challenges.

<!-- TOC -->
* [CoT Agent Demonstration](#cot-agent-demonstration)
  * [Overview](#overview)
    * [Key Features](#key-features)
    * [Limitations](#limitations)
  * [How It Works](#how-it-works)
    * [Chain of Thought Approaches](#chain-of-thought-approaches)
  * [Components](#components)
    * [Workflow Steps](#workflow-steps)
  * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Running the Studio](#running-the-studio)
<!-- TOC -->

## Overview

The CoT agent leverages:
- **Orchestrator Model**: A lightweight model (`llama-3.2-3b-instruct`) to analyze the problem, identify edge cases, and devise a solution plan.
- **Coder Model**: A larger coding-focused model (`qwen2.5-coder-14b-instruct`) to generate Python solutions based on the orchestrator's insights.

### Key Features
1. **Iterative Code Generation and Validation**:
   - The coder model generates a Python solution.
   - The solution is validated by executing the code locally.
   - If validation fails, error feedback is provided to refine the solution.
2. **Flexible Workflow**:
   - Each stage (problem fetching, analysis, code generation, validation) is modular and reusable.
   - A decision tree determines whether to retry code generation or finalize the solution.

### Limitations
- The CoT approach used here enriches the context with planning and edge cases but does not decompose problems into smaller subtasks.
- The orchestrator (`llama-3.2`) is used primarily for demonstration purposes. The coder model (`qwen2.5`) is sufficiently capable of handling tasks independently.

---

## How It Works

### Chain of Thought Approaches
1. **Regular Model Inference**:
   - Example: “Solve problem X.”
2. **Single-Shot Chain of Thought**:
   - Example: “Solve problem X. Think step by step.”
3. **Multi-Step Chain of Thought**:
   - Iteratively plan, execute, evaluate, and refine solutions.
   - Example Workflow:
     - "Create a plan to solve the problem."
     - "Execute the plan, use tools when needed."
     - "Evaluate the produced response and plan next steps."
   - Note: Determining when to stop is a complex challenge.

---

## Components


- **`cot_agent.py`**: Contains the core logic for the CoT agent.
- **Langgraph Studio**: Visualizes and manages the StateGraph workflow defined in `cot_agent.py`.

### Workflow Steps
1. **Fetch Problem Description**:
   - Fetches raw HTML of Advent of Code problem statements from the official website.
2. **Parse Problem**:
   - Extracts and cleans the problem description from HTML.
3. **Analyze Problem**:
   - Uses the orchestrator model to create a solution strategy and identify edge cases.
4. **Generate Code**:
   - The coder model generates Python code based on the problem description and strategy.
5. **Validate Code**:
   - Executes the generated code locally.
   - If validation fails, provides feedback to the coder model for refinement.
6. **Finalize Solution**:
   - Outputs the validated solution or an error message if retries are exhausted.

---

## Prerequisites

### Installation
1. **Install Required Tools**:
   - Python 3.11
   - Langgraph CLI: `pip install langgraph-cli[inmem]`
   - LM Studio for hosting models

2. **Load Models**:
   - **`llama-3.2`**: Orchestrator model
   - **`qwen2.5`**: Coder model

3. **Start OpenAI-Compatible API Server**:
   - Use LM Studio to serve the models on `localhost:1234`.

### Running the Studio
Launch Langgraph Studio to visualize and interact with the StateGraph:
```shell
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```