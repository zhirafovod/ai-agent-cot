# CoT agent demonstration
This code demonstrates a usage of a Chain-of-Through (CoT) approach using [langchain-ai/langgraph] library and langgraph studio.

This project demonstrates usage of an orchestrator model (llama3.2, just for the sake of demonstration), which is prompted to analyze the input and provis a additional context, like edge cases and plan to solve the problem. This extended context is passed to the coder model (qwen2.5), which is prompted to write a pyton code for the given problem and solution analysis and plan. 

Finally, the code is attempted to run (don't repeat it at home, it's dangerous). If fails, the coder model is provided with the failed solution code, error context and prompted to fix the code. 

Note:
* The usage of llama3.2 and qwen2.5 is just for demonstration. Qwen2.5 is a way bigger model and is capable of solving the problem without the need of llama3.2.
* The CoT iterative solution is naivy, it does not break down the problem to smaller tasks, and just enriches the context with the plan and edge case generated in the previous step.

# Chain of Thought
## Regular model inference:
“Solve problem …”

## Single Shot Chain of Thought (smart prompting)
“Solve problem …. . Think step by step”

## Multi-shot Chain of thought - continues steps
“Create a plan to solve the problem” 
“Execute the plan, use tools when needed”
“Evaluate produced response, plan next steps…”
(btw, the hardest problem is to understand where to stop). 

## Skipping advanced decision tree approaches
...

# Prerequisites
TODO: add prerequisites and how to install it
LM Studio, load the models: 
llama3.2 model
qwen2.5 coder model

Starting an openai-compatible API server to serve the models

# Running the studio
TODO: add explanation of what is uvx and how to set it up
```shell
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

