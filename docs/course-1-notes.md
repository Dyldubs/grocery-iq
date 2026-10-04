# Course 1 notes: Develop Generative AI Applications

Short notes connecting Course 1 (LangChain basics, prompt templates, chains, tool
calling, model selection) to GroceryIQ's agent. A starter to expand as you work
through the lessons.

## Prompt templates
A prompt template is a parameterised string whose variables are filled at call
time. In GroceryIQ the agent's system prompt is a template holding the retail
role, tone, the four tool descriptions, and the hard rule that every answer cites
its source. Few-shot examples (question, chosen tool, cited answer) live here too.

## Chains
A chain composes steps: prompt, LLM, output parse. GroceryIQ's simplest chain is
question, retrieval, answer-with-citations. The agent is a chain with a decision
step (which tool to call) in the middle.

## Tool calling
Tools are typed functions the model can invoke: it picks the tool and arguments,
the runtime runs it, and the result returns to the context. GroceryIQ's tools map
onto the Milestone 1 models plus RAG:
- forecast_demand (XGBoost demand forecast)
- product_search (RAG over Open Food Facts)
- segment_customers (KMeans segmentation)
- price_elasticity (log-log OLS)
- sql_analytics (ad-hoc DuckDB queries)
Each tool needs a clear name, a one-line description the model reads to choose it,
and a typed input/output schema.

## Model selection and evaluation
Course 1 covers choosing and evaluating an LLM. GroceryIQ uses Gemini 1.5 Flash
(free tier) for cost; end-to-end answer quality is evaluated with Ragas later
(Milestone 8).

## Flask vs FastAPI
Course 1 builds the demo with Flask. GroceryIQ serves the /ask endpoint with
FastAPI (async, typed) instead. Same concept, different framework.

## How this maps forward
Milestone 3 turns these tools into a ReAct agent; Milestone 4 moves to a LangGraph
state machine with query routing and a citation self-check.
