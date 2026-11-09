
import ollama

MODEL_NAME = "llama3.2:3b"

ALLOWED_AGENTS = [
    "data_analyst_agent",
    "ml_agent",
    "visualization_agent",
]


def classify_user_query(user_query):

    query = user_query.lower()

    # 1. High-confidence visualization requests
    visualization_keywords = [
        "chart",
        "graph",
        "plot",
        "visualize",
        "visualization",
        "draw a graph",
        "show a graph",
    ]

    if any(word in query for word in visualization_keywords):
        return "visualization_agent"

    # 2. High-confidence machine learning requests
    ml_keywords = [
        "predict",
        "prediction",
        "forecast",
        "estimate sales",
        "estimate revenue",
        "machine learning",
        "feature importance",
        "influence sales predictions",
    ]

    if any(word in query for word in ml_keywords):
        return "ml_agent"

    # 3. Use LLM for other or ambiguous requests
    prompt = f"""
You are an intelligent supervisor for a business
analytics multi-agent platform.

Select exactly one agent.

Available agents:

data_analyst_agent:
For business summaries, sales analysis,
customer segments, regions, categories,
products and business metrics.

ml_agent:
For machine learning, sales prediction,
forecasting and feature importance.

visualization_agent:
For charts, graphs, plots and visual trends.

Examples:

Question: Which region generates the most sales?
Answer: data_analyst_agent

Question: Which customer segment performs best?
Answer: data_analyst_agent

Question: Predict revenue for 10 units.
Answer: ml_agent

Question: Estimate future sales.
Answer: ml_agent

Question: Draw a graph of regional sales.
Answer: visualization_agent

Question: Give me an overall business overview.
Answer: data_analyst_agent

User question:
{user_query}

Return ONLY one agent name from the available agents.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={"temperature": 0},
    )

    raw_response = response["message"]["content"].strip().lower()

    for agent in ALLOWED_AGENTS:
        if agent in raw_response:
            return agent

    return "data_analyst_agent"


if __name__ == "__main__":

    test_queries = [
        "Give me an overview of business performance",
        "Which customer segment is performing best?",
        "Estimate sales for 10 units at 500 rupees",
        "Show me a graph of regional sales",
        "Which factors influence sales predictions?",
    ]

    for query in test_queries:

        agent = classify_user_query(query)

        print("\nUser Query:", query)
        print("Selected Agent:", agent)
