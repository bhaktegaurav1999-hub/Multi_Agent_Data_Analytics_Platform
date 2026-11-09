
import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from supervisor_llm import classify_user_query

from data_analyst_agent import (
    sales_by_region,
    sales_by_customer_segment,
    sales_by_category,
    top_products,
    monthly_sales,
    get_dataset_summary,
)

from ml_analyst_agent import (
    predict_sales,
    get_feature_importance,
)

from visualization_agent import (
    plot_monthly_sales,
    plot_sales_by_region,
    plot_sales_by_category,
    plot_sales_by_customer_segment,
    plot_profit_by_region,
)


# --------------------------------------------------
# AGENT STATE
# --------------------------------------------------

class AgentState(TypedDict):
    user_query: str
    route: str
    result: object


# --------------------------------------------------
# SUPERVISOR AGENT
# --------------------------------------------------


def supervisor(state: AgentState):
    query = state["user_query"]

    try:
        route = classify_user_query(query)

    except Exception as e:
        print(f"LLM Supervisor Error: {e}")

        # Fallback routing if Ollama is unavailable
        lower_query = query.lower()

        if any(word in lower_query for word in [
            "predict", "prediction", "forecast",
            "machine learning", "feature importance",
        ]):
            route = "ml_agent"

        elif any(word in lower_query for word in [
            "chart", "graph", "plot", "visualize",
            "visualization", "trend",
        ]):
            route = "visualization_agent"

        else:
            route = "data_analyst_agent"

    return {"route": route}


# --------------------------------------------------
# DATA ANALYST AGENT
# --------------------------------------------------

def data_analyst_node(state: AgentState):
    query = state["user_query"].lower()

    if "customer segment" in query or "segment" in query:
        result = sales_by_customer_segment()

    elif "categor" in query:
        result = sales_by_category()

    elif "top product" in query:
        result = top_products()

    elif "monthly" in query or "month" in query:
        result = monthly_sales()

    elif "summary" in query or "overview" in query:
        result = get_dataset_summary()

    else:
        result = sales_by_region()

    if hasattr(result, "to_dict"):
        result = result.to_dict(orient="records")

    return {"result": result}


# --------------------------------------------------
# ML ANALYST AGENT
# --------------------------------------------------

def extract_order_details(query: str):
    query = query.lower().replace(",", "")

    quantity_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:units?|items?|pieces?)",
        query,
    )

    price_match = re.search(
        r"(?:₹|rs\.?\s*)(\d+(?:\.\d+)?)|"
        r"(\d+(?:\.\d+)?)\s*(?:rupees|inr)",
        query,
    )

    discount_match = re.search(
        r"(\d+(?:\.\d+)?)\s*%",
        query,
    )

    quantity = (
        float(quantity_match.group(1))
        if quantity_match
        else 5
    )

    if price_match:
        unit_price = float(
            price_match.group(1) or price_match.group(2)
        )
    else:
        unit_price = 1000

    if "no discount" in query or "without discount" in query:
        discount = 0.0

    elif discount_match:
        discount = float(discount_match.group(1)) / 100

    else:
        discount = 0.10

    if quantity <= 0 or unit_price <= 0:
        raise ValueError(
            "Quantity and unit price must be greater than zero."
        )

    if not 0 <= discount <= 1:
        raise ValueError(
            "Discount must be between 0% and 100%."
        )

    return quantity, unit_price, discount


def ml_agent_node(state: AgentState):
    query = state["user_query"].lower()

    if (
         "feature importance" in query
    or "important for prediction" in query
    or "features are most important" in query
    or "features influence" in query
    or "influencing sales predictions" in query
    or "important features" in query
    ):
        result = get_feature_importance()

        return {
            "result": result.to_dict(orient="records")
        }

    quantity, unit_price, discount = extract_order_details(query)

    prediction = predict_sales(
        quantity=quantity,
        unit_price=unit_price,
        discount=discount,
    )

    return {
        "result": {
            "quantity": quantity,
            "unit_price": unit_price,
            "discount": discount,
            "predicted_sales": float(prediction),
        }
    }


# --------------------------------------------------
# VISUALIZATION AGENT
# --------------------------------------------------

def visualization_node(state: AgentState):
    query = state["user_query"].lower()

    # Profit by region
    if "profit" in query and "region" in query:
        chart_path = plot_profit_by_region()
        chart_type = "profit_by_region"

    # Sales by customer segment
    elif "segment" in query:
        chart_path = plot_sales_by_customer_segment()
        chart_type = "sales_by_customer_segment"

    # Sales by product category
    elif "categor" in query:
        chart_path = plot_sales_by_category()
        chart_type = "sales_by_category"

    # Sales by region
    elif "region" in query:
        chart_path = plot_sales_by_region()
        chart_type = "sales_by_region"

    # Monthly sales trend (default)
    else:
        chart_path = plot_monthly_sales()
        chart_type = "monthly_sales"

    return {
        "result": {
            "chart_type": chart_type,
            "chart_path": chart_path,
        }
    }


# --------------------------------------------------
# ROUTING FUNCTION
# --------------------------------------------------

def route_to_agent(state: AgentState):
    return state["route"]


# --------------------------------------------------
# BUILD LANGGRAPH WORKFLOW
# --------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor)
builder.add_node("data_analyst", data_analyst_node)
builder.add_node("ml_agent", ml_agent_node)
builder.add_node("visualization", visualization_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "data_analyst_agent": "data_analyst",
        "ml_agent": "ml_agent",
        "visualization_agent": "visualization",
    },
)

builder.add_edge("data_analyst", END)
builder.add_edge("ml_agent", END)
builder.add_edge("visualization", END)

graph = builder.compile()


# --------------------------------------------------
# TEST MULTI-AGENT WORKFLOW
# --------------------------------------------------

if __name__ == "__main__":

    test_queries = [
        "Which region has the highest sales?",
        "Which customer segment generates the most profit?",
        "Predict sales for a new order",
        "Which features are most important for prediction?",
        "Show me a chart of monthly sales",
        "Show me a chart of sales by region",
        "Show me sales by product category chart",
        "Show me sales by customer segment chart",
        "Show me profit by region chart",
        "Predict sales for 8 units at ₹1,500 each with a 5% discount",
        "Predict sales for 3 units at ₹800 each with no discount",
    ]

    for query in test_queries:

        result = graph.invoke(
            {
                "user_query": query,
                "route": "",
                "result": "",
            }
        )

        print("\nQuery:", query)
        print("Route:", result["route"])
        print("Result:", result["result"])
