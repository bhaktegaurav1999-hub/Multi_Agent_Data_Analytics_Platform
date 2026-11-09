
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from llm_agent import generate_business_insight

from data_analyst_agent import (
    get_dataset_summary,
    sales_by_region,
    sales_by_customer_segment,
)

from ml_analyst_agent import (
    get_feature_importance,
    train_sales_model,
    predict_sales,
)

from visualization_agent import (
    plot_sales_by_region,
    plot_sales_by_category,
    plot_profit_by_region,
)


# --------------------------------------------------
# SHARED STATE FOR MULTI-AGENT COLLABORATION
# --------------------------------------------------

class CollaborationState(TypedDict):

    user_query: str
    data_analysis: Any
    ml_analysis: Any
    visualization_results: Any
    final_report: str


# --------------------------------------------------
# DATA ANALYST COLLABORATION NODE
# --------------------------------------------------

def data_analyst_collaboration_node(state: CollaborationState):

    print("\nData Analyst Agent: Analyzing business performance...")

    summary = get_dataset_summary()

    region_analysis = sales_by_region()

    segment_analysis = sales_by_customer_segment()

    # Convert DataFrames into dictionaries
    if hasattr(region_analysis, "to_dict"):
        region_analysis = region_analysis.to_dict(
            orient="records"
        )

    if hasattr(segment_analysis, "to_dict"):
        segment_analysis = segment_analysis.to_dict(
            orient="records"
        )

    analysis_results = {
        "overall_summary": summary,
        "sales_by_region": region_analysis,
        "sales_by_customer_segment": segment_analysis,
    }

    print("Data Analyst Agent: Analysis completed.")

    return {
        "data_analysis": analysis_results
    }


# --------------------------------------------------
# ML ANALYST COLLABORATION NODE
# --------------------------------------------------

def ml_analyst_collaboration_node(state: CollaborationState):

    print("\nML Analyst Agent: Analyzing machine learning results...")

    feature_importance = get_feature_importance()

    if hasattr(feature_importance, "to_dict"):
        feature_importance = feature_importance.to_dict(
            orient="records"
        )

    model, mae, r2 = train_sales_model()

    sample_prediction = predict_sales(
        quantity=5,
        unit_price=1000,
        discount=0.10,
    )

    ml_results = {
        "feature_importance": feature_importance,

        "model_performance": {
            "mae": float(mae),
            "r2_score": float(r2),
        },

        "sample_prediction": {
            "quantity": 5,
            "unit_price": 1000,
            "discount": 0.10,
            "predicted_sales": float(sample_prediction),
        },
    }

    print("ML Analyst Agent: Analysis completed.")

    return {
        "ml_analysis": ml_results
    }


# --------------------------------------------------
# VISUALIZATION COLLABORATION NODE
# --------------------------------------------------

def visualization_collaboration_node(state: CollaborationState):

    print("\nVisualization Agent: Generating business charts...")

    regional_chart = plot_sales_by_region()

    category_chart = plot_sales_by_category()

    profit_chart = plot_profit_by_region()

    visualization_results = {
        "sales_by_region": regional_chart,
        "sales_by_category": category_chart,
        "profit_by_region": profit_chart,
    }

    print("Visualization Agent: Charts generated successfully.")

    return {
        "visualization_results": visualization_results
    }


# --------------------------------------------------
# BUSINESS INSIGHT VALIDATION
# --------------------------------------------------

def validate_business_insights(data, ml):

    region_data = data["sales_by_region"]

    segment_data = data["sales_by_customer_segment"]

    feature_data = ml["feature_importance"]

    # Highest and lowest sales regions
    highest_sales_region = max(
        region_data,
        key=lambda x: x["total_sales"]
    )

    lowest_sales_region = min(
        region_data,
        key=lambda x: x["total_sales"]
    )

    # Highest and lowest profit regions
    highest_profit_region = max(
        region_data,
        key=lambda x: x["total_profit"]
    )

    lowest_profit_region = min(
        region_data,
        key=lambda x: x["total_profit"]
    )

    # Best and lowest customer segments by sales
    best_segment = max(
        segment_data,
        key=lambda x: x["total_sales"]
    )

    lowest_segment = min(
        segment_data,
        key=lambda x: x["total_sales"]
    )

    # Most important ML feature
    most_important_feature = max(
        feature_data,
        key=lambda x: x["importance"]
    )

    return {
        "highest_sales_region": highest_sales_region,
        "lowest_sales_region": lowest_sales_region,
        "highest_profit_region": highest_profit_region,
        "lowest_profit_region": lowest_profit_region,
        "best_customer_segment": best_segment,
        "lowest_customer_segment": lowest_segment,
        "most_important_feature": most_important_feature,
    }


# --------------------------------------------------
# EVIDENCE-BASED BUSINESS RECOMMENDATIONS
# --------------------------------------------------

def generate_verified_recommendations(verified_insights):

    lowest_sales_region = verified_insights[
        "lowest_sales_region"
    ]["region"]

    lowest_profit_region = verified_insights[
        "lowest_profit_region"
    ]["region"]

    lowest_segment = verified_insights[
        "lowest_customer_segment"
    ]["customer_segment"]

    recommendations = [
        {
            "observation": (
                f"{lowest_sales_region} has the lowest "
                "total regional sales."
            ),

            "action": (
                f"Review product-category and monthly sales "
                f"patterns in {lowest_sales_region} to identify "
                "potential improvement opportunities."
            ),
        },

        {
            "observation": (
                f"{lowest_profit_region} has the lowest "
                "total regional profit."
            ),

            "action": (
                f"Compare profit margins and discount patterns "
                f"across regions, particularly "
                f"{lowest_profit_region}, before deciding "
                "whether pricing changes are needed."
            ),
        },

        {
            "observation": (
                f"{lowest_segment} has the lowest total "
                "sales among customer segments."
            ),

            "action": (
                f"Investigate order volume and average order "
                f"value for the {lowest_segment} segment "
                "before considering targeted campaigns."
            ),
        },
    ]

    return recommendations


# --------------------------------------------------
# FINAL LLM BUSINESS INTELLIGENCE NODE
# --------------------------------------------------

def final_report_node(state: CollaborationState):

    print("\nLLM Agent: Preparing combined business report...")

    data = state["data_analysis"]

    ml = state["ml_analysis"]

    summary = data["overall_summary"]

    # 1. Verified financial metrics
    verified_metrics = {
        "total_orders": int(summary["total_orders"]),
        "total_customers": int(summary["total_customers"]),
        "total_sales": float(summary["total_sales"]),
        "total_profit": float(summary["total_profit"]),
        "average_order_value": float(summary["average_order_value"]),
    }

    # 2. Validate business insights using Python
    verified_insights = validate_business_insights(data, ml)

    # 3. Generate evidence-based recommendations using Python
    verified_recommendations = generate_verified_recommendations(
        verified_insights
    )

    # 4. Prepare verified findings for LLM
    ai_context = {
        "highest_sales_region": {
            "name": verified_insights["highest_sales_region"]["region"]
        },

        "lowest_sales_region": {
            "name": verified_insights["lowest_sales_region"]["region"]
        },

        "highest_profit_region": {
            "name": verified_insights["highest_profit_region"]["region"]
        },

        "lowest_profit_region": {
            "name": verified_insights["lowest_profit_region"]["region"]
        },

        "best_customer_segment": {
            "name": verified_insights["best_customer_segment"]["customer_segment"]
        },

        "lowest_customer_segment": {
            "name": verified_insights["lowest_customer_segment"]["customer_segment"]
        },

        "most_important_feature": {
            "name": verified_insights["most_important_feature"]["feature"]
        },
    }

    # 5. Ask LLM to explain verified findings
    ai_insights = generate_business_insight(
        user_query=(
            "Act as a professional business intelligence consultant. "
            "Explain the supplied Python-validated findings. "
            "Do not change any rankings. "
            "Do not make causal claims about predictive features. "
            "Do not assume customer loyalty, satisfaction or behavior. "
            "Do not invent reasons for regional or segment performance. "
            "Do not introduce unsupported figures or facts. "
            "Keep the explanation concise and professional."
        ),

        analysis_result=ai_context,
    )

    # 6. Format Python-generated recommendations
    recommendation_text = "\n\n".join(
        (
            f"{i}. Observation: {item['observation']}\n"
            f"   Recommended Action: {item['action']}"
        )
        for i, item in enumerate(verified_recommendations, 1)
    )

    # 7. Build final business intelligence report
    verified_report = f"""
VERIFIED BUSINESS PERFORMANCE
--------------------------------

Total Orders: {verified_metrics['total_orders']:,}
Total Customers: {verified_metrics['total_customers']:,}
Total Sales: ₹{verified_metrics['total_sales']:,.2f}
Total Profit: ₹{verified_metrics['total_profit']:,.2f}
Average Order Value: ₹{verified_metrics['average_order_value']:,.2f}


PYTHON-VALIDATED BUSINESS INSIGHTS
--------------------------------

Highest Sales Region: {ai_context['highest_sales_region']['name']}
Lowest Sales Region: {ai_context['lowest_sales_region']['name']}

Highest Profit Region: {ai_context['highest_profit_region']['name']}
Lowest Profit Region: {ai_context['lowest_profit_region']['name']}

Best Customer Segment by Sales: {ai_context['best_customer_segment']['name']}
Lowest Customer Segment by Sales: {ai_context['lowest_customer_segment']['name']}

Most Important ML Feature: {ai_context['most_important_feature']['name']}


LLM-GENERATED BUSINESS EXPLANATION
--------------------------------

{ai_insights}


PYTHON-GENERATED BUSINESS RECOMMENDATIONS
--------------------------------

{recommendation_text}
"""

    print("LLM Agent: Business report generated.")

    return {
        "final_report": verified_report
    }


# --------------------------------------------------
# BUILD MULTI-AGENT COLLABORATION WORKFLOW
# --------------------------------------------------

builder = StateGraph(CollaborationState)

# Register collaboration nodes
builder.add_node(
    "data_analyst",
    data_analyst_collaboration_node
)

builder.add_node(
    "ml_analyst",
    ml_analyst_collaboration_node
)

builder.add_node(
    "visualization",
    visualization_collaboration_node
)

builder.add_node(
    "llm_report",
    final_report_node
)


# Define execution sequence
builder.add_edge(START, "data_analyst")

builder.add_edge("data_analyst", "ml_analyst")

builder.add_edge("ml_analyst", "visualization")

builder.add_edge("visualization", "llm_report")

builder.add_edge("llm_report", END)


# Compile graph
collaboration_graph = builder.compile()


# --------------------------------------------------
# TEST MULTI-AGENT COLLABORATION
# --------------------------------------------------

if __name__ == "__main__":

    initial_state = {
        "user_query": (
            "Analyze overall business performance "
            "and identify improvement opportunities."
        ),

        "data_analysis": {},
        "ml_analysis": {},
        "visualization_results": {},
        "final_report": "",
    }

    print("\nStarting Multi-Agent Collaboration...")
    print("=" * 50)

    result = collaboration_graph.invoke(initial_state)

    print("\nFINAL BUSINESS INTELLIGENCE REPORT")
    print("=" * 50)

    print(result["final_report"])

    print("\nGENERATED CHARTS")
    print("=" * 50)

    for name, path in result["visualization_results"].items():
        print(f"{name}: {path}")

    print("\nMulti-Agent Collaboration Completed!")
