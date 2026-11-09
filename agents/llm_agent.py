import json
import ollama

MODEL_NAME = "llama3.2:3b"


def generate_business_insight(user_query, analysis_result):
    """
    Generate a concise business explanation using verified analytics data.
    Rankings are calculated in Python, not left to the LLM.
    """

    # Convert DataFrame-like results to a list of dictionaries if needed.
    if hasattr(analysis_result, "to_dict"):
        analysis_result = analysis_result.to_dict(orient="records")

    verified_summary = {
        "user_question": user_query,
        "analytics_results": analysis_result,
    }

    # Calculate rankings from the actual numerical data.
    if isinstance(analysis_result, list) and analysis_result:
        first_row = analysis_result[0]

        if isinstance(first_row, dict):
            for metric in ("total_sales", "total_profit"):
                if metric in first_row:
                    highest = max(
                        analysis_result,
                        key=lambda row: row.get(metric, float("-inf")),
                    )
                    lowest = min(
                        analysis_result,
                        key=lambda row: row.get(metric, float("inf")),
                    )

                    verified_summary[f"highest_{metric}"] = {
                        "category": (
                            highest.get("region")
                            or highest.get("customer_segment")
                            or highest.get("product_category")
                            or highest.get("product")
                        ),
                        "value": highest[metric],
                    }

                    verified_summary[f"lowest_{metric}"] = {
                        "category": (
                            lowest.get("region")
                            or lowest.get("customer_segment")
                            or lowest.get("product_category")
                            or lowest.get("product")
                        ),
                        "value": lowest[metric],
                    }

    prompt = f"""
You are a precise business analytics assistant.

User question:
{user_query}

Verified data and pre-calculated rankings:
{json.dumps(verified_summary, ensure_ascii=False, indent=2, default=str)}

Instructions:
- Answer the user's exact question first.
- Use Indian Rupees (₹) for monetary values.
- Preserve all supplied figures accurately.
- Use the pre-calculated rankings when discussing highest or lowest values.
- Never contradict the supplied rankings or calculate different rankings.
- Do not invent causes, trends, or unsupported business explanations.
- If the data does not answer the question, clearly say so.
- Give no more than 3 concise bullet points.
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

    return response["message"]["content"]


if __name__ == "__main__":
    test_result = [
        {
            "region": "East",
            "total_sales": 2728661.43,
            "total_profit": 809029.58,
        },
        {
            "region": "South",
            "total_sales": 2716481.95,
            "total_profit": 822294.35,
        },
        {
            "region": "West",
            "total_sales": 2687747.76,
            "total_profit": 815241.46,
        },
        {
            "region": "North",
            "total_sales": 2663006.85,
            "total_profit": 795922.37,
        },
    ]

    insight = generate_business_insight(
        "Which region has the highest sales and which has the highest profit?",
        test_result,
    )

    print("\nBusiness Insight")
    print("----------------")
    print(insight)
