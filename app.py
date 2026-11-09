import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ==================================================
# 1. PROJECT PATHS AND IMPORTS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent
AGENTS_DIR = PROJECT_DIR / "agents"

if str(AGENTS_DIR) not in sys.path:
    sys.path.append(str(AGENTS_DIR))

from supervisor_agent import graph
from llm_agent import generate_business_insight
from data_analyst_agent import (
    get_dataset_summary,
    load_sales_data,
)
from collaboration_agent import collaboration_graph


# ==================================================
# 2. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Multi-Agent Data Analytics Platform",
    page_icon="📊",
    layout="wide",
)


# ==================================================
# 3. SESSION STATE
# ==================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_query" not in st.session_state:
    st.session_state.analysis_query = ""

if "analysis_route" not in st.session_state:
    st.session_state.analysis_route = ""

if "ai_explanation" not in st.session_state:
    st.session_state.ai_explanation = None

if "collaboration_result" not in st.session_state:
    st.session_state.collaboration_result = None


# ==================================================
# 4. DASHBOARD HEADER
# ==================================================

st.title("📊 Multi-Agent Data Analytics Platform")

st.markdown(
    """
    Welcome to your AI-powered analytics platform.

    Ask questions about sales, analyze business performance,
    predict sales for new orders, and generate visualizations.
    """
)

st.divider()


# ==================================================
# 5. LOAD SALES DATA
# ==================================================

try:
    sales_df = load_sales_data().copy()

    sales_df["order_date"] = pd.to_datetime(
        sales_df["order_date"],
        errors="coerce"
    )

    required_columns = [
        "order_id",
        "order_date",
        "customer_id",
        "customer_segment",
        "region",
        "product_category",
        "sales",
        "profit",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in sales_df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing dataset columns: {missing_columns}"
        )
        st.stop()

    sales_df = sales_df.dropna(
        subset=["order_date", "sales", "profit"]
    )

except Exception as e:
    st.error(f"Unable to load sales dataset: {e}")
    st.stop()


# ==================================================
# 6. INTERACTIVE ANALYTICS FILTERS
# ==================================================

st.header("🎛️ Interactive Analytics Filters")

st.write(
    "Select your preferred region, product category, "
    "customer segment, and date range."
)

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    selected_region = st.selectbox(
        "Select Region",
        ["All"] + sorted(
            sales_df["region"].dropna().unique().tolist()
        ),
        key="filter_region",
    )

with filter_col2:
    selected_category = st.selectbox(
        "Select Product Category",
        ["All"] + sorted(
            sales_df["product_category"]
            .dropna()
            .unique()
            .tolist()
        ),
        key="filter_category",
    )

with filter_col3:
    selected_segment = st.selectbox(
        "Select Customer Segment",
        ["All"] + sorted(
            sales_df["customer_segment"]
            .dropna()
            .unique()
            .tolist()
        ),
        key="filter_segment",
    )


# Date range filter

min_date = sales_df["order_date"].min().date()
max_date = sales_df["order_date"].max().date()

selected_dates = st.date_input(
    "Select Order Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="filter_dates",
)


# ==================================================
# 7. APPLY FILTERS
# ==================================================

filtered_df = sales_df.copy()

if selected_region != "All":
    filtered_df = filtered_df[
        filtered_df["region"] == selected_region
    ]

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["product_category"] == selected_category
    ]

if selected_segment != "All":
    filtered_df = filtered_df[
        filtered_df["customer_segment"] == selected_segment
    ]

if len(selected_dates) == 2:
    start_date, end_date = selected_dates

    filtered_df = filtered_df[
        (filtered_df["order_date"].dt.date >= start_date)
        &
        (filtered_df["order_date"].dt.date <= end_date)
    ]


# ==================================================
# 8. FILTERED BUSINESS KPI CARDS
# ==================================================

st.divider()

st.header("📊 Business Performance Overview")

if filtered_df.empty:
    st.warning(
        "No records found for the selected filters. "
        "Please change your filter selections."
    )

else:
    total_sales = filtered_df["sales"].sum()
    total_profit = filtered_df["profit"].sum()

    total_orders = filtered_df["order_id"].nunique()
    total_customers = filtered_df["customer_id"].nunique()

    average_order_value = (
        total_sales / total_orders
        if total_orders > 0
        else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Sales",
            f"₹{total_sales:,.2f}"
        )

    with col2:
        st.metric(
            "Total Profit",
            f"₹{total_profit:,.2f}"
        )

    with col3:
        st.metric(
            "Total Orders",
            f"{total_orders:,}"
        )

    with col4:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col5:
        st.metric(
            "Average Order Value",
            f"₹{average_order_value:,.2f}"
        )


# ==================================================
# 9. FILTERED BUSINESS VISUALIZATIONS
# ==================================================

st.divider()

st.header("📈 Filtered Business Analytics")

if not filtered_df.empty:

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Sales by Region")

        region_chart = (
            filtered_df.groupby("region")["sales"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(
            region_chart,
            x_label="Region",
            y_label="Sales (₹)",
        )

    with chart_col2:
        st.subheader("Sales by Product Category")

        category_chart = (
            filtered_df.groupby("product_category")["sales"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(
            category_chart,
            x_label="Product Category",
            y_label="Sales (₹)",
        )

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.subheader("Profit by Region")

        profit_chart = (
            filtered_df.groupby("region")["profit"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(
            profit_chart,
            x_label="Region",
            y_label="Profit (₹)",
        )

    with chart_col4:
        st.subheader("Sales by Customer Segment")

        segment_chart = (
            filtered_df.groupby("customer_segment")["sales"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(
            segment_chart,
            x_label="Customer Segment",
            y_label="Sales (₹)",
        )


# ==================================================
# 10. FILTERED DATA TABLE
# ==================================================

with st.expander("View Filtered Sales Dataset"):

    st.write(
        f"Showing {len(filtered_df):,} records "
        f"out of {len(sales_df):,} total records."
    )

    st.dataframe(
        filtered_df,
        width="stretch",
        hide_index=True,
    )

    # ==================================================
# 11. INTERACTIVE SALES PREDICTION
# ==================================================

st.divider()

st.header("🤖 Interactive Sales Prediction")

st.write(
    "Enter order details to predict sales using your ML Agent."
)

prediction_col1, prediction_col2, prediction_col3 = st.columns(3)

with prediction_col1:
    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=5,
        step=1,
    )

with prediction_col2:
    unit_price = st.number_input(
        "Unit Price (₹)",
        min_value=0.01,
        value=1000.0,
        step=100.0,
    )

with prediction_col3:
    discount_percent = st.number_input(
        "Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
    )

if st.button("Predict Sales", key="predict_sales_button"):

    try:
        from ml_analyst_agent import predict_sales

        with st.spinner("Predicting sales..."):

            predicted_sales = predict_sales(
                quantity=quantity,
                unit_price=unit_price,
                discount=discount_percent / 100,
            )

        st.metric(
            "Predicted Sales",
            f"₹{predicted_sales:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ==================================================
# 12. DYNAMIC MONTHLY SALES TREND
# ==================================================

st.divider()

st.header("📅 Monthly Sales Trend")

if not filtered_df.empty:

    monthly_sales = (
        filtered_df
        .groupby(
            filtered_df["order_date"].dt.to_period("M")
        )["sales"]
        .sum()
    )

    monthly_sales.index = monthly_sales.index.astype(str)

    st.line_chart(
        monthly_sales,
        x_label="Month",
        y_label="Sales (₹)",
    )

else:
    st.info(
        "No monthly sales data available "
        "for the selected filters."
    )


# ==================================================
# 13. ANALYTICS ASSISTANT
# ==================================================

st.divider()

st.header("💬 Ask Your Analytics Assistant")

user_query = st.text_input(
    "Enter your question",
    placeholder="Example: Which region has the highest sales?",
    key="analytics_query",
)

if st.button("Analyze", key="analyze_button"):

    if user_query.strip():

        try:
            with st.spinner(
                "Agents are analyzing your question..."
            ):

                result = graph.invoke(
                    {
                        "user_query": user_query,
                        "route": "",
                        "result": "",
                    }
                )

            st.session_state.analysis_result = result["result"]
            st.session_state.analysis_query = user_query
            st.session_state.analysis_route = result["route"]
            st.session_state.ai_explanation = None

        except Exception as e:
            st.error(f"Something went wrong: {e}")

    else:
        st.warning("Please enter a question.")


# ==================================================
# 14. DISPLAY ANALYTICS ASSISTANT RESULTS
# ==================================================

if st.session_state.analysis_result is not None:

    analysis_result = st.session_state.analysis_result

    st.success(
        f"Handled by: {st.session_state.analysis_route}"
    )

    df_result = None
    entity_column = None
    entity_name = None

    if isinstance(analysis_result, list):

        df_result = pd.DataFrame(analysis_result)

    possible_entities = {
        "region": "Region",
        "product_category": "Product Category",
        "customer_segment": "Customer Segment",
    }

    if df_result is not None and not df_result.empty:

        for column, label in possible_entities.items():

            if (
                column in df_result.columns
                and "total_sales" in df_result.columns
                and "total_profit" in df_result.columns
            ):

                entity_column = column
                entity_name = label
                break

    verified_facts = ""

    if entity_column is not None:

        sales_leader = df_result.loc[
            df_result["total_sales"].idxmax()
        ]

        profit_leader = df_result.loc[
            df_result["total_profit"].idxmax()
        ]

        verified_facts = (
            f"Highest sales {entity_name.lower()}: "
            f"{sales_leader[entity_column]}, "
            f"₹{sales_leader['total_sales']:,.2f}. "
            f"Highest profit {entity_name.lower()}: "
            f"{profit_leader[entity_column]}, "
            f"₹{profit_leader['total_profit']:,.2f}."
        )

    # AI Explanation

    st.subheader("🧠 AI Business Insights")

    if st.button(
        "Generate AI Explanation",
        key="generate_ai_explanation",
    ):

        try:

            query_for_llm = st.session_state.analysis_query

            if verified_facts:
                query_for_llm += (
                    "\n\nVerified facts calculated from the data: "
                    + verified_facts
                )

            with st.spinner(
                "Llama is generating insights..."
            ):

                explanation = generate_business_insight(
                    query_for_llm,
                    analysis_result,
                )

            st.session_state.ai_explanation = explanation

        except Exception as e:
            st.error(f"LLM explanation failed: {e}")

    if st.session_state.ai_explanation:
        st.write(st.session_state.ai_explanation)

    if verified_facts:
        st.info(f"Verified rankings: {verified_facts}")

    # Display analysis results

    st.subheader("📊 Analysis Results")

    if df_result is not None:

        if entity_column is not None:

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    f"Highest Sales {entity_name}",
                    str(sales_leader[entity_column]),
                    f"₹{sales_leader['total_sales']:,.2f}",
                )

            with col2:
                st.metric(
                    f"Highest Profit {entity_name}",
                    str(profit_leader[entity_column]),
                    f"₹{profit_leader['total_profit']:,.2f}",
                )

        st.dataframe(
            df_result,
            width="stretch",
            hide_index=True,
        )

    elif isinstance(analysis_result, dict):

        if "predicted_sales" in analysis_result:

            st.metric(
                "Predicted Sales",
                f"₹{analysis_result['predicted_sales']:,.2f}",
            )

            st.write(
                f"Quantity: {analysis_result['quantity']}"
            )

            st.write(
                f"Unit price: ₹{analysis_result['unit_price']:,.2f}"
            )

            st.write(
                f"Discount: {analysis_result['discount']:.0%}"
            )

        elif "chart_path" in analysis_result:

            generated_chart = Path(
                analysis_result["chart_path"]
            )

            if generated_chart.exists():

                st.image(
                    str(generated_chart),
                    caption="Generated Sales Chart",
                    width="stretch",
                )

            else:
                st.warning(
                    "The generated chart file was not found."
                )

        else:
            st.json(analysis_result)

    else:
        st.write(analysis_result)

        # ==================================================
# 15. MULTI-AGENT BUSINESS INTELLIGENCE
# ==================================================

st.divider()

st.title("🤝 Multi-Agent Business Intelligence")

st.write(
    """
    Collaborate with multiple AI agents to analyze business
    performance, evaluate machine learning results, generate
    visualizations, and prepare a business intelligence report.
    """
)

collaboration_query = st.text_area(
    "Enter your business analysis request",
    placeholder=(
        "Example: Analyze overall business performance "
        "and identify improvement opportunities."
    ),
    key="collaboration_query",
    height=100,
)


# ==================================================
# 16. MULTI-AGENT EXECUTION WITH LIVE MONITORING
# ==================================================

if st.button(
    "Run Multi-Agent Analysis",
    key="run_collaboration",
):

    if collaboration_query.strip():

        initial_state = {
            "user_query": collaboration_query,
            "data_analysis": {},
            "ml_analysis": {},
            "visualization_results": {},
            "final_report": "",
        }

        try:

            with st.status(
                "Multi-Agent workflow is running...",
                expanded=True,
            ) as workflow_status:

                result = initial_state.copy()

                for update in collaboration_graph.stream(
                    initial_state,
                    stream_mode="updates",
                ):

                    for agent_name, agent_output in update.items():

                        result.update(agent_output)

                        display_name = (
                            agent_name
                            .replace("_", " ")
                            .title()
                        )

                        workflow_status.write(
                            f"✅ {display_name} completed"
                        )

                workflow_status.update(
                    label="All agents completed successfully!",
                    state="complete",
                    expanded=False,
                )

                st.session_state.collaboration_result = result

            st.success(
                "Multi-Agent Analysis completed successfully!"
            )

        except Exception as e:

            st.error(
                f"Multi-Agent Analysis failed: {e}"
            )

    else:

        st.warning(
            "Please enter a business analysis request."
        )


# ==================================================
# 17. DISPLAY MULTI-AGENT RESULTS
# ==================================================

if st.session_state.collaboration_result is not None:

    result = st.session_state.collaboration_result

    summary = result["data_analysis"]["overall_summary"]

    st.divider()

    # BUSINESS KPI CARDS

    st.subheader("📊 Business Performance Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Sales",
            f"₹{summary['total_sales']:,.0f}"
        )

    with col2:
        st.metric(
            "Total Profit",
            f"₹{summary['total_profit']:,.0f}"
        )

    with col3:
        st.metric(
            "Total Orders",
            f"{summary['total_orders']:,}"
        )

    with col4:
        st.metric(
            "Total Customers",
            f"{summary['total_customers']:,}"
        )

    with col5:
        st.metric(
            "Average Order Value",
            f"₹{summary['average_order_value']:,.2f}"
        )

    # BUSINESS INTELLIGENCE REPORT

    st.divider()

    st.subheader("📋 Business Intelligence Report")

    st.markdown(
        result["final_report"].replace("\n", "  \n")
    )

    # GENERATED VISUALIZATIONS

    st.divider()

    st.subheader("📊 Generated Business Visualizations")

    visualization_results = result["visualization_results"]

    chart_titles = {
        "sales_by_region": "Sales by Region",
        "sales_by_category": "Sales by Product Category",
        "profit_by_region": "Profit by Region",
    }

    chart_columns = st.columns(3)

    for column, (name, path) in zip(
        chart_columns,
        visualization_results.items(),
    ):

        chart_path = Path(path)

        with column:

            st.markdown(
                f"**{chart_titles.get(name, name)}**"
            )

            if chart_path.exists():

                st.image(
                    str(chart_path),
                    width="stretch",
                )

            else:

                st.warning(
                    f"Chart not found: {path}"
                )


# ==================================================
# END OF APPLICATION
# ==================================================