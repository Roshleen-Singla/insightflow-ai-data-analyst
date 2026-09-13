''' In this particular section, i will be building the analysis tools that will help us understand sales, quantity, and revenue data. 
The tools will be designed to provide insights into the data, allowing us to make informed decisions based on the analysis.
The tools will include functions for calculating key metrics, generating visualizations, and performing statistical analyses. 
By leveraging these tools, we can gain a deeper understanding of our sales performance and identify areas for improvement.
The list is as follows:
1. Sales analysis
    Total sales
    Total quantity
    Average order value
    Sales by month
    Sales by state
    Sales by category

2. Product analysis
    Best-selling categories
    Best-selling products/SKUs
    Quantity by category
    Revenue by category

3. Location analysis
    Top states
    Top cities
    Sales by state
    Orders by state

4. Order analysis
    Order status distribution
    Cancelled orders
    Delivered orders
    Fulfilment analysis
    B2B vs B2C
 '''

import json
from langchain_core.tools import tool

def load_cleaned_data(file_path: str):
    """
    Load the cleaned dataset.
    """
    return pd.read_csv(file_path)

# SALES ANALYSIS TOOLS
import pandas as pd

def analyze_sales(df: pd.DataFrame):
    """
    Analyze overall sales performance.
    """

    results = {}

    # -----------------------------
    # BASIC SALES METRICS
    # -----------------------------

    results["total_sales"] = round(
        df["amount"].sum(), 2
    )

    results["total_units_sold"] = int(
        df["quantity"].sum()
    )

    results["total_orders"] = int(
        df["order_id"].nunique()
    )

    results["average_order_value"] = round(
        df.groupby("order_id")["amount"]
        .sum()
        .mean(),
        2
    )

    # -----------------------------
    # SALES BY MONTH
    # -----------------------------

    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce"
    )

    monthly_sales = (
        df.groupby(
            df["order_date"].dt.to_period("M")
        )["amount"]
        .sum()
        .sort_index()
    )

    results["monthly_sales"] = {
        str(month): round(sales, 2)
        for month, sales in monthly_sales.items()
    }

    # -----------------------------
    # BEST MONTH
    # -----------------------------

    if not monthly_sales.empty:

        best_month = monthly_sales.idxmax()

        results["best_sales_month"] = {
            "month": str(best_month),
            "sales": round(
                monthly_sales.max(), 2
            )
        }

    return results


# PRODUCT ANALYSIS TOOLS
def analyze_products(df: pd.DataFrame):
    """
    Analyze product performance.
    """

    results = {}

    # -----------------------------
    # BEST-SELLING CATEGORIES
    # -----------------------------

    category_sales = (
        df.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(5) # top 5 best-selling categories
    )

    results["best_selling_categories"] = {
        category: round(sales, 2)
        for category, sales in category_sales.items()
    }

    # -----------------------------
    # BEST-SELLING PRODUCTS/SKUs
    # -----------------------------

    product_sales = (
        df.groupby("sku")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(5) # top 5 best-selling products
    )

    results["best_selling_products"] = {
        sku: round(sales, 2)
        for sku, sales in product_sales.items()
    }

    #------------------------------
    # QUANTITY BY CATEGORY
    #------------------------------

    category_quantity = (
        df.groupby("category")["quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    results["quantity_by_category"] = {
        category: round(quantity, 2)
        for category, quantity in category_quantity.items()
    }

    #------------------------------
    # REVENUE BY CATEGORY
    #------------------------------

    category_revenue = (
        df.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    results["revenue_by_category"] = {
        category: round(revenue, 2)
        for category, revenue in category_revenue.items()
    }

    return results 


# LOCATION ANALYSIS TOOLS
def analyze_locations(df: pd.DataFrame):
    """
    Analyze sales performance by location.
    """

    results = {}

    # -----------------------------
    # TOP STATES (Sales by state)
    # -----------------------------

    state_sales = (
        df.groupby("ship_state")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    results["top_states"] = {
        state: round(sales, 2)
        for state, sales in state_sales.items()
    }

    # -----------------------------
    # TOP CITIES (Sales by city)
    # -----------------------------

    city_sales = (
        df.groupby("ship_city")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10) # top 10 cities by sales
    )

    results["top_cities"] = {
        city: round(sales, 2)
        for city, sales in city_sales.items()
    }

    # ------------------------------
    #  ORDERS BY STATE
    #-------------------------------

    state_orders = (
        df.groupby("ship_state")["order_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    results["orders_by_state"] = {
        state: orders
        for state, orders in state_orders.items()
    }

    return results


# ORDER ANALYSIS TOOLS
def analyze_orders(df: pd.DataFrame):
    """
    Analyze order performance.
    """

    results = {}

    # -----------------------------
    # ORDER STATUS DISTRIBUTION
    # -----------------------------

    status_distribution = (
        df.groupby("status")["order_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    results["order_status_distribution"] = {
        status: orders
        for status, orders in status_distribution.items()
    }

    #----------------------------------
    # CANCELLED ORDERS
    #----------------------------------
    cancelled_orders = (
        df[df["status"] == "Cancelled"]
        .groupby("fulfilment")["order_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    results["cancelled_orders"] = {
        fulfilment: orders
        for fulfilment, orders in cancelled_orders.items()
    }

    
    #----------------------------------
    # DELIVERED ORDERS
    #----------------------------------
    delivered_orders = (
        df[df["status"] == "Shipped - Delivered to Buyer"]
        .groupby("fulfilment")["order_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    results["delivered_orders"] = {
        fulfilment: orders
        for fulfilment, orders in delivered_orders.items()
    }

    #----------------------------------
    # FULLFILLMENT ANALYSIS
    #----------------------------------
    fulfilment_analysis = (
        df.groupby("fulfilment")["order_id"]
        .nunique()
        .sort_values(ascending=False)   
    )

    results["fulfilment_analysis"] = {
        fulfilment: orders
        for fulfilment, orders in fulfilment_analysis.items()
    }

    #----------------------------------
    # SALES CHANNEL ANALYSIS
    #----------------------------------
    sales_channel_analysis = (
        df.groupby("sales_channel")["order_id"]
        .count()
        .sort_values(ascending=False)
    )

    results["sales_channel_analysis"] = {
        channel: orders
        for channel, orders in sales_channel_analysis.items()
    }

    #----------------------------------
    # B2B vs B2C
    #----------------------------------
    b2b_b2c_analysis = (
        df.groupby("b2b")["order_id"]
        .nunique()
    .sort_values(ascending=False)
     )

    results["b2b_b2c_analysis"] = {
        b2b: orders
        for b2b, orders in b2b_b2c_analysis.items()
    }

    return results



def run_analysis(file_path):

    df = load_cleaned_data(file_path)

    sales = analyze_sales(df)
    products = analyze_products(df)
    locations = analyze_locations(df)
    orders = analyze_orders(df)

    return {
        "sales": sales,
        "products": products,
        "locations": locations,
        "orders": orders
    }


@tool
def analysis_tool(
    file_path: str = "data/cleaned_amazon_sales.csv"
) -> str:
    """
    Runs sales, product, location, and order analysis on the cleaned
    Amazon sales dataset. Use this when the user asks about revenue,
    top products, sales trends, or order performance.
    """

    results = run_analysis(file_path)

    return json.dumps(
        results,
        indent=2,
        default=str
    )