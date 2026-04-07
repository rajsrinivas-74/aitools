"""
Streamlit Frontend for FinSight AI.
Professional spending analysis with AI-powered insights.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import sys
from pathlib import Path

# Add project root to path so imports work regardless of how script is run
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables (settings module handles .env loading)
from config.settings import settings

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Initializing Streamlit Frontend")

# Page configuration
st.set_page_config(
    page_title="FinSight AI",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Mode Professional Theme
st.markdown("""
<style>
    /* Root Colors */
    :root {
        --primary-bg: #0F1419;      /* Very dark blue-gray */
        --secondary-bg: #1A1F2E;    /* Darker blue-gray */
        --tertiary-bg: #252D3D;     /* Medium dark blue-gray */
        --accent-blue: #0EA5E9;     /* Professional blue */
        --accent-green: #10B981;    /* Success green */
        --accent-red: #EF4444;      /* Alert red */
        --text-primary: #F1F5F9;    /* Almost white */
        --text-secondary: #CBD5E1;  /* Light gray */
        --border-color: #334155;    /* Subtle border */
    }
    
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #0F1419 0%, #1A1F2E 100%);
        color: var(--text-primary);
    }
    
    /* Headers and titles */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1F2E 0%, #252D3D 100%);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] h2 {
        color: var(--accent-blue) !important;
        border-bottom: 2px solid var(--accent-blue);
        padding-bottom: 12px;
        margin-bottom: 20px;
    }
    
    [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        margin-top: 20px;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1A1F2E 0%, #252D3D 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1A1F2E 0%, #252D3D 100%) !important;
        padding: 20px !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Metric label and value text */
    [data-testid="metric-container"] > div > label {
        color: #FFFFFF !important;
        font-size: 0.875rem !important;
    }
    
    [data-testid="metric-container"] > div > div {
        color: #FFFFFF !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        box-shadow: 0 6px 16px rgba(14, 165, 233, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: var(--tertiary-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stDateInput > div > div > input:focus,
    .stTimeInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1) !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background: linear-gradient(135deg, #252D3D 0%, #1A1F2E 100%);
        border: 2px dashed var(--border-color);
        border-radius: 8px;
        padding: 20px;
    }
    
    /* Alert boxes */
    .stAlert {
        background: linear-gradient(135deg, #1A1F2E 0%, #252D3D 100%) !important;
        border-radius: 8px !important;
        border-left: 4px solid var(--accent-blue) !important;
    }
    
    .stAlert > div > div > div {
        color: var(--text-primary) !important;
    }
    
    /* Success messages */
    .stAlert.success {
        border-left-color: var(--accent-green) !important;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%) !important;
    }
    
    /* Warning/Error messages */
    .stAlert.warning,
    .stAlert.error {
        border-left-color: var(--accent-red) !important;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%) !important;
    }
    
    /* Expandable sections */
    .stExpander {
        border: 1px solid var(--border-color) !important;
        background: transparent !important;
        border-radius: 8px !important;
    }
    
    .stExpander > div > div > button > p {
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    
    /* Subheaders */
    .stSubheader {
        color: #FFFFFF !important;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 8px;
    }
    
    /* Text and labels */
    .stLabel {
        color: var(--text-secondary) !important;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    /* Dividers */
    hr {
        border-color: var(--border-color) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--tertiary-bg);
        border-bottom: 1px solid var(--border-color);
    }
    
    .stTabs [role="tab"] {
        color: var(--text-secondary) !important;
        border-bottom: 2px solid transparent;
    }
    
    .stTabs [role="tab"][aria-selected="true"] {
        color: var(--accent-blue) !important;
        border-bottom-color: var(--accent-blue) !important;
    }
    
    /* Spinner and progress */
    .stSpinner {
        color: var(--accent-blue) !important;
    }
    
    /* Info box */
    .stInfo {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(14, 165, 233, 0.05) 100%) !important;
        border-left: 4px solid var(--accent-blue) !important;
        border-radius: 8px !important;
    }
    
    /* Code blocks */
    code {
        background: var(--tertiary-bg) !important;
        color: #10B981 !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
    }
    
    pre {
        background: var(--secondary-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }
    
    /* Markdown text */
    .stMarkdown p {
        color: var(--text-secondary);
    }
    
    /* JSON viewer */
    .stJson {
        background: var(--secondary-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }
    
    /* Plotly charts */
    .plotly-graph-div {
        filter: invert(0) !important;
    }
    
    /* Slider */
    .stSlider [role="slider"] {
        background: var(--accent-blue) !important;
    }
    
    /* Fixed summary box */
    .fixed-summary {
        position: fixed;
        top: 60px;
        right: 20px;
        width: 300px;
        background: linear-gradient(135deg, #1A1F2E 0%, #252D3D 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        z-index: 999;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .fixed-summary h3 {
        color: #0EA5E9 !important;
        margin-top: 0;
        font-size: 1.1rem;
        border-bottom: 2px solid var(--accent-blue);
        padding-bottom: 10px;
    }
    
    .fixed-summary p {
        color: #CBD5E1;
        font-size: 0.9rem;
        margin: 8px 0;
        line-height: 1.5;
    }
    
    .fixed-summary .summary-stat {
        background: rgba(14, 165, 233, 0.1);
        padding: 8px 12px;
        border-radius: 6px;
        margin: 8px 0;
        border-left: 3px solid #0EA5E9;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
BACKEND_URL = "http://localhost:5000"


def get_backend_health() -> bool:
    """Check if backend is running."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def upload_files(uploaded_files: List) -> Dict:
    """Upload files to backend."""
    if not uploaded_files:
        return {"error": "No files selected"}
    
    files = [('files', file) for file in uploaded_files]
    
    try:
        response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=30)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def analyze_transactions(start_date: datetime, end_date: datetime) -> Optional[Dict]:
    """Trigger simple analysis on backend (fast, transaction-only)."""
    try:
        payload = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "analysis_type": "simple",  # Always use simple mode for frontend
            "boost_confidence": True    # Always boost confidence with LLM
        }
        
        response = requests.post(f"{BACKEND_URL}/analyze", json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_data = response.json()
                st.error(f"Analysis error: {error_data.get('error', 'Unknown error')}")
            except:
                st.error(f"Analysis error: HTTP {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None


def query_transactions(user_query: str, start_date: datetime, end_date: datetime) -> Optional[Dict]:
    """Query LLM about transactions with user question."""
    if not user_query.strip():
        st.error("Please enter a question to query")
        return None
    
    try:
        payload = {
            "query": user_query,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        
        response = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_data = response.json()
                st.error(f"Query error: {error_data.get('error', 'Unknown error')}")
            except:
                st.error(f"Query error: HTTP {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        return None


def render_summary(summary: Dict):
    """Render summary metrics with professional styling."""
    # Support both old and new field names for backward compatibility
    total_income = summary.get('total_income', summary.get('income', 0))
    total_expense = summary.get('total_expense', summary.get('expense', 0))
    net_savings = summary.get('net_savings', summary.get('net', 0))
    savings_rate = summary.get('savings_rate', 0)
    transaction_count = summary.get('transaction_count', 0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Income metric - Green accent
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(16, 185, 129, 0.02) 100%);
            border-left: 4px solid #10B981;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        '>
            <p style='color: #CBD5E1; font-size: 0.875rem; margin: 0 0 8px 0;'>Total Income</p>
            <p style='color: #10B981; font-size: 1.75rem; font-weight: 700; margin: 0;'>${:,.2f}</p>
        </div>
        """.format(total_income), unsafe_allow_html=True)
    
    with col2:
        # Expense metric - Red accent
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.05) 0%, rgba(239, 68, 68, 0.02) 100%);
            border-left: 4px solid #EF4444;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        '>
            <p style='color: #CBD5E1; font-size: 0.875rem; margin: 0 0 8px 0;'>Total Expenses</p>
            <p style='color: #EF4444; font-size: 1.75rem; font-weight: 700; margin: 0;'>${:,.2f}</p>
        </div>
        """.format(total_expense), unsafe_allow_html=True)
    
    with col3:
        # Net saving - Blue accent (or dynamic based on value)
        net_color = "#10B981" if net_savings > 0 else "#EF4444" if net_savings < 0 else "#0EA5E9"
        savings_label = "positive" if net_savings > 0 else "negative" if net_savings < 0 else "neutral"
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.05) 0%, rgba(14, 165, 233, 0.02) 100%);
            border-left: 4px solid {net_color};
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        '>
            <p style='color: #CBD5E1; font-size: 0.875rem; margin: 0 0 8px 0;'>Net Savings</p>
            <p style='color: {net_color}; font-size: 1.75rem; font-weight: 700; margin: 0;'>${net_savings:,.2f}</p>
            <p style='color: #94A3B8; font-size: 0.75rem; margin: 4px 0 0 0;'>{savings_rate:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Transaction count metric - Blue accent
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.05) 0%, rgba(14, 165, 233, 0.02) 100%);
            border-left: 4px solid #0EA5E9;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        '>
            <p style='color: #CBD5E1; font-size: 0.875rem; margin: 0 0 8px 0;'>Transactions</p>
            <p style='color: #0EA5E9; font-size: 1.75rem; font-weight: 700; margin: 0;'>{}</p>
        </div>
        """.format(transaction_count), unsafe_allow_html=True)


def render_category_chart(categories):
    """Render category pie chart."""
    # Handle both old (list) and new (dict with income/expense) formats
    if isinstance(categories, dict):
        expense_cats = categories.get('expense', [])
        if not expense_cats:
            st.warning("No expense category data available")
            return
        df = pd.DataFrame(expense_cats)
        chart_data = df
        title = 'Expense Categories'
        names_field = 'category'
    else:
        # Legacy format
        if not categories:
            st.warning("No category data available")
            return
        df = pd.DataFrame(categories)
        chart_data = df
        title = 'Spending by Category'
        names_field = df.columns[0] if 'name' not in df.columns else 'name'
    
    fig = px.pie(
        chart_data,
        values='amount',
        names=names_field,
        title=title,
        hover_data=['percentage']
    )
    
    st.plotly_chart(fig, width='stretch')


def render_trends_chart(trends: List[Dict]):
    """Render trends line chart."""
    if not trends:
        st.warning("No trend data available")
        return
    
    df = pd.DataFrame(trends)
    df['date'] = pd.to_datetime(df['date'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['income'],
        mode='lines+markers',
        name='Income',
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['expense'],
        mode='lines+markers',
        name='Expenses',
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['net'],
        mode='lines+markers',
        name='Net',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title='Income vs Expenses Over Time',
        xaxis_title='Date',
        yaxis_title='Amount ($)',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, width='stretch')


def render_analysis_summary(analysis: Dict):
    """Render formatted analysis summary with key findings, habits, and recommendations."""
    
    # Executive Summary
    executive_summary = analysis.get('executive_summary', '')
    if executive_summary:
        st.subheader("📋 Executive Summary")
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.05) 0%, rgba(14, 165, 233, 0.02) 100%);
            border-left: 5px solid #0EA5E9;
            border-radius: 8px;
            padding: 16px;
            margin: 12px 0;
        '>
            <p style='color: #F1F5F9; margin: 0; line-height: 1.6;'>{executive_summary}</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Key Findings
    with col1:
        st.subheader("🔍 Key Findings")
        key_findings = analysis.get('key_findings', [])
        if key_findings:
            for i, finding in enumerate(key_findings, 1):
                st.markdown(f"**{i}. {finding}**")
        else:
            st.info("No key findings available")
    
    # Recommendations
    with col2:
        st.subheader("💡 Recommendations")
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}. {rec}**")
        else:
            st.info("No recommendations available")
    
    # Spending Habits
    st.subheader("📊 Spending Habits")
    spending_habits = analysis.get('spending_habits', [])
    if spending_habits:
        cols = st.columns(min(3, len(spending_habits)))
        for idx, habit in enumerate(spending_habits[:3]):
            with cols[idx % 3]:
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, rgba(14, 165, 233, 0.05) 0%, rgba(14, 165, 233, 0.02) 100%);
                    border-left: 4px solid #0EA5E9;
                    border-radius: 8px;
                    padding: 12px;
                    margin: 8px 0;
                '>
                    <p style='color: #F1F5F9; margin: 0; font-size: 0.95rem;'>{habit}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No spending habits identified")
    
    # Risk Alerts
    risk_alerts = analysis.get('risk_alerts', [])
    if risk_alerts:
        st.subheader("⚠️ Risk Alerts")
        for alert in risk_alerts:
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.05) 0%, rgba(239, 68, 68, 0.02) 100%);
                border-left: 4px solid #EF4444;
                border-radius: 8px;
                padding: 12px;
                margin: 8px 0;
            '>
                <p style='color: #F1F5F9; margin: 0;'>🚨 {alert}</p>
            </div>
            """, unsafe_allow_html=True)


def render_fixed_summary(analysis: Dict = None, metadata: Dict = None):
    """Render fixed summary box in top-right corner with sections in order: Financial Summary, Detailed Analysis, High Value Transactions, Spending by Category."""
    
    summary_html = """
    <div class="fixed-summary">
        <h3>✨ FinSight AI</h3>
    """
    
    if analysis:
        # 1. FINANCIAL SUMMARY
        fin_summary = analysis.get('financial_summary', {})
        if fin_summary:
            summary_html += f"""
            <p style='margin-top: 16px; font-weight: 700; color: #0EA5E9; border-bottom: 2px solid #0EA5E9; padding-bottom: 8px;'>💰 Financial Summary</p>
            <p>Debits: <span style='color: #EF4444; font-weight: 600;'>${fin_summary.get('total_debit', 0):,.0f}</span></p>
            <p>Credits: <span style='color: #10B981; font-weight: 600;'>${fin_summary.get('total_credit', 0):,.0f}</span></p>
            <p>Net: <span style='color: #0EA5E9; font-weight: 600;'>${fin_summary.get('net_balance', 0):,.0f}</span></p>
            <p>Avg Transaction: <span style='color: #0EA5E9;'>${fin_summary.get('average_transaction', 0):,.0f}</span></p>
            """
        
        # 2. DETAILED ANALYSIS
        recommendations = analysis.get('recommendations', [])
        habits = analysis.get('spending_habits', [])
        if recommendations or habits:
            summary_html += f"""
            <p style='margin-top: 16px; font-weight: 700; color: #0EA5E9; border-bottom: 2px solid #0EA5E9; padding-bottom: 8px;'>📊 Detailed Analysis</p>
            """
            if recommendations:
                summary_html += "<p style='font-weight: 600; color: #CBD5E1;'>Recommendations:</p>"
                for i, rec in enumerate(recommendations[:2], 1):
                    summary_html += f"<p>• {rec}</p>"
            if habits:
                summary_html += "<p style='font-weight: 600; color: #CBD5E1; margin-top: 8px;'>Spending Habits:</p>"
                for habit in habits[:2]:
                    summary_html += f"<p>• {habit}</p>"
        
        # 3. KEY FINDINGS (Executive Summary)
        exec_summary = analysis.get('executive_summary', '')
        key_findings = analysis.get('key_findings', [])
        if exec_summary or key_findings:
            summary_html += f"""
            <p style='margin-top: 16px; font-weight: 700; color: #0EA5E9; border-bottom: 2px solid #0EA5E9; padding-bottom: 8px;'>🔍 High Value Transactions</p>
            """
            if exec_summary:
                summary_html += f"<p style='font-style: italic; color: #CBD5E1; margin: 8px 0;'>{exec_summary}</p>"
            if key_findings:
                for finding in key_findings[:2]:
                    summary_html += f"<p>• {finding}</p>"
    else:
        summary_html += """
        <p style='color: #94A3B8; font-style: italic;'>Upload files and click Analyze to see summary here...</p>
        """
    
    summary_html += "</div>"
    
    st.markdown(summary_html, unsafe_allow_html=True)


def render_bottom_right_details(analysis: Dict = None):
    """Render detailed analysis in bottom right area."""
    
    if not analysis:
        return
    
    st.subheader("📊 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    # Recommendations
    with col1:
        st.subheader("💡 Recommendations")
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}. {rec}**")
        else:
            st.info("No recommendations")
    
    # Spending Habits
    with col2:
        st.subheader("📈 Spending Habits")
        habits = analysis.get('spending_habits', [])
        if habits:
            for habit in habits[:3]:
                st.markdown(f"• {habit}")
        else:
            st.info("No habits identified")
    
    # Risk Alerts (spanning full width if present)
    risk_alerts = analysis.get('risk_alerts', [])
    if risk_alerts:
        st.subheader("⚠️ Risk Alerts")
        for alert in risk_alerts:
            st.warning(f"🚨 {alert}")


def render_main_content(analysis: Dict = None, metadata: Dict = None):
    """Render main analysis content in order: Financial Summary, Detailed Analysis, High Value Transactions, Spending by Category."""
    
    if not analysis or not metadata:
        st.info("👈 Upload files and click 'Analyze' to see results")
        return
    
    # 1. FINANCIAL SUMMARY - Enhanced Visual Design
    st.subheader("💰 Financial Summary")
    fin_summary = analysis.get('financial_summary', {})
    
    # Create enhanced visual cards for metrics - ensure numeric conversion
    total_debit = float(fin_summary.get('total_debit', 0)) if fin_summary.get('total_debit') else 0
    total_credit = float(fin_summary.get('total_credit', 0)) if fin_summary.get('total_credit') else 0
    net_balance = float(fin_summary.get('net_balance', 0)) if fin_summary.get('net_balance') else 0
    avg_transaction = float(fin_summary.get('average_transaction', 0)) if fin_summary.get('average_transaction') else 0
    
    # Color coding for net balance
    net_color = "#10B981" if net_balance >= 0 else "#EF4444"  # Green for positive, Red for negative
    debit_color = "#EF4444"  # Red for debits (money out)
    credit_color = "#10B981"  # Green for credits (money in)
    neutral_color = "#0EA5E9"  # Blue for average
    
    # Create styled metrics cards using string formatting instead of f-string for safety
    debit_str = f"${total_debit:,.2f}"
    credit_str = f"${total_credit:,.2f}"
    balance_str = f"${net_balance:,.2f}"
    avg_str = f"${avg_transaction:,.2f}"
    balance_label = "Positive" if net_balance >= 0 else "Negative"
    
    # Build metrics HTML with clean formatting
    metrics_html = '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0;">'
    
    # Debits Card
    metrics_html += '<div style="background: linear-gradient(135deg, #1A1F2E 0%, #252D3D 100%); border: 2px solid #EF4444; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);">'
    metrics_html += '<p style="color: #CBD5E1; font-size: 13px; margin: 0 0 8px 0; text-transform: uppercase;">Debits</p>'
    metrics_html += '<p style="color: #EF4444; font-size: 28px; font-weight: 700; margin: 0;">' + debit_str + '</p>'
    metrics_html += '<p style="color: #94A3B8; font-size: 12px; margin: 8px 0 0 0;">Outgoing</p></div>'
    
    # Credits Card
    metrics_html += '<div style="background: linear-gradient(135deg, #1A1F2E 0%, #252D3D 100%); border: 2px solid #10B981; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);">'
    metrics_html += '<p style="color: #CBD5E1; font-size: 13px; margin: 0 0 8px 0; text-transform: uppercase;">Credits</p>'
    metrics_html += '<p style="color: #10B981; font-size: 28px; font-weight: 700; margin: 0;">' + credit_str + '</p>'
    metrics_html += '<p style="color: #94A3B8; font-size: 12px; margin: 8px 0 0 0;">Incoming</p></div>'
    
    # Net Balance Card
    metrics_html += '<div style="background: linear-gradient(135deg, #1A1F2E 0%, #252D3D 100%); border: 2px solid ' + net_color + '; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);">'
    metrics_html += '<p style="color: #CBD5E1; font-size: 13px; margin: 0 0 8px 0; text-transform: uppercase;">Net Balance</p>'
    metrics_html += '<p style="color: ' + net_color + '; font-size: 28px; font-weight: 700; margin: 0;">' + balance_str + '</p>'
    metrics_html += '<p style="color: #94A3B8; font-size: 12px; margin: 8px 0 0 0;">' + balance_label + '</p></div>'
    
    # Average Transaction Card
    metrics_html += '<div style="background: linear-gradient(135deg, #1A1F2E 0%, #252D3D 100%); border: 2px solid #0EA5E9; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);">'
    metrics_html += '<p style="color: #CBD5E1; font-size: 13px; margin: 0 0 8px 0; text-transform: uppercase;">Avg Transaction</p>'
    metrics_html += '<p style="color: #0EA5E9; font-size: 28px; font-weight: 700; margin: 0;">' + avg_str + '</p>'
    metrics_html += '<p style="color: #94A3B8; font-size: 12px; margin: 8px 0 0 0;">Per Transaction</p></div>'
    
    metrics_html += '</div>'
    
    st.markdown(metrics_html, unsafe_allow_html=True)
    
    # Executive Summary
    executive_summary = analysis.get('executive_summary', '')
    if executive_summary:
        st.markdown("<div style='margin: 24px 0;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a472a 0%, #0f2818 100%); 
                    border-left: 4px solid #10B981; 
                    border-radius: 8px; 
                    padding: 16px; 
                    margin: 16px 0;">
            <p style='color: #10B981; font-weight: bold; font-size: 16px; margin: 0 0 12px 0;'>📋 Executive Summary</p>
            <p style='color: #FFFFFF; line-height: 1.6; margin: 0;'>{executive_summary}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Findings
    key_findings = analysis.get('key_findings', [])
    if key_findings:
        findings_html = """
        <div style="background: linear-gradient(135deg, #1a3a4a 0%, #0f2633 100%); 
                    border-left: 4px solid #0EA5E9; 
                    border-radius: 8px; 
                    padding: 16px; 
                    margin: 16px 0;">
            <p style='color: #0EA5E9; font-weight: bold; font-size: 16px; margin: 0 0 12px 0;'>🔍 Key Findings</p>
        """
        for finding in key_findings:
            findings_html += f"<p style='color: #FFFFFF; margin: 8px 0; line-height: 1.5;'>✓ {finding}</p>"
        findings_html += "</div>"
        st.markdown(findings_html, unsafe_allow_html=True)
    
    # Spending Habits
    habits = analysis.get('spending_habits', [])
    if habits:
        habits_html = """
        <div style="background: linear-gradient(135deg, #3a2a1a 0%, #2a1a0f 100%); 
                    border-left: 4px solid #F59E0B; 
                    border-radius: 8px; 
                    padding: 16px; 
                    margin: 16px 0;">
            <p style='color: #F59E0B; font-weight: bold; font-size: 16px; margin: 0 0 12px 0;'>📈 Spending Habits</p>
        """
        for habit in habits:
            habits_html += f"<p style='color: #FFFFFF; margin: 8px 0; line-height: 1.5;'>→ {habit}</p>"
        habits_html += "</div>"
        st.markdown(habits_html, unsafe_allow_html=True)
    
    # Recommendations
    recommendations = analysis.get('recommendations', [])
    if recommendations:
        rec_html = """
        <div style="background: linear-gradient(135deg, #2a3a1a 0%, #1a2a0f 100%); 
                    border-left: 4px solid #84CC16; 
                    border-radius: 8px; 
                    padding: 16px; 
                    margin: 16px 0;">
            <p style='color: #84CC16; font-weight: bold; font-size: 16px; margin: 0 0 12px 0;'>💡 Recommendations</p>
        """
        for i, rec in enumerate(recommendations, 1):
            rec_html += f"<p style='color: #FFFFFF; margin: 8px 0; line-height: 1.5;'><strong>{i}.</strong> {rec}</p>"
        rec_html += "</div>"
        st.markdown(rec_html, unsafe_allow_html=True)
    
    # Risk Alerts
    risk_alerts = analysis.get('risk_alerts', [])
    if risk_alerts:
        alerts_html = """
        <div style="background: linear-gradient(135deg, #3a1a1a 0%, #2a0f0f 100%); 
                    border-left: 4px solid #EF4444; 
                    border-radius: 8px; 
                    padding: 16px; 
                    margin: 16px 0;">
            <p style='color: #EF4444; font-weight: bold; font-size: 16px; margin: 0 0 12px 0;'>⚠️ Risk Alerts</p>
        """
        for alert in risk_alerts:
            alerts_html += f"<p style='color: #FFFFFF; margin: 8px 0; line-height: 1.5;'>🚨 {alert}</p>"
        alerts_html += "</div>"
        st.markdown(alerts_html, unsafe_allow_html=True)
    
    st.divider()
    
    # 2. HIGH-VALUE TRANSACTIONS
    if 'high_value_transactions' in analysis and analysis['high_value_transactions']:
        st.subheader("💳 High-Value Transactions")
        render_transactions_tree(analysis['high_value_transactions'])
        st.divider()
    
    # 3. SPENDING BY CATEGORY
    st.subheader("📊 Spending by Category")
    spending_cats = analysis.get('spending_by_category', [])
    if spending_cats:
        col1, col2 = st.columns(2)
        with col1:
            for cat in spending_cats[:5]:
                st.markdown(f"**{cat['category']}**: ${cat['total']:,.2f} ({cat['count']} trans)")
        with col2:
            if spending_cats:
                df_cats = pd.DataFrame(spending_cats[:5])
                fig = px.pie(df_cats, names='category', values='total', title='Top 5 Categories')
                st.plotly_chart(fig, width='stretch')


def render_transactions_tree(transactions: List[Dict]):
    """Render transactions in a tree structure organized by category."""
    
    if not transactions:
        st.info("No transactions available")
        return
    
    # Organize transactions by category
    from collections import defaultdict
    by_category = defaultdict(list)
    
    for trans in transactions:
        category = trans.get('category', 'Other')
        by_category[category].append(trans)
    
    # Sort categories by total amount
    category_totals = []
    for cat, trans_list in by_category.items():
        total = sum(t.get('amount', 0) for t in trans_list)
        category_totals.append((cat, total, trans_list))
    
    category_totals.sort(key=lambda x: x[1], reverse=True)
    
    # Display as tree structure with expandable categories
    for category, total, trans_list in category_totals:
        with st.expander(f"📁 {category} (${total:,.2f} | {len(trans_list)} transactions)"):
            # Create dataframe for this category
            cat_data = []
            for trans in sorted(trans_list, key=lambda t: t['date'], reverse=True):
                cat_data.append({
                    'Date': trans['date'],
                    'Description': trans['description'][:40],
                    'Amount': f"${trans['amount']:.2f}",
                    'Type': trans['type'].upper(),
                    'Confidence': f"{trans.get('confidence', 0):.0%}"
                })
            
            # Display as a compact table
            df = pd.DataFrame(cat_data)
            st.dataframe(df, width='stretch', hide_index=True)


def render_insights(insights: List[str], requires_review: List[Dict]):
    """Render insights and review items with professional color styling."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Key Insights")
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.05) 0%, rgba(14, 165, 233, 0.02) 100%);
            border-left: 4px solid #0EA5E9;
            border-radius: 8px;
            padding: 12px;
        '>
        """ , unsafe_allow_html=True)
        
        for insight in insights[:5]:
            st.markdown(f"• {insight}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.subheader("⚠️ Requires Review")
        if requires_review:
            for item in requires_review[:5]:
                # Color-code based on amount/risk
                accent_color = "#EF4444" if float(item['amount']) > 1000 else "#F97316" if float(item['amount']) > 500 else "#EAB308"
                category = item.get('category', 'Other')
                
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, rgba(239, 68, 68, 0.05) 0%, rgba(239, 68, 68, 0.02) 100%);
                    border-left: 4px solid {accent_color};
                    border-radius: 8px;
                    padding: 12px;
                    margin: 8px 0;
                '>
                    <p style='color: #F1F5F9; font-weight: 600; margin: 0 0 4px 0;'>{item['date']} - {item['description'][:40]}...</p>
                    <p style='color: #CBD5E1; font-size: 0.875rem; margin: 0 0 4px 0;'>
                        <span style='color: #0EA5E9;'>📁 {category}</span> | 
                        <span style='color: {accent_color};'>${item['amount']:,.2f}</span>
                    </p>
                    <p style='color: #94A3B8; font-size: 0.75rem; margin: 0;'>
                        Confidence: {item.get('confidence', 'N/A')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(16, 185, 129, 0.02) 100%);
                border-left: 4px solid #10B981;
                border-radius: 8px;
                padding: 12px;
                text-align: center;
            '>
                <p style='color: #10B981; margin: 0;'>✓ No items requiring review</p>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    # Enhanced title with professional styling
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(14, 165, 233, 0.05) 100%);
        border-left: 5px solid #0EA5E9;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    '>
        <h1 style='color: #0EA5E9; margin: 0 0 8px 0;'>✨ FinSight AI</h1>
        <p style='color: #CBD5E1; margin: 0; font-size: 1.1rem;'>
            Professional spending analysis powered by artificial intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check backend connection
    if not get_backend_health():
        st.error("""
        ❌ **Backend connection failed**
        
        Please ensure the Flask backend is running:
        ```bash
        python backend/app.py
        ```
        """)
        return
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # File upload section
    st.sidebar.subheader("📁 Upload Files")
    uploaded_files = st.sidebar.file_uploader(
        "Select CSV or PDF files",
        type=['csv', 'pdf'],
        accept_multiple_files=True,
        help="Upload transaction data in CSV or PDF format"
    )
    
    if uploaded_files:
        if st.sidebar.button("📤 Upload Files"):
            with st.spinner("Uploading files..."):
                result = upload_files(uploaded_files)
            
            if "error" not in result:
                st.sidebar.success(f"✅ Uploaded {len(result['uploaded_files'])} files")
                st.sidebar.write(f"Total transactions found: {result['total_transactions']}")
                
                if result['errors']:
                    with st.sidebar.expander("⚠️ Errors"):
                        for error in result['errors']:
                            st.write(f"- {error['filename']}: {error['error']}")
            else:
                st.sidebar.error(f"❌ Upload failed: {result['error']}")
    
    # Date range section
    st.sidebar.subheader("📅 Date Range")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=90)
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
    
    # Analyze button
    if st.sidebar.button("🔍 Analyze"):
        if not uploaded_files:
            st.warning("Please upload files first")
            return
        
        with st.spinner("⏳ Analyzing transactions..."):
            results = analyze_transactions(
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time())
            )
        
        if results:
            st.session_state.analysis_results = results
            st.success("✅ Analysis complete!")
        else:
            st.error("❌ Analysis failed")
    
    # Divider
    st.sidebar.divider()
    
    # Query LLM section
    st.sidebar.subheader("💬 Query LLM")
    user_query = st.sidebar.text_area(
        "Ask a question about your transactions",
        placeholder="e.g., What are my top spending categories? Where can I save money?",
        height=80,
        help="Ask the LLM anything about your transaction data"
    )
    
    if st.sidebar.button("🤖 Query"):
        if not uploaded_files:
            st.warning("Please upload files first")
        elif not user_query.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("Querying LLM..."):
                query_result = query_transactions(
                    user_query,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time())
                )
            
            if query_result:
                st.session_state.query_result = query_result
                st.sidebar.success("✅ Query complete!")
            else:
                st.sidebar.error("❌ Query failed")
    
    # Display results if available
    if "analysis_results" in st.session_state and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Handle both old format (with 'summary' key) and new format (with 'analysis' key)
        if isinstance(results, dict) and 'analysis' in results:
            analysis = results['analysis']
            metadata = results.get('metadata', {})
            
            # Display Main Content in Center (includes all sections in order)
            render_main_content(analysis, metadata)
            
        # Old format compatibility
        elif isinstance(results, dict) and 'summary' in results:
            st.subheader("📈 Summary")
            render_summary(results['summary'])
            
            # Charts
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Category Distribution")
                render_category_chart(results.get('categories', []))
            with col2:
                st.subheader("Trends")
                render_trends_chart(results.get('trends', []))
            
            # Insights
            st.subheader("💡 Analysis")
            render_insights(results.get('insights', []), results.get('requires_human_review', []))
        else:
            st.error("❌ Invalid results format or missing analysis data")
    else:
        st.info("👈 Upload files and click 'Analyze' to see results")
    
    # Display Query Results if available
    if "query_result" in st.session_state and st.session_state.query_result:
        st.divider()
        
        st.subheader("🤖 LLM Query Response")
        query_result = st.session_state.query_result
        
        # Display the analysis metadata
        if "metadata" in query_result:
            metadata = query_result["metadata"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Analysis Type", metadata.get("analysis_type", "unknown"))
            with col2:
                st.metric("Transactions Analyzed", metadata.get("transactions_analyzed", 0))
            with col3:
                st.metric("Response Time", f"{metadata.get('timestamp', 'unknown')[-8:]}")
        
        # Display the response
        if isinstance(query_result, dict):
            if "response" in query_result:
                st.markdown("""
                <div style='
                    background: linear-gradient(135deg, rgba(14, 165, 233, 0.05) 0%, rgba(14, 165, 233, 0.02) 100%);
                    border-left: 4px solid #0EA5E9;
                    border-radius: 8px;
                    padding: 16px;
                    margin: 12px 0;
                '>
                    <p style='color: #CBD5E1; font-size: 0.875rem; margin: 0 0 8px 0;'>Your Question:</p>
                    <p style='color: #F1F5F9; font-size: 0.95rem; margin: 0 0 16px 0;'><em>""" + query_result.get("query", "") + """</em></p>
                    <hr style='border-color: #334155; margin: 12px 0;'>
                    <p style='color: #CBD5E1; font-size: 0.875rem; margin: 0 0 8px 0;'>LLM Response:</p>
                    <p style='color: #F1F5F9; margin: 0;'>""" + query_result["response"].replace('\n', '<br>') + """</p>
                </div>
                """, unsafe_allow_html=True)
            elif "answer" in query_result:
                st.markdown("""
                <div style='
                    background: linear-gradient(135deg, rgba(14, 165, 233, 0.05) 0%, rgba(14, 165, 233, 0.02) 100%);
                    border-left: 4px solid #0EA5E9;
                    border-radius: 8px;
                    padding: 16px;
                    margin: 12px 0;
                '>
                """, unsafe_allow_html=True)
                
                st.markdown(query_result["answer"])
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.json(query_result)


if __name__ == "__main__":
    # Initialize session state
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "query_result" not in st.session_state:
        st.session_state.query_result = None
    
    main()