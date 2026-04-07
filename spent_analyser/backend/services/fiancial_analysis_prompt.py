"""
Financial Analysis and Spend Intelligence System Prompt.
Defines the system instructions and expected output format for financial analysis AI agent.
Ensures all LLM outputs are factual, structured, and visualization-ready.
"""

# ===========================
# SYSTEM PROMPT FOR LLM
# ===========================

FINANCIAL_ANALYSIS_SYSTEM_PROMPT = """You are a **Financial Analysis and Spend Intelligence AI Agent** embedded within a production-grade application.

Your role is to analyze **normalized financial transaction data** derived from heterogeneous sources (CSV, PDF bank statements) and generate **accurate, structured, and insight-driven outputs** for visualization and financial understanding.

---

# 🎯 Core Responsibilities

1. Analyze financial transactions (income and expenses)
2. Generate:

   * Summary metrics
   * Category breakdown
   * Time-based trends
   * Behavioral insights
3. Ensure all outputs are:

   * **Strictly factual**
   * **Derived only from provided data**
   * **Structured for graphical consumption**

---

# ⚠️ Critical Rules

* Do NOT assume missing data
* Do NOT hallucinate transactions or categories
* Do NOT override provided classifications unless clearly inconsistent
* If ambiguity exists → flag under `"requires_human_review"`

---

# 🧾 Classification Awareness

You may receive pre-categorized data. If not:

## Income Categories

* Salary
* Business Income
* Interest
* Refund
* Investment Returns
* Other Income

## Expense Categories

* Food & Dining
* Groceries
* Rent / Housing
* Utilities
* Transportation
* Shopping
* Entertainment
* Healthcare
* Insurance
* Education
* Travel
* Investments
* EMI / Loans
* Taxes
* Miscellaneous

If classification confidence is low:

* Use `"Uncategorized"`
* Add to `"requires_human_review"`

---

# 📊 Analytical Requirements

## 1. Summary Metrics

* Total Income
* Total Expense
* Net Savings (Income - Expense)
* Savings Rate (%)

---

## 2. Category Breakdown

* Aggregate totals per category
* Percentage contribution

---

## 3. Time-Based Trends

Generate:

* Monthly trends (mandatory)
* Weekly trends (if sufficient data)
* Daily trends (optional)

Each entry must include:

* income
* expense
* net

---

## 4. Behavioral Insights

Provide insights such as:

* High spending categories
* Increasing/decreasing trends
* Irregular spikes
* Savings observations

⚠️ Do NOT provide speculative financial advice.

---

# 🧠 Reasoning Guidelines

* Use deterministic aggregation for totals
* Use consistent date grouping logic
* Ensure all numbers reconcile (no mismatches)
* Maintain numerical accuracy

---

# 📈 Output Constraints

* Output MUST be valid JSON
* No explanations outside JSON
* No markdown, no extra text
* Values must be numeric (no strings for numbers)
* Dates must follow ISO format

---

# ✅ Success Criteria

* Accurate aggregation
* Clear categorization
* Meaningful trends
* Actionable but non-speculative insights
* Fully visualization-ready structure

---

Act as a **precision financial analytics engine**, not a conversational assistant."""

# ===========================
# EXPECTED OUTPUT JSON SCHEMA
# ===========================

FINANCIAL_ANALYSIS_OUTPUT_SCHEMA = {
    "summary": {
        "total_income": "number (float)",
        "total_expense": "number (float)",
        "net_savings": "number (float)",
        "savings_rate": "number (percentage 0-100)",
    },
    "category_breakdown": {
        "income": [
            {
                "category": "string (Income Category)",
                "amount": "number (float)",
                "percentage": "number (0-100)"
            }
        ],
        "expense": [
            {
                "category": "string (Expense Category)",
                "amount": "number (float)",
                "percentage": "number (0-100)"
            }
        ]
    },
    "trends": {
        "monthly": [
            {
                "period": "string (YYYY-MM)",
                "income": "number (float)",
                "expense": "number (float)",
                "net": "number (float)"
            }
        ],
        "weekly": [],  # Optional
        "daily": []    # Optional
    },
    "insights": [
        "string (actionable, factual insight based on data)"
    ],
    "requires_human_review": [
        {
            "reason": "string (ambiguity or data quality issue)",
            "transaction_reference": "string (transaction ID or description)"
        }
    ]
}

# ===========================
# EXPECTED INPUT CONTEXT
# ===========================

FINANCIAL_ANALYSIS_INPUT_INSTRUCTIONS = """
You will receive:

1. **Normalized Transactions (Canonical Schema)**

Each transaction follows:
{
  "date": "YYYY-MM-DD",
  "description": "string",
  "amount": number,
  "type": "credit/debit",
  "category": "string",
  "confidence": "high/medium/low"
}

2. **Optional Context from Retrieval Systems (RAG)**:
   * Semantic context (FAISS vector search)
   * Relationship insights (Neo4j graph queries)

3. **User Constraints**:
   * Date range filter (mandatory to respect)
"""


def build_financial_analysis_prompt(
    transactions_context: str,
    rag_context: str = "",
    date_range: str = ""
) -> str:
    """
    Build a complete financial analysis prompt for the LLM.
    
    Args:
        transactions_context: Summary or context about transactions
        rag_context: Optional RAG retrieval results
        date_range: Optional date range specification
        
    Returns:
        Complete prompt for LLM analysis
    """
    prompt_parts = [
        "Analyze the following financial transaction data and generate structured insights.\n",
        "OUTPUT FORMAT: Valid JSON only (no other text)\n",
        "---\n"
    ]
    
    if date_range:
        prompt_parts.append(f"DATE RANGE: {date_range}\n")
    
    prompt_parts.append(f"TRANSACTIONS:\n{transactions_context}\n")
    
    if rag_context:
        prompt_parts.append(f"\nADDITIONAL CONTEXT:\n{rag_context}\n")
    
    prompt_parts.extend([
        "\nGenerating analysis following the output schema provided in system instructions.\n",
        "Ensure all metrics are accurate, insights are data-driven, and output is valid JSON.\n"
    ])
    
    return "".join(prompt_parts)


def validate_financial_analysis_output(output: dict) -> tuple[bool, list[str]]:
    """
    Validate financial analysis output against expected schema.
    
    Args:
        output: Dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check summary
    if "summary" not in output:
        errors.append("Missing 'summary' field")
    else:
        summary = output["summary"]
        required_fields = ["total_income", "total_expense", "net_savings", "savings_rate"]
        for field in required_fields:
            if field not in summary:
                errors.append(f"Missing 'summary.{field}'")
            elif not isinstance(summary[field], (int, float)):
                errors.append(f"'summary.{field}' must be numeric")
    
    # Check category_breakdown
    if "category_breakdown" not in output:
        errors.append("Missing 'category_breakdown' field")
    else:
        breakdown = output["category_breakdown"]
        for key in ["income", "expense"]:
            if key not in breakdown:
                errors.append(f"Missing 'category_breakdown.{key}'")
            elif not isinstance(breakdown[key], list):
                errors.append(f"'category_breakdown.{key}' must be a list")
    
    # Check trends
    if "trends" not in output:
        errors.append("Missing 'trends' field")
    else:
        trends = output["trends"]
        if "monthly" not in trends:
            errors.append("Missing 'trends.monthly'")
    
    # Check insights
    if "insights" not in output:
        errors.append("Missing 'insights' field")
    elif not isinstance(output["insights"], list):
        errors.append("'insights' must be a list")
    
    # Check requires_human_review
    if "requires_human_review" not in output:
        errors.append("Missing 'requires_human_review' field")
    elif not isinstance(output["requires_human_review"], list):
        errors.append("'requires_human_review' must be a list")
    
    return len(errors) == 0, errors


def extract_insights_from_analysis(analysis_output: dict) -> list[str]:
    """
    Extract insights from validated financial analysis output.
    
    Args:
        analysis_output: Validated analysis dictionary
        
    Returns:
        List of insight strings
    """
    insights = []
    
    summary = analysis_output.get("summary", {})
    
    # Income/Expense insights
    if summary.get("total_income", 0) > 0:
        insights.append(f"Total income: ${summary['total_income']:,.2f}")
    
    if summary.get("total_expense", 0) > 0:
        insights.append(f"Total expenses: ${summary['total_expense']:,.2f}")
    
    # Savings insights
    net_savings = summary.get("net_savings", 0)
    if net_savings > 0:
        savings_rate = summary.get("savings_rate", 0)
        insights.append(
            f"Net savings: ${net_savings:,.2f} "
            f"({savings_rate:.1f}% savings rate)"
        )
    elif net_savings < 0:
        insights.append(f"Net deficit: ${abs(net_savings):,.2f}")
    
    # Category insights
    categories = analysis_output.get("category_breakdown", {})
    top_expense = categories.get("expense", [])
    if top_expense:
        top = top_expense[0]
        insights.append(
            f"Highest spending category: {top['category']} "
            f"(${top['amount']:,.2f}, {top['percentage']:.1f}%)"
        )
    
    # Trend insights
    trends = analysis_output.get("trends", {}).get("monthly", [])
    if len(trends) > 1:
        latest = trends[-1]
        previous = trends[-2]
        if latest["expense"] > previous["expense"]:
            pct_change = (
                (latest["expense"] - previous["expense"]) / 
                previous["expense"] * 100
            )
            insights.append(
                f"Expenses increased {pct_change:.1f}% "
                f"in {latest['period']} compared to {previous['period']}"
            )
    
    # Add provided insights
    insights.extend(analysis_output.get("insights", []))
    
    return insights