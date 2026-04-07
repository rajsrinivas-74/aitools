"""
Flask REST API for Spending Analysis.
Uses UnifiedExtractor for modern transaction processing and LLM-powered analysis.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from decimal import Decimal

# Add project root to path so imports work regardless of how script is run
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, jsonify, request
from flask_cors import CORS

# Import settings (which loads .env files automatically)
from config.settings import settings
from backend.services.parser import DocumentParser
from backend.unified_extraction import UnifiedExtractor
from backend.models.transaction import Transaction, TransactionBatch, TransactionType

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Suppress Flask/Werkzeug INFO logs (HTTP server logs)
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)
flask_logger = logging.getLogger('flask')
flask_logger.setLevel(logging.ERROR)

# Create Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
app.config['SECRET_KEY'] = settings.SECRET_KEY

# Initialize services
parser = DocumentParser()
extractor = UnifiedExtractor()

# Global state
uploaded_files = {}
analysis_results = {}


@app.route('/', methods=['GET'])
def index():
    """Root endpoint - returns API information and available endpoints."""
    return jsonify({
        "app": settings.APP_NAME,
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /config": "Get application configuration",
            "POST /upload": "Upload transaction files (CSV/PDF)",
            "POST /analyze": "Analyze uploaded transactions",
            "POST /query": "Query transactions with user question using LLM",
            "GET /results": "Get latest analysis results",
            "GET /transactions": "Get all uploaded transactions with category details"
        }
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "app": settings.APP_NAME,
        "environment": settings.APP_ENV
    }), 200


@app.route('/config', methods=['GET'])
def get_config():
    """Get application configuration."""
    return jsonify({
        "app_name": settings.APP_NAME,
        "app_env": settings.APP_ENV,
        "flask_host": settings.FLASK_HOST,
        "flask_port": settings.FLASK_PORT,
        "openai_model": settings.OPENAI_MODEL,
        "neo4j_uri": settings.NEO4J_URI,
    }), 200


@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Upload and parse transaction files.
    
    POST /upload
    Files: Multiple CSV or PDF files
    Returns: File metadata and parsing status
    """
    logger.debug(f"Received upload request from {request.remote_addr}")
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({"error": "No files selected"}), 400
        
        results = {
            "uploaded_files": [],
            "errors": [],
            "total_transactions": 0
        }
        
        for file in files:
            if not file.filename:
                continue
            
            logger.debug(f"Processing uploaded file: {file.filename}")
            # Save uploaded file
            file_path = settings.UPLOAD_DIR / file.filename
            file.save(file_path)
            logger.debug(f"File saved to {file_path}")
            
            try:
                # Parse file
                logger.debug(f"Parsing file: {file.filename}")
                transactions, schema_mapping = parser.parse_and_normalize(str(file_path))
                logger.debug(f"Successfully parsed {len(transactions)} transactions with confidence {schema_mapping.confidence_score}")
                
                # Store results
                uploaded_files[file.filename] = {
                    "transactions": transactions,
                    "schema_mapping": schema_mapping,
                    "file_path": str(file_path)
                }
                
                results["uploaded_files"].append({
                    "filename": file.filename,
                    "status": "success",
                    "transactions_found": len(transactions),
                    "schema_confidence": schema_mapping.confidence_score
                })
                
                results["total_transactions"] += len(transactions)
                
            except Exception as e:
                results["errors"].append({
                    "filename": file.filename,
                    "error": str(e)
                })
                logger.error(f"Error parsing {file.filename}: {str(e)}")
        
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze_transactions():
    """
    Analyze uploaded transactions using LLM-powered unified pipeline.
    
    POST /analyze
    JSON: {
        "start_date": "YYYY-MM-DD" (optional),
        "end_date": "YYYY-MM-DD" (optional),
        "analysis_type": "simple" or "comprehensive" (optional, defaults to "simple"),
        "use_llm_for_enhancement": true/false (optional, defaults to false)
            - Uses LLM to boost confidence and improve category mapping
    }
    
    Analysis Modes:
    - simple: ⚡ Fast (2-9s), transaction-only, no database overhead
    - comprehensive: 🐢 Slower (15-25s), uses FAISS embeddings + Neo4j graphs
    """
    logger.debug(f"Analyze request from {request.remote_addr}")
    try:
        if not uploaded_files:
            return jsonify({"error": "No files uploaded. Upload files first using /upload endpoint."}), 400
        
        # Parse request
        data = request.get_json() or {}
        start_date_str = data.get('start_date')
        end_date_str = data.get('end_date')
        analysis_type = data.get('analysis_type', 'simple')
        use_llm_for_enhancement = data.get('use_llm_for_enhancement', False)
        
        if analysis_type not in ('simple', 'comprehensive'):
            return jsonify({"error": "analysis_type must be 'simple' or 'comprehensive'"}), 400
        
        # Collect transactions from all uploaded files
        all_canonical = []
        for file_data in uploaded_files.values():
            all_canonical.extend(file_data["transactions"])
        
        if not all_canonical:
            return jsonify({"error": "No transactions found in uploaded files."}), 400
        
        logger.debug(f"Processing {len(all_canonical)} transactions")
        
        # Filter by date if provided
        if start_date_str and end_date_str:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            all_canonical = [
                t for t in all_canonical
                if start_date.date() <= datetime.strptime(t.date, "%Y-%m-%d").date() <= end_date.date()
            ]
            logger.debug(f"After date filter: {len(all_canonical)} transactions")
        
        # Convert to unified Transaction format
        unified_transactions = []
        for trans in all_canonical:
            unified_transactions.append(Transaction(
                date=trans.date,
                description=trans.description,
                amount=Decimal(str(trans.amount)),
                type=TransactionType.DEBIT if trans.type.lower() == 'debit' else TransactionType.CREDIT,
                currency=getattr(trans, 'currency', '$'),
                category=getattr(trans, 'category', 'Other'),
                confidence_score=0.85,
                extraction_method='parser',
                source_file='uploaded',
                balance=Decimal(str(trans.balance)) if hasattr(trans, 'balance') and trans.balance else None,
            ))
        
        # Create batch
        batch = TransactionBatch(
            transactions=unified_transactions,
            source_file='uploaded_transactions',
            extraction_method='parser'
        )
        
        logger.debug(f"Batch created: {len(batch.transactions)} transactions, avg confidence {batch.total_confidence:.2f}")
        
        # Apply LLM enhancement (confidence boosting + category mapping) if enabled
        if use_llm_for_enhancement and extractor.llm_service:
            logger.debug("Applying LLM confidence boosting and category mapping")
            batch = extractor.boost_confidence_with_llm(batch, confidence_threshold=0.80)
            logger.debug(f"After LLM enhancement: avg confidence {batch.total_confidence:.2f}")
        
        # Generate analysis
        logger.debug(f"Generating {analysis_type} analysis")
        if analysis_type == 'simple':
            # Simple analysis: LLM enhancement applied above, final insights from transaction data only
            logger.debug("Simple analysis mode: Transaction-based final insights")
            analysis = extractor.generate_simple_analysis(batch)
        else:
            # Comprehensive analysis: Database indexing + LLM-powered insights with embeddings + graph
            db_results = extractor.load_to_databases(batch)
            logger.debug(f"DB load: FAISS={db_results.get('faiss_indexed')}, Neo4j={db_results.get('neo4j_indexed')}")
            analysis = extractor.generate_llm_analysis_summary(batch, user_id="default_user")
        
        # Store results
        analysis_results["latest"] = {
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": analysis_type,
            "batch_stats": {
                "total_transactions": len(batch.transactions),
                "average_confidence": batch.total_confidence,
                "high_confidence_count": batch.high_confidence_count,
                "medium_confidence_count": batch.medium_confidence_count,
                "low_confidence_count": batch.low_confidence_count,
            }
        }
        
        logger.debug(f"Analysis complete: {analysis_type} mode")
        return jsonify({
            "status": "success",
            "analysis": analysis,
            "metadata": {
                "analysis_type": analysis_type,
                "total_transactions": len(batch.transactions),
                "average_confidence": round(batch.total_confidence, 2),
                "timestamp": datetime.now().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/results', methods=['GET'])
def get_results():
    """
    Get latest analysis results.
    
    GET /results
    Returns: Latest analysis result with metadata
    """
    logger.debug("Retrieving latest analysis results")
    try:
        if "latest" not in analysis_results:
            logger.warning("No analysis results available yet")
            return jsonify({"error": "No analysis results available"}), 404
        
        result = analysis_results["latest"]
        logger.debug(f"Returning analysis results (type: {result['analysis_type']})")
        
        return jsonify({
            "status": "success",
            "analysis": result["analysis"],
            "metadata": {
                "analysis_type": result["analysis_type"],
                "timestamp": result["timestamp"],
                "batch_stats": result["batch_stats"],
                "total_transactions": result["batch_stats"]["total_transactions"]
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Results error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/transactions', methods=['GET'])
def get_transactions():
    """
    Get all uploaded transactions with complete details including category.
    
    GET /transactions?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&category=Category
    
    Query Parameters:
    - start_date: Filter by start date (optional)
    - end_date: Filter by end date (optional)
    - category: Filter by category (optional)
    
    Returns: List of all transactions with category information
    """
    logger.debug(f"Retrieving transactions request from {request.remote_addr}")
    try:
        if not uploaded_files:
            return jsonify({"error": "No files uploaded"}), 400
        
        # Collect transactions from all uploaded files
        all_canonical = []
        for file_data in uploaded_files.values():
            all_canonical.extend(file_data["transactions"])
        
        if not all_canonical:
            return jsonify({"error": "No transactions found"}), 400
        
        # Parse query parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        category_filter = request.args.get('category')
        
        # Filter by date if provided
        if start_date_str and end_date_str:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            all_canonical = [
                t for t in all_canonical
                if start_date.date() <= datetime.strptime(t.date, "%Y-%m-%d").date() <= end_date.date()
            ]
        
        # Filter by category if provided
        if category_filter:
            all_canonical = [
                t for t in all_canonical
                if getattr(t, 'category', 'Other').lower() == category_filter.lower()
            ]
        
        # Convert to JSON-serializable format with category
        transactions_list = []
        for trans in all_canonical:
            transactions_list.append({
                'date': trans.date,
                'description': trans.description,
                'amount': round(float(trans.amount), 2),
                'type': trans.type.lower() if hasattr(trans.type, 'lower') else trans.type,
                'category': getattr(trans, 'category', 'Other'),
                'currency': getattr(trans, 'currency', '$'),
                'balance': round(float(trans.balance), 2) if hasattr(trans, 'balance') and trans.balance else None,
                'confidence': round(getattr(trans, 'confidence_score', 0.85), 2)
            })
        
        logger.debug(f"Returning {len(transactions_list)} transactions")
        
        return jsonify({
            "status": "success",
            "transactions": transactions_list,
            "metadata": {
                "total_count": len(transactions_list),
                "date_range": {
                    "from": transactions_list[0]['date'] if transactions_list else None,
                    "to": transactions_list[-1]['date'] if transactions_list else None
                }
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Transactions error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query_transactions():
    """
    Query transactions with user question using LLM-powered analysis.
    
    POST /query
    JSON: {
        "query": "User's question about transactions",
        "start_date": "YYYY-MM-DD" (optional),
        "end_date": "YYYY-MM-DD" (optional)
    }
    
    Returns: LLM response answering the user's question based on analysis summary
    """
    logger.debug(f"Query request from {request.remote_addr}")
    try:
        if not uploaded_files:
            return jsonify({"error": "No files uploaded. Upload files first using /upload endpoint."}), 400
        
        # Parse request
        data = request.get_json() or {}
        user_query = data.get('query', '').strip()
        start_date_str = data.get('start_date')
        end_date_str = data.get('end_date')
        
        if not user_query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Collect transactions from all uploaded files
        all_canonical = []
        for file_data in uploaded_files.values():
            all_canonical.extend(file_data["transactions"])
        
        if not all_canonical:
            return jsonify({"error": "No transactions found in uploaded files."}), 400
        
        logger.debug(f"Processing {len(all_canonical)} transactions for query: {user_query[:50]}...")
        
        # Filter by date if provided
        if start_date_str and end_date_str:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            all_canonical = [
                t for t in all_canonical
                if start_date.date() <= datetime.strptime(t.date, "%Y-%m-%d").date() <= end_date.date()
            ]
            logger.debug(f"After date filter: {len(all_canonical)} transactions")
        
        # Convert to unified Transaction format
        unified_transactions = []
        for trans in all_canonical:
            unified_transactions.append(Transaction(
                date=trans.date,
                description=trans.description,
                amount=Decimal(str(trans.amount)),
                type=TransactionType.DEBIT if trans.type.lower() == 'debit' else TransactionType.CREDIT,
                currency=getattr(trans, 'currency', '$'),
                category=getattr(trans, 'category', 'Other'),
                confidence_score=0.85,
                extraction_method='parser',
                source_file='uploaded',
                balance=Decimal(str(trans.balance)) if hasattr(trans, 'balance') and trans.balance else None,
            ))
        
        # Create batch
        batch = TransactionBatch(
            transactions=unified_transactions,
            source_file='uploaded_transactions',
            extraction_method='parser'
        )
        
        logger.debug(f"Batch created: {len(batch.transactions)} transactions, avg confidence {batch.total_confidence:.2f}")
        
        # Generate simple analysis (transaction-only, fast)
        logger.debug("Generating simple analysis for query context...")
        analysis = extractor.generate_simple_analysis(batch)
        
        if not extractor.llm_service:
            return jsonify({"error": "LLM service not available"}), 500
        
        # Build prompt using analysis context + user question
        logger.debug(f"Building prompt with analysis context and user query...")
        
        # Extract key data from analysis
        financial_summary = analysis.get('financial_summary', {})
        spending_by_category = analysis.get('spending_by_category', [])
        high_value = analysis.get('high_value_transactions', [])
        key_findings = analysis.get('key_findings', [])
        recommendations = analysis.get('recommendations', [])
        
        # Build comprehensive context
        context = f"""TRANSACTION ANALYSIS CONTEXT:

Financial Summary:
- Total Debits: ${financial_summary.get('total_debit', 0):,.2f}
- Total Credits: ${financial_summary.get('total_credit', 0):,.2f}
- Net Balance: ${financial_summary.get('net_balance', 0):,.2f}
- Average Transaction: ${financial_summary.get('average_transaction', 0):,.2f}
- Total Transactions: {analysis.get('transaction_count', 0)}
- Average Confidence: {analysis.get('average_confidence', 0):.1%}

Spending by Category:"""
        
        for cat in spending_by_category[:5]:
            context += f"\n- {cat.get('category', 'Unknown')}: ${cat.get('total', 0):,.2f} ({cat.get('count', 0)} transactions)"
        
        context += "\n\nHigh-Value Transactions:"
        for trans in high_value[:5]:
            context += f"\n- {trans.get('date', 'Unknown')}: {trans.get('description', '')[:40]} | ${trans.get('amount', 0):,.2f}"
        
        if key_findings:
            context += "\n\nKey Findings:"
            for finding in key_findings[:3]:
                context += f"\n- {finding}"
        
        if recommendations:
            context += "\n\nRecommendations:"
            for rec in recommendations[:3]:
                context += f"\n- {rec}"
        
        prompt = f"""{context}

USER QUESTION: {user_query}

Using the transaction analysis context above, please answer the user's question. Be specific, professional, and base your answer on the data provided.
"""
        
        # Query LLM with analysis context
        logger.debug("Sending query to LLM with analysis context...")
        response = extractor.llm_service.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert financial analyst. Answer questions about spending patterns based on transaction analysis data. Be precise, actionable, and reference specific numbers from the analysis context when appropriate."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        llm_response = response.choices[0].message.content
        logger.debug("LLM query completed successfully")
        
        return jsonify({
            "status": "success",
            "query": user_query,
            "response": llm_response,
            "metadata": {
                "analysis_type": analysis.get('analysis_type', 'simple'),
                "transactions_analyzed": len(batch.transactions),
                "timestamp": datetime.now().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors."""
    return jsonify({
        "error": f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB"
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    logger.debug(f"Starting {settings.APP_NAME} backend server...")
    app.run(
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        debug=settings.DEBUG,
        use_reloader=False  # Disable auto-reloader to prevent double initialization
    )