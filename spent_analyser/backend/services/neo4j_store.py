"""
Neo4j Graph Database Store for transaction relationships.
"""

import logging
from typing import List, Dict, Optional

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
except ImportError:
    GraphDatabase = None

from backend.models import ProcessedTransaction
from config.settings import settings

logger = logging.getLogger(__name__)


class Neo4jStoreError(Exception):
    """Custom exception for Neo4j operations."""
    pass


class Neo4jStore:
    """Neo4j graph database for transaction relationships."""
    
    def __init__(self):
        """Initialize Neo4j connection."""
        self.logger = logging.getLogger(__name__)
        
        # Log initialization start
        self.logger.info("="*70)
        self.logger.info("🔗 NEO4J STORE INITIALIZATION")
        self.logger.info("="*70)
        
        if GraphDatabase is None:
            self.logger.warning("⚠ Neo4j driver not installed")
            self.connected = False
            self.driver = None
            return
        
        self.driver = None
        self.connected = False
        
        try:
            self._connect()
        except Exception as e:
            self.logger.error(f"❌ Could not initialize Neo4j connection: {str(e)}")
    
    def _connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            # Log connection parameters
            self.logger.info("📡 NEO4J CONNECTION PARAMETERS:")
            self.logger.info(f"  URI: {settings.NEO4J_URI}")
            self.logger.info(f"  USERNAME: {settings.NEO4J_USERNAME}")
            self.logger.info(f"  PASSWORD: {'[SET]' if settings.NEO4J_PASSWORD and settings.NEO4J_PASSWORD != 'password' else '[DEFAULT]'}")
            
            # Determine if we need to set encryption explicitly
            # For encrypted schemes (neo4j+s, neo4j+ssc, bolt+s, bolt+ssc), encryption is handled by the scheme
            # For standard schemes (neo4j, bolt), we set the encrypted parameter
            encrypted_schemes = ['neo4j+s', 'neo4j+ssc', 'bolt+s', 'bolt+ssc']
            is_encrypted_scheme = any(settings.NEO4J_URI.startswith(scheme + '://') for scheme in encrypted_schemes)
            
            if is_encrypted_scheme:
                self.logger.info(f"  ENCRYPTION: Handled by URI scheme")
            else:
                self.logger.info(f"  ENCRYPTION: False (explicit)")
            
            self.logger.info("🔐 Attempting to connect to Neo4j database...")
            
            # Create driver with appropriate settings
            if is_encrypted_scheme:
                # For encrypted schemes, don't set the encrypted parameter
                self.driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
                )
            else:
                # For standard schemes, explicitly set encryption
                self.driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
                    encrypted=False
                )
            
            # Test connection by running a simple query
            self.logger.info("🧪 Testing Neo4j connection...")
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            
            self.connected = True
            self.logger.info("✅ NEO4J CONNECTION SUCCESSFUL")
            self.logger.info(f"  Connected to: {settings.NEO4J_URI}")
            self.logger.info(f"  Username: {settings.NEO4J_USERNAME}")
            self.logger.info("="*70)
            
        except ServiceUnavailable as e:
            self.logger.error(f"❌ NEO4J SERVICE UNAVAILABLE: {str(e)}")
            self.logger.error(f"   Could not reach: {settings.NEO4J_URI}")
            self.connected = False
            
        except AuthError as e:
            self.logger.error(f"❌ NEO4J AUTHENTICATION FAILED: {str(e)}")
            self.logger.error(f"   Username: {settings.NEO4J_USERNAME}")
            self.logger.error("   Check credentials in .env file")
            self.connected = False
            
        except Exception as e:
            self.logger.error(f"❌ NEO4J CONNECTION ERROR: {type(e).__name__}: {str(e)}")
            self.connected = False
    
    def add_transactions(self, transactions: List[ProcessedTransaction], user_id: str = "default_user") -> None:
        """Add transactions to graph database."""
        if not self.connected or not self.driver:
            self.logger.warning(f"⚠ Not connected to Neo4j, skipping add_transactions for user: {user_id}")
            return
        
        try:
            self.logger.info(f"📝 Adding {len(transactions)} transactions to Neo4j for user: {user_id}")
            
            with self.driver.session() as session:
                for i, transaction in enumerate(transactions, 1):
                    self._add_transaction_to_graph(session, transaction, user_id)
            
            self.logger.info(f"✅ Successfully added {len(transactions)} transactions to Neo4j")
            
        except Exception as e:
            self.logger.error(f"❌ Error adding transactions: {type(e).__name__}: {str(e)}")
    
    def _add_transaction_to_graph(self, session, transaction: ProcessedTransaction, user_id: str) -> None:
        """Add single transaction to graph."""
        try:
            # Create/merge user node
            session.run(
                "MERGE (u:User {id: $user_id})",
                user_id=user_id
            )
            
            # Create/merge category node
            session.run(
                "MERGE (c:Category {name: $category})",
                category=transaction.category
            )
            
            # Create transaction node
            session.run(
                """
                CREATE (t:Transaction {
                    id: $trans_id,
                    date: $date,
                    amount: $amount,
                    description: $description,
                    type: $trans_type
                })
                """,
                trans_id=transaction.transaction_id,
                date=transaction.canonical.date,
                amount=float(transaction.canonical.amount),
                description=transaction.canonical.description,
                trans_type="income" if transaction.is_income else "expense"
            )
            
            # Create relationship: User -> Transaction
            session.run(
                """
                MATCH (u:User {id: $user_id})
                MATCH (t:Transaction {id: $trans_id})
                MERGE (u)-[:MADE]->(t)
                """,
                user_id=user_id,
                trans_id=transaction.transaction_id
            )
            
            # Create relationship: Transaction -> Category
            session.run(
                """
                MATCH (t:Transaction {id: $trans_id})
                MATCH (c:Category {name: $category})
                MERGE (t)-[:BELONGS_TO]->(c)
                """,
                trans_id=transaction.transaction_id,
                category=transaction.category
            )
            
            self.logger.debug(f"  ✓ Added transaction: {transaction.transaction_id} | ${transaction.canonical.amount} | {transaction.category}")
            
        except Exception as e:
            self.logger.error(f"❌ Error adding transaction {transaction.transaction_id}: {type(e).__name__}: {str(e)}")
    
    def get_spending_patterns(self, user_id: str = "default_user", limit: int = 10) -> List[Dict]:
        """Query spending patterns from graph."""
        if not self.connected or not self.driver:
            self.logger.warning(f"⚠ Not connected to Neo4j, cannot query patterns for user: {user_id}")
            return []
        
        try:
            self.logger.info(f"🔍 Querying spending patterns for user: {user_id} (limit: {limit})")
            
            results = []
            with self.driver.session() as session:
                query_result = session.run(
                    """
                    MATCH (u:User {id: $user_id})-[:MADE]->(t:Transaction)-[:BELONGS_TO]->(c:Category)
                    WITH c.name as category, COUNT(t) as count, SUM(t.amount) as total
                    ORDER BY total DESC
                    LIMIT $limit
                    RETURN category, count, total
                    """,
                    user_id=user_id,
                    limit=limit
                )
                
                for record in query_result:
                    results.append({
                        "category": record["category"],
                        "count": record["count"],
                        "total": record["total"]
                    })
            
            self.logger.info(f"✅ Found {len(results)} spending patterns for user: {user_id}")
            if results:
                self.logger.debug(f"   Top patterns: {', '.join([r['category'] for r in results[:3]])}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error querying patterns: {type(e).__name__}: {str(e)}")
            return []
    
    def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            try:
                self.driver.close()
                self.connected = False
                self.logger.info("🔌 Neo4j connection closed successfully")
            except Exception as e:
                self.logger.error(f"❌ Error closing Neo4j connection: {type(e).__name__}: {str(e)}")