"""Knowledge Graph RAG system using graphiti and Neo4j.

Ingest text documents into Neo4j as a knowledge graph using graphiti's add_episode() method.

Behavior:
- Chunk a text document (default 500 token chars, 100 overlap)
- Use graphiti.add_episode() to add chunks and extract entities automatically
- Query flow: extract entities from question, find chunks mentioning those entities,
  and call OpenAI LLM for an answer using the retrieved contexts.

Environment variables:
- OPENAI_API_KEY
- NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

import os
import json
import argparse
import logging
from typing import List, Tuple, Optional

try:
	from dotenv import load_dotenv, dotenv_values
	DOTENV_AVAILABLE = True
except Exception:
	DOTENV_AVAILABLE = False

try:
	import openai
except Exception:
	openai = None

try:
	from neo4j import GraphDatabase
except Exception:
	GraphDatabase = None

try:
	import graphiti
	GRAPHITI_AVAILABLE = True
except Exception:
	graphiti = None
	GRAPHITI_AVAILABLE = False


LOG_LEVEL = os.getenv("KG_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _short_repr(obj, maxlen=200):
	try:
		if isinstance(obj, str):
			return obj if len(obj) <= maxlen else obj[:maxlen] + "..."
		if isinstance(obj, (list, tuple)):
			return f"{type(obj).__name__}(len={len(obj)})"
		return repr(obj)
	except Exception:
		return "<unrepresentable>"


def log_calls(func):
	def wrapper(*args, **kwargs):
		try:
			logger.info("ENTER %s args=%s kwargs=%s", func.__name__, _short_repr(args), _short_repr(kwargs))
			res = func(*args, **kwargs)
			logger.info("EXIT %s -> %s", func.__name__, _short_repr(res))
			return res
		except Exception as e:
			logger.exception("EXCEPTION in %s: %s", func.__name__, e)
			raise
	wrapper.__name__ = func.__name__
	wrapper.__doc__ = func.__doc__
	return wrapper


@staticmethod
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
	"""Chunk text into overlapping pieces."""
	if chunk_size <= 0:
		return [text]
	chunks = []
	start = 0
	length = len(text)
	while start < length:
		end = min(start + chunk_size, length)
		chunk = text[start:end]
		chunks.append(chunk.strip())
		if end == length:
			break
		start = end - overlap
	return [c for c in chunks if c]


class KnowledgeGraphIndexer:
	"""Knowledge Graph RAG system using graphiti and Neo4j."""

	def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, 
				 neo4j_password: str = None, openai_api_key: str = None,
				 llm_model: str = "gpt-3.5-turbo", env_file: str = None):
		"""Initialize the Knowledge Graph RAG system.
		
		Args:
			neo4j_uri: Neo4j connection URI
			neo4j_user: Neo4j username
			neo4j_password: Neo4j password
			openai_api_key: OpenAI API key
			llm_model: LLM model to use
			env_file: Path to .env file for loading environment variables
		"""
		# Load environment variables
		self.env_vars = self._load_env_file(env_file)
		
		self.neo4j_uri = neo4j_uri or self.env_vars.get("NEO4J_URI") or os.getenv("NEO4J_URI")
		self.neo4j_user = neo4j_user or self.env_vars.get("NEO4J_USER") or os.getenv("NEO4J_USER")
		self.neo4j_password = neo4j_password or self.env_vars.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD")
		self.openai_api_key = openai_api_key or self.env_vars.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
		self.llm_model = llm_model or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
		
		self.driver = None
		self._validate_dependencies()
		self._connect_neo4j()

	def _load_env_file(self, path: str = None) -> dict:
		"""Load environment variables from .env file."""
		if not DOTENV_AVAILABLE:
			return {}
		candidates = [] if path else [os.path.expanduser("~/.env"), ".env"]
		if path:
			candidates = [path]
		for p in candidates:
			if p and os.path.exists(p):
				vals = dotenv_values(p)
				load_dotenv(p)
				return {k: v for k, v in vals.items() if k is not None}
		return {}

	def _validate_dependencies(self):
		"""Validate required dependencies."""
		if not GRAPHITI_AVAILABLE:
			raise RuntimeError("graphiti package is required (pip install graphiti)")
		if GraphDatabase is None:
			raise RuntimeError("neo4j package is required (pip install neo4j)")
		if openai is None:
			raise RuntimeError("openai package is required (pip install openai)")

	def _connect_neo4j(self):
		"""Connect to Neo4j database."""
		if not self.neo4j_uri or not self.neo4j_user or not self.neo4j_password:
			raise RuntimeError(
				"Neo4j credentials missing. Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD "
				"environment variables or pass them to __init__"
			)
		self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
		logger.info("Connected to Neo4j at %s", self.neo4j_uri)


	def index_document(self, path: str, chunk_size: int = 500, overlap: int = 100):
		"""Index a document by chunking and adding episodes to the knowledge graph.
		
		Args:
			path: Path to the text file to index
			chunk_size: Size of each chunk in characters
			overlap: Overlap between chunks in characters
		"""
		with open(path, "r", encoding="utf-8") as f:
			text = f.read()
		chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
		logger.info("Created %d chunks from %s", len(chunks), path)

		doc_name = os.path.basename(path)
		
		# Use graphiti.add_episode() to add each chunk
		try:
			for i, chunk in enumerate(chunks):
				episode_id = f"{doc_name}::chunk::{i}"
				if hasattr(graphiti, "add_episode"):
					graphiti.add_episode(
						episode_id=episode_id,
						text=chunk,
						metadata={"pos": i, "doc": doc_name, "path": os.path.abspath(path)}
					)
					logger.info("Added episode %d/%d for document %s", i + 1, len(chunks), doc_name)
				else:
					raise RuntimeError(
						"graphiti.add_episode() not found. Available methods: " +
						str([m for m in dir(graphiti) if not m.startswith("_")])
					)
		except Exception as ex:
			logger.exception("Failed to add episodes: %s", ex)
			raise

		logger.info("Indexing complete: document %s stored using graphiti", doc_name)

	def query_index(self, question: str, k: int = 4) -> List[Tuple[str, float]]:
		"""Retrieve top-k chunks by matching entities extracted from the question.

		This avoids embeddings and uses entity matching instead. It:
		- Extracts entities from the question using graphiti
		- Finds Chunk nodes that mention those entities
		- Ranks by how many matched entities they mention
		- Returns a list of (chunk_text, score)
		"""
		extracted = []
		if GRAPHITI_AVAILABLE:
			try:
				if hasattr(graphiti, "extract_entities"):
					ents = graphiti.extract_entities(question)
				elif hasattr(graphiti, "extract"):
					ents = graphiti.extract(question)
				elif hasattr(graphiti, "run"):
					ents = graphiti.run(question)
				else:
					ents = None
				if ents:
					for e in ents:
						if isinstance(e, dict):
							name = e.get("text") or e.get("name") or e.get("entity")
							if name:
								extracted.append(name.strip())
						elif isinstance(e, (list, tuple)) and len(e) >= 1:
							extracted.append(str(e[0]).strip())
						else:
							v = str(e).strip()
							if v:
								extracted.append(v)
			except Exception as ex:
				logger.warning("Graphiti entity extraction failed: %s", ex)
				extracted = []
		else:
			raise RuntimeError("graphiti package is required (pip install graphiti)")

		results: List[Tuple[str, float]] = []
		if not self.driver:
			raise RuntimeError("Database connection not initialized")

		with self.driver.session() as session:
			if extracted:
				cypher = (
					"WITH $names AS names "
					"UNWIND names AS n "
					"MATCH (e:Entity) WHERE toLower(e.name) = toLower(n) "
					"MATCH (e)<-[:MENTIONS]-(c:Chunk) "
					"RETURN c.id AS id, c.text AS text, collect(DISTINCT e.name) AS matched, count(DISTINCT e) AS score "
					"ORDER BY score DESC LIMIT $k"
				)
				res = session.run(cypher, names=extracted, k=k)
				for r in res:
					results.append((r.get("text"), float(r.get("score") or 0)))
			else:
				# Fallback: simple keyword search on chunk text
				qlower = question.lower()
				cypher = (
					"MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS $q "
					"RETURN c.id AS id, c.text AS text LIMIT $k"
				)
				res = session.run(cypher, q=qlower, k=k)
				for i, r in enumerate(res):
					results.append((r.get("text"), float(k - i)))

		return results[:k]

	def answer_with_llm(self, question: str, contexts: List[str]) -> str:
		"""Generate an answer using OpenAI LLM given question and context chunks.
		
		Args:
			question: The user's question
			contexts: List of context chunks to use for answering
			
		Returns:
			The LLM-generated answer
		"""
		if openai is None:
			raise RuntimeError("openai package is required (pip install openai)")
		system = (
			"You are a helpful assistant. Use the provided context to answer the user's question. "
			"If the answer is not contained in the context, say you do not know the answer. Be concise."
		)
		context_text = "\n\n---\n\n".join(contexts)
		prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer using only the context above."
		messages = [
			{"role": "system", "content": system},
			{"role": "user", "content": prompt},
		]
		resp = openai.chat.completions.create(model=self.llm_model, messages=messages, temperature=0.0)
		return resp.choices[0].message.content.strip()

	def close(self):
		"""Close the database connection."""
		if self.driver:
			self.driver.close()
			logger.info("Database connection closed")



def main():
	"""CLI entry point for the Knowledge Graph RAG system."""
	parser = argparse.ArgumentParser(description="Index documents using Knowledge Graph RAG")
	parser.add_argument("--doc", help="Path to text document to index")
	parser.add_argument("--ask", help="Ask a single question against the saved graph")
	parser.add_argument("--chunk_size", type=int, default=500)
	parser.add_argument("--overlap", type=int, default=100)
	parser.add_argument("--k", type=int, default=4, help="Number of retrieved chunks")
	parser.add_argument("--show-env", action="store_true", help="Show variables loaded from .env")
	parser.add_argument("--env-file", help="Path to .env file")
	args = parser.parse_args()

	try:
		# Initialize the Knowledge Graph RAG system
		kg = KnowledgeGraphIndexer(env_file=args.env_file)

		if args.show_env:
			print("Loaded environment variables:")
			for k, v in kg.env_vars.items():
				if v:
					print(f"{k}={v[:20]}..." if len(str(v)) > 20 else f"{k}={v}")

		if args.doc:
			kg.index_document(args.doc, chunk_size=args.chunk_size, overlap=args.overlap)

		if args.ask:
			results = kg.query_index(args.ask, k=args.k)
			contexts = [t for t, s in results]
			answer = kg.answer_with_llm(args.ask, contexts)
			print("\nAnswer:")
			print(answer)
			print("\nRetrieved contexts:")
			for i, (ctx, score) in enumerate(results, 1):
				preview = ctx[:400].replace("\n", " ")
				suffix = "..." if len(ctx) > 400 else ""
				print(f"[{i}] (score: {score:.2f}) {preview}{suffix}\n")

		if not args.doc and not args.ask:
			print("Provide --doc to index a document or --ask to query the graph.")

		kg.close()

	except Exception as e:
		logger.error("Error: %s", e)
		print(f"Error: {e}")
		return 1

	return 0


if __name__ == "__main__":
	main()

