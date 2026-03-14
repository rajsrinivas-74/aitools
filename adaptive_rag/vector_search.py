"""Vector Search RAG system using OpenAI embeddings and FAISS.

Ingest text documents into a FAISS vector database with OpenAI embeddings.

Behavior:
- Chunk a text document (default 500 character chunks with 100 character overlap)
- Generate embeddings for each chunk using OpenAI's embedding model
- Store embeddings and chunk metadata in FAISS index
- Query flow: embed the question, find similar chunks using FAISS,
  and call OpenAI LLM for an answer using the retrieved contexts.

Environment variables:
- OPENAI_API_KEY
"""

import os
import json
import argparse
import logging
import pickle
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
	import faiss
	import numpy as np
	FAISS_AVAILABLE = True
except Exception:
	faiss = None
	np = None
	FAISS_AVAILABLE = False


LOG_LEVEL = os.getenv("VS_LOG_LEVEL", "INFO").upper()
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


class VectorSearchIndexer:
	"""Vector Search RAG system using OpenAI embeddings and FAISS."""

	def __init__(self, openai_api_key: str = None, embedding_model: str = "text-embedding-3-small",
				 llm_model: str = "gpt-3.5-turbo", env_file: str = None, index_path: str = None):
		"""Initialize the Vector Search RAG system.
		
		Args:
			openai_api_key: OpenAI API key
			embedding_model: OpenAI embedding model to use
			llm_model: LLM model to use for answering questions
			env_file: Path to .env file for loading environment variables
			index_path: Path to save/load FAISS index and metadata
		"""
		# Load environment variables
		self.env_vars = self._load_env_file(env_file)
		
		self.openai_api_key = openai_api_key or self.env_vars.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
		self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
		self.llm_model = llm_model or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
		self.index_path = index_path or os.getenv("FAISS_INDEX_PATH", "faiss_index")
		
		self.faiss_index = None
		self.chunk_metadata = []  # List of (chunk_text, doc_name, chunk_id)
		
		self._validate_dependencies()
		if os.path.exists(f"{self.index_path}.index"):
			self._load_index()
		else:
			logger.info("No existing index found at %s", self.index_path)

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
		if not FAISS_AVAILABLE:
			raise RuntimeError("faiss-cpu package is required (pip install faiss-cpu)")
		if openai is None:
			raise RuntimeError("openai package is required (pip install openai)")

	@log_calls
	def _get_embedding(self, text: str) -> List[float]:
		"""Get embedding for a text using OpenAI API.
		
		Args:
			text: Text to embed
			
		Returns:
			Embedding vector
		"""
		if openai is None:
			raise RuntimeError("openai package is required (pip install openai)")
		
		try:
			response = openai.Embedding.create(
				input=text,
				model=self.embedding_model
			)
			return response['data'][0]['embedding']
		except AttributeError:
			# Newer OpenAI client API
			response = openai.embeddings.create(
				input=text,
				model=self.embedding_model
			)
			return response.data[0].embedding

	@log_calls
	def index_document(self, path: str, chunk_size: int = 500, overlap: int = 100):
		"""Index a document by chunking and storing embeddings in FAISS.
		
		Args:
			path: Path to the text file to index
			chunk_size: Size of each chunk in characters
			overlap: Overlap between chunks in characters
		"""
		if not os.path.exists(path):
			raise FileNotFoundError(f"Document not found: {path}")
			
		with open(path, "r", encoding="utf-8") as f:
			text = f.read()
		
		chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
		logger.info("Created %d chunks from %s", len(chunks), path)

		doc_name = os.path.basename(path)
		embeddings = []
		
		# Generate embeddings for all chunks
		for i, chunk in enumerate(chunks):
			try:
				embedding = self._get_embedding(chunk)
				embeddings.append(embedding)
				chunk_id = f"{doc_name}::chunk::{i}"
				self.chunk_metadata.append((chunk, doc_name, chunk_id))
				logger.info("Embedded chunk %d/%d for document %s", i + 1, len(chunks), doc_name)
			except Exception as ex:
				logger.exception("Failed to embed chunk %d: %s", i, ex)
				raise

		# Create or update FAISS index
		if embeddings:
			embeddings_array = np.array(embeddings, dtype=np.float32)
			
			if self.faiss_index is None:
				# Create new index
				dimension = embeddings_array.shape[1]
				self.faiss_index = faiss.IndexFlatL2(dimension)
				logger.info("Created new FAISS index with dimension %d", dimension)
			
			self.faiss_index.add(embeddings_array)
			logger.info("Added %d embeddings to FAISS index", len(embeddings))

		logger.info("Indexing complete: document %s stored in FAISS", doc_name)

	@log_calls
	def query_index(self, question: str, k: int = 4) -> List[Tuple[str, float]]:
		"""Retrieve top-k similar chunks using FAISS vector search.

		Args:
			question: The user's question
			k: Number of top results to return
			
		Returns:
			List of (chunk_text, similarity_score) tuples
		"""
		if self.faiss_index is None:
			raise RuntimeError("No index available. Index a document first using index_document()")
		
		if not self.chunk_metadata:
			raise RuntimeError("No chunk metadata available")

		try:
			# Get embedding for the question
			question_embedding = self._get_embedding(question)
			question_vector = np.array([question_embedding], dtype=np.float32)
			
			# Search FAISS index
			distances, indices = self.faiss_index.search(question_vector, min(k, len(self.chunk_metadata)))
			
			results = []
			for i, idx in enumerate(indices[0]):
				if idx >= 0 and idx < len(self.chunk_metadata):
					chunk_text, doc_name, chunk_id = self.chunk_metadata[idx]
					# Convert L2 distance to a similarity score (lower distance = higher similarity)
					similarity = 1 / (1 + distances[0][i])
					results.append((chunk_text, float(similarity)))
			
			return results
		
		except Exception as ex:
			logger.exception("Query failed: %s", ex)
			raise

	@log_calls
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
		
		try:
			resp = openai.chat.completions.create(model=self.llm_model, messages=messages, temperature=0.0)
			return resp.choices[0].message.content.strip()
		except AttributeError:
			# Older OpenAI client API
			resp = openai.ChatCompletion.create(model=self.llm_model, messages=messages, temperature=0.0)
			return resp['choices'][0]['message']['content'].strip()

	def save_index(self, path: str = None):
		"""Save FAISS index and metadata to disk.
		
		Args:
			path: Path to save index (default: self.index_path)
		"""
		if self.faiss_index is None:
			logger.warning("No index to save")
			return
		
		save_path = path or self.index_path
		
		# Save FAISS index
		faiss.write_index(self.faiss_index, f"{save_path}.index")
		
		# Save metadata
		with open(f"{save_path}.metadata", "wb") as f:
			pickle.dump(self.chunk_metadata, f)
		
		logger.info("Index saved to %s", save_path)

	def _load_index(self):
		"""Load FAISS index and metadata from disk."""
		try:
			self.faiss_index = faiss.read_index(f"{self.index_path}.index")
			with open(f"{self.index_path}.metadata", "rb") as f:
				self.chunk_metadata = pickle.load(f)
			logger.info("Index loaded from %s", self.index_path)
		except Exception as ex:
			logger.warning("Failed to load index: %s", ex)
			self.faiss_index = None
			self.chunk_metadata = []


def main():
	"""CLI entry point for the Vector Search RAG system."""
	parser = argparse.ArgumentParser(description="Index documents using Vector Search RAG with FAISS")
	parser.add_argument("--doc", help="Path to text document to index")
	parser.add_argument("--ask", help="Ask a single question against the saved index")
	parser.add_argument("--chunk_size", type=int, default=500)
	parser.add_argument("--overlap", type=int, default=100)
	parser.add_argument("--k", type=int, default=4, help="Number of retrieved chunks")
	parser.add_argument("--index-path", default="faiss_index", help="Path to save/load FAISS index")
	parser.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model")
	parser.add_argument("--llm-model", default="gpt-3.5-turbo", help="OpenAI LLM model")
	parser.add_argument("--show-env", action="store_true", help="Show variables loaded from .env")
	parser.add_argument("--env-file", help="Path to .env file")
	args = parser.parse_args()

	try:
		# Initialize the Vector Search RAG system
		vs = VectorSearchIndexer(
			embedding_model=args.embedding_model,
			llm_model=args.llm_model,
			env_file=args.env_file,
			index_path=args.index_path
		)

		if args.show_env:
			print("Loaded environment variables:")
			for k, v in vs.env_vars.items():
				if v:
					print(f"{k}={v[:20]}..." if len(str(v)) > 20 else f"{k}={v}")

		if args.doc:
			vs.index_document(args.doc, chunk_size=args.chunk_size, overlap=args.overlap)
			vs.save_index()

		if args.ask:
			results = vs.query_index(args.ask, k=args.k)
			contexts = [t for t, s in results]
			answer = vs.answer_with_llm(args.ask, contexts)
			print("\nAnswer:")
			print(answer)
			print("\nRetrieved contexts:")
			for i, (ctx, score) in enumerate(results, 1):
				preview = ctx[:400].replace("\n", " ")
				suffix = "..." if len(ctx) > 400 else ""
				print(f"[{i}] (similarity: {score:.4f}) {preview}{suffix}\n")

		if not args.doc and not args.ask:
			print("Provide --doc to index a document or --ask to query the index.")

	except Exception as e:
		logger.error("Error: %s", e)
		print(f"Error: {e}")
		return 1

	return 0


if __name__ == "__main__":
	main()
