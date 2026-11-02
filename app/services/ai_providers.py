import abc
import hashlib
import math
import re
from typing import List, Optional
from app.core.config import get_settings

settings = get_settings()


class EmbeddingProvider(abc.ABC):
    """Abstract Base Class for Embedding Providers."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        pass

    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic Mock Embedding Provider.
    Generates deterministic, L2-normalized float vectors based on text tokens and hashing.
    Text with overlapping keywords yields high cosine similarity, enabling 100% reliable offline testing.
    """

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _generate_vector(self, text: str) -> List[float]:
        if not text:
            vec = [0.0] * self._dim
            vec[0] = 1.0
            return vec

        # Initialize dimension bins
        bins = [0.0] * self._dim

        # Split into normalized tokens
        words = re.findall(r"\w+", text.lower())
        if not words:
            words = [text.strip().lower()]

        stopwords = {
            "the", "and", "for", "are", "is", "was", "were", "with", "that", "this",
            "from", "have", "has", "had", "what", "when", "where", "which", "who",
            "whom", "how", "why", "can", "will", "would", "could", "should", "into",
            "your", "our", "their", "about", "does", "did", "many", "much", "days",
            "day", "year", "years", "month", "months", "provided", "provide", "provides",
            "per", "all", "any", "only", "must", "may", "not", "but", "under", "now"
        }

        for word in words:
            weight = 0.05 if word in stopwords else 1.0
            # Deterministic bucket from hash
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            primary_idx = h % self._dim
            secondary_idx = (h >> 8) % self._dim
            bins[primary_idx] += 1.0 * weight
            bins[secondary_idx] += 0.5 * weight

        # Baseline text hash to ensure distinct vectors for distinct text
        full_hash = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        for i in range(min(16, self._dim)):
            bit = (full_hash >> i) & 1
            bins[i] += 0.05 if bit else -0.05

        # L2-normalize
        norm = math.sqrt(sum(x * x for x in bins))
        if norm > 0:
            return [x / norm for x in bins]
        return [1.0 / math.sqrt(self._dim)] * self._dim

    def embed_query(self, text: str) -> List[float]:
        return self._generate_vector(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]


class LocalSentenceTransformerEmbedding(EmbeddingProvider):
    """Local SentenceTransformer embedding provider (all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._dim = 384
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self._dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load local SentenceTransformer model '{model_name}'. "
                f"Install sentence-transformers or set USE_MOCK_AI=true. Error: {e}"
            )

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI Embedding Provider (e.g., text-embedding-3-small)."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self._dim = 1536
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(input=[text], model=self.model_name)
        return resp.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(input=texts, model=self.model_name)
        return [d.embedding for d in resp.data]


# =====================================================================
# LLM Providers
# =====================================================================

class LLMProvider(abc.ABC):
    """Abstract Base Class for LLM Providers."""

    @abc.abstractmethod
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass


class MockLLMProvider(LLMProvider):
    """
    Deterministic Mock LLM Provider for testing.
    Adheres strictly to the prompt fence instructions:
    - Answers based solely on provided context.
    - If context is empty or irrelevant, returns standard insufficient evidence phrasing.
    - Ignores prompt injection attacks inside <untrusted_document_context>.
    """

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Extract untrusted document context
        context_match = re.search(r"<untrusted_document_context>(.*?)</untrusted_document_context>", prompt, re.DOTALL)
        context = context_match.group(1).strip() if context_match else ""

        # Extract user query/question
        user_query_match = re.search(r"Question:\s*(.*)", prompt, re.IGNORECASE)
        question = user_query_match.group(1).strip() if user_query_match else prompt.strip()

        # If no context is provided or context is explicitly empty
        if not context:
            return "I do not have enough information in the uploaded documents to answer this question."

        # Defense against prompt injection within context
        # Extract meaningful lines from context while stripping headers and potential injections
        cleaned_context_lines = []
        for line in context.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("[Source") or line_str.startswith("[Document"):
                continue
            # Filter obvious adversarial directives
            if any(inj in line_str.lower() for inj in ["ignore previous", "system prompt", "you are now", "act as"]):
                continue
            cleaned_context_lines.append(line_str)

        if not cleaned_context_lines:
            return "I do not have enough information in the uploaded documents to answer this question."

        cleaned_text = " ".join(cleaned_context_lines)

        # Boilerplate / generic stopwords to isolate core topical entities
        generic_terms = {
            "what", "when", "where", "which", "who", "whom", "how", "why", "this", "that",
            "these", "those", "there", "here", "from", "with", "about", "have", "has", "had",
            "been", "will", "would", "could", "should", "must", "may", "can", "does", "did",
            "are", "was", "were", "is", "be", "do", "not", "only", "all", "any", "some",
            "many", "much", "more", "most", "other", "such", "our", "your", "their", "its",
            "day", "days", "year", "years", "month", "months", "time", "times", "now", "current",
            "provide", "provided", "provides", "providing", "give", "given", "allow", "allowed",
            "require", "required", "requires", "explain", "describe", "tell", "referring",
            "corporate", "company", "policy", "policies", "rule", "rules", "handbook", "guidelines",
            "guideline", "protocol", "protocols", "document", "documents", "information", "details",
            "standard", "standards", "office", "system", "service", "services", "acme", "general",
            "under", "the", "and", "for", "per", "into", "onto", "across", "during", "after",
            "before", "user", "enterprise"
        }
        all_q_words = [w for w in re.findall(r"\w+", question.lower()) if len(w) > 2]
        core_q_words = [w for w in all_q_words if w not in generic_terms]

        def _term_matches_text(term: str, target_text: str, target_tokens: set) -> bool:
            if term in target_tokens:
                return True
            if len(term) >= 4 and term in target_text:
                return True
            if len(term) >= 6 and any(tok.startswith(term[:5]) for tok in target_tokens):
                return True
            return False

        cleaned_norm = cleaned_text.lower().replace("roll over", "rollover").replace("carry over", "rollover")
        cleaned_tokens = set(re.findall(r"\w+", cleaned_norm))

        # If question has specific topical terms and NONE appear in the context, evidence is insufficient
        if core_q_words:
            core_matches = [w for w in core_q_words if _term_matches_text(w, cleaned_norm, cleaned_tokens)]
            if not core_matches:
                return "I do not have enough information in the uploaded documents to answer this question."

        q_words = core_q_words if core_q_words else all_q_words

        # Extract and rank the most relevant sentences from cleaned context by keyword overlap
        scored_sentences = []
        seen_sentences = set()
        current_section_bonus = 0

        for l in cleaned_context_lines:
            l_strip = l.strip()
            # Detect short section headers (lines without sentence-ending punctuation)
            if len(l_strip) < 65 and not l_strip.endswith("."):
                l_norm = l_strip.lower().replace("roll over", "rollover").replace("carry over", "rollover")
                l_tokens = set(re.findall(r"\w+", l_norm))
                current_section_bonus = sum(1 for w in q_words if _term_matches_text(w, l_norm, l_tokens))
                continue

            for s in re.split(r"\. |\.\n", l_strip):
                s_clean = s.strip().rstrip(".")
                if not s_clean or len(s_clean) < 15 or s_clean in seen_sentences:
                    continue
                s_lower = s_clean.lower()
                # Normalize compound synonyms like "roll over" / "carry over" -> "rollover"
                s_norm = s_lower.replace("roll over", "rollover").replace("carry over", "rollover")
                s_tokens = set(re.findall(r"\w+", s_norm))
                direct_matches = sum(1 for w in q_words if _term_matches_text(w, s_norm, s_tokens))
                total_score = direct_matches + (current_section_bonus if direct_matches > 0 or current_section_bonus > 0 else 0)
                if total_score > 0:
                    seen_sentences.add(s_clean)
                    scored_sentences.append((total_score, direct_matches, -len(scored_sentences), s_clean))

        if scored_sentences:
            scored_sentences.sort(key=lambda item: (-item[0], -item[1], -item[2]))
            top_sentences = [item[3] for item in scored_sentences[:3]]
            return ". ".join(top_sentences) + "."
        return "I do not have enough information in the uploaded documents to answer this question."


class GroqLLMProvider(LLMProvider):
    """Groq LLM Provider."""

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq client: {e}")

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
        )
        return completion.choices[0].message.content or ""


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM Provider."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
        )
        return completion.choices[0].message.content or ""


class GeminiLLMProvider(LLMProvider):
    """Google Gemini LLM Provider."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {e}")

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}" if system_prompt else prompt
        response = self.model.generate_content(full_prompt)
        return response.text or ""


# =====================================================================
# Factory Functions
# =====================================================================

def get_embedding_provider() -> EmbeddingProvider:
    current_settings = get_settings()
    if current_settings.USE_MOCK_AI or current_settings.EMBEDDING_PROVIDER == "mock":
        return MockEmbeddingProvider(dimension=current_settings.VECTOR_DIM)

    if current_settings.EMBEDDING_PROVIDER == "openai" and current_settings.OPENAI_API_KEY:
        return OpenAIEmbeddingProvider(
            api_key=current_settings.OPENAI_API_KEY,
            model_name=current_settings.OPENAI_EMBEDDING_MODEL
        )

    if current_settings.EMBEDDING_PROVIDER == "local":
        try:
            return LocalSentenceTransformerEmbedding(model_name=current_settings.LOCAL_EMBEDDING_MODEL)
        except Exception:
            # Fallback to mock provider if sentence-transformers not installed
            return MockEmbeddingProvider(dimension=current_settings.VECTOR_DIM)

    return MockEmbeddingProvider(dimension=current_settings.VECTOR_DIM)


def get_llm_provider() -> LLMProvider:
    current_settings = get_settings()
    if current_settings.USE_MOCK_AI or current_settings.LLM_PROVIDER == "mock":
        return MockLLMProvider()

    if current_settings.LLM_PROVIDER == "groq" and current_settings.GROQ_API_KEY:
        return GroqLLMProvider(api_key=current_settings.GROQ_API_KEY, model_name=current_settings.GROQ_MODEL)

    if current_settings.LLM_PROVIDER == "openai" and current_settings.OPENAI_API_KEY:
        return OpenAILLMProvider(api_key=current_settings.OPENAI_API_KEY, model_name=current_settings.OPENAI_MODEL)

    if current_settings.LLM_PROVIDER == "gemini" and current_settings.GEMINI_API_KEY:
        return GeminiLLMProvider(api_key=current_settings.GEMINI_API_KEY, model_name=current_settings.GEMINI_MODEL)

    return MockLLMProvider()
