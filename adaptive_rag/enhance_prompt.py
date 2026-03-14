"""
Prompt Enhancement Module

Provides utilities for analyzing, improving, and scoring prompts using LLM.
Uses centralized LLM initialization from app_config.
"""

import re
from typing import Tuple
import streamlit as st

from app_config import get_config


class PromptEnhancer:
    def __init__(self, llm=None):
        """
        Initialize PromptEnhancer with optional injected LLM.

        Args:
            llm: Optional pre-initialized LLM instance. If None, gets from app_config.
        """
        config = get_config()
        self.llm = llm or config.get_llm()
        self._MODEL_CACHE = {}

    def init_chat_model(self, model, temperature: float = 0.0, **kwargs):
        """Return the cached LLM instance or create if not available."""
        try:
            key = (model, float(temperature), tuple(sorted(kwargs.items())))
        except Exception:
            key = (model, float(temperature))
        
        if key in self._MODEL_CACHE:
            return self._MODEL_CACHE[key]
        
        # Use injected LLM or get from config
        if self.llm:
            self._MODEL_CACHE[key] = self.llm
            return self.llm
        
        # Fallback: create new instance (shouldn't happen with proper initialization)
        try:
            from langchain.chat_models import ChatOpenAI
            inst = ChatOpenAI(model_name=model, temperature=temperature, **kwargs)
            self._MODEL_CACHE[key] = inst
            return inst
        except Exception as e:
            raise RuntimeError(f"LangChain ChatOpenAI not available: {e}")

    def langchain_chat(self, user_prompt: str, system: str | None = None, model: str = "gpt-3.5-turbo") -> str:
        prompt = (system + "\n\n" + user_prompt) if system else user_prompt
        try:
            llm = self.init_chat_model(model, temperature=0.2)
        except TypeError:
            llm = self.init_chat_model(model)
        try:
            if hasattr(llm, "invoke"):
                resp = llm.invoke(prompt)
            else:
                resp = llm(prompt)
            return getattr(resp, "content", str(resp))
        except Exception as e:
            raise RuntimeError(f"LangChain init_chat_model call failed: {e}")

    def reflect_prompt_openai(self, prompt: str) -> str:
        system = "You are an expert prompt engineer. Inspect the user's prompt and list weaknesses and 3 clarifying questions."
        user = (
            "Analyze the following prompt for weaknesses (clarity, missing context, missing output format, missing constraints). "
            "Return a short numbered list of issues and then 3 clarifying questions.\n\n" + f"Prompt:\n{prompt}"
        )
        try:
            return self.langchain_chat(user, system=system)
        except Exception as e:
            return f"(reflection failed: {e})"

    def improve_prompt_openai(self, prompt: str, reflection: str) -> str:
        system = "You are an expert prompt writer. Rewrite the user's prompt to be clear, explicit about format, constraints, examples and tone when appropriate."
        user = (
            "Using the original prompt and the reflection feedback, produce a single improved prompt the user can paste back into an LLM. "
            "Be concise but explicit about output format and any constraints.\n\n" + f"Original Prompt:\n{prompt}\n\nReflection:\n{reflection}\n\nImproved Prompt:\n"
        )
        try:
            return self.langchain_chat(user, system=system)
        except Exception as e:
            return f"(improve failed: {e})"

    def improve_prompt_openai_with_confidence(self, prompt: str, reflection: str, confidence_score: float) -> str:
        """
        Improve a prompt based on reflection and confidence score.
        Lower confidence scores trigger more aggressive improvements.

        Args:
            prompt: Original prompt
            reflection: Reflection feedback on the prompt
            confidence_score: Confidence score between 0 and 1 from query analysis

        Returns:
            Improved prompt
        """
        # Adjust improvement intensity based on confidence score
        if confidence_score < 0.3:
            intensity = "very aggressively"
            focus = "Add significant clarifications, constraints, and multiple examples."
        elif confidence_score < 0.5:
            intensity = "aggressively"
            focus = "Add clear constraints, examples, and expected output format."
        elif confidence_score < 0.7:
            intensity = "moderately"
            focus = "Enhance clarity and add missing context details."
        else:
            intensity = "slightly"
            focus = "Make minor refinements for better clarity."

        system = (
            "You are an expert prompt writer. Your task is to rewrite prompts to be clear, explicit, and actionable. "
            "Consider the confidence score and improvement intensity provided."
        )
        user = (
            f"Rewrite the following prompt {intensity} based on the reflection feedback. "
            f"Confidence score: {confidence_score:.2f} (0=very low, 1=very high). "
            f"Focus: {focus}\n\n"
            f"Original Prompt:\n{prompt}\n\n"
            f"Reflection:\n{reflection}\n\n"
            f"Improved Prompt:\n"
        )
        try:
            return self.langchain_chat(user, system=system)
        except Exception as e:
            return f"(improve with confidence failed: {e})"

    def score_prompt_openai(self, prompt: str) -> Tuple[int, str]:
        system = (
            "You are a precise evaluator. Rate prompts 1-10 and justify briefly. "
            "Score objectively based on these criteria: "
            "1. Context provided (is the background clear?) "
            "2. Specificity (are requirements and details explicit?) "
            "3. Constraints (are there clear rules or boundaries?) "
            "4. Output format (is the expected output format described?) "
            "5. Clarity (is the language unambiguous and direct?) "
            "For each, consider if the prompt is strong, average, or weak. "
            "Sum your assessment to produce an overall quality score between 1 (very poor) and 10 (excellent). "
            "Return the score on the first line as an integer, then a short justification."
        )
        user = (
            "Rate the following prompt on a scale of 1-10 (10 best) for how well it will produce a high-quality, unambiguous answer. "
            "Return the score on the first line as an integer, then a short justification.\n\n" + f"Prompt:\n{prompt}"
        )
        try:
            out = self.langchain_chat(user, system=system)
            m = re.search(r"\b(10|[1-9])\b", out)
            score = int(m.group(0)) if m else 5
            return score, out
        except Exception as e:
            return 5, f"(scoring failed: {e})"

    def optimize_prompt(self, prompt: str, max_iters: int = 3, target_score: int = 8) -> dict:
        history = []
        current = prompt
        for i in range(1, max_iters + 1):
            reflection = self.reflect_prompt_openai(current)
            improved = self.improve_prompt_openai(current, reflection)
            score, score_text = self.score_prompt_openai(improved)
            history.append({
                "iteration": i,
                "prompt": current,
                "reflection": reflection,
                "improved": improved,
                "score": score,
                "score_text": score_text,
            })
            if score >= target_score:
                break
            current = improved
        return {"final": history[-1] if history else None, "history": history}

    @staticmethod
    def run_streamlit(enhancer=None):
        """
        Run Streamlit UI with optional injected PromptEnhancer.

        Args:
            enhancer: Optional pre-initialized PromptEnhancer instance.
        """
        if enhancer is None:
            enhancer = PromptEnhancer()
        
        config = get_config()
        st.set_page_config(page_title="Prompt Optimizer (LangChain)", layout="centered")
        st.title("Prompt Optimizer — LangChain backend")
        st.write("Enter a plain-text prompt below. The app will reflect on weaknesses, request improvements via LangChain, score the improved prompt, and iterate if needed.")

        if enhancer.init_chat_model is None or not config.openai_api_key:
            st.error("LangChain chat model not configured. Install `langchain langchain-openai` and set the OPENAI_API_KEY environment variable.")

        if "prompt_input" not in st.session_state:
            st.session_state["prompt_input"] = ""
        if "processing" not in st.session_state:
            st.session_state["processing"] = False

        prompt = st.text_area("Prompt", height=220, placeholder="Type your prompt here...", key="prompt_input")
        cols = st.columns([1, 1, 1])
        max_iters = cols[0].number_input("Max iterations", min_value=1, max_value=5, value=3)
        target = cols[1].slider("Target score", 1, 10, 8)
        prompt_len = len(st.session_state.get("prompt_input", "").strip())
        run = cols[2].button(
            "Analyze & Improve",
            disabled=st.session_state.get("processing", False),
        )

        if run:
            current_prompt = st.session_state.get("prompt_input", "").strip()
            if len(current_prompt) < 30:
                st.warning("Please enter a prompt of at least 30 characters before submitting.")
            else:
                st.session_state["processing"] = True
                result = None
                try:
                    with st.spinner("Running reflection and improvement..."):
                        try:
                            init_score, init_text = enhancer.score_prompt_openai(current_prompt)
                        except Exception as e:
                            st.error(f"Failed to compute initial score: {e}")
                            init_score = None
                            init_text = None
                        if init_score is not None:
                            st.info(f"Initial score: {init_score}")
                            if init_text:
                                st.caption(init_text)
                        if init_score is not None and init_score < 3:
                            st.warning("Initial prompt scored below 3 — please update the prompt before proceeding.")
                            result = None
                        else:
                            result = enhancer.optimize_prompt(current_prompt, max_iters, target)
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                finally:
                    st.session_state["processing"] = False
                if result is None:
                    return
            history = result["history"]
            if not history:
                st.info("No iterations produced (empty prompt?).")
                return
            for step in history:
                st.markdown(f"---\n**Iteration {step['iteration']} — score: {step['score']}**")
                st.write("**Original Prompt**")
                st.code(step["prompt"])
                st.write("**Reflection**")
                st.info(step["reflection"])
                st.write("**Improved Prompt**")
                st.code(step["improved"])
                st.write("**Scoring Notes**")
                st.caption(step["score_text"])
            st.success("Optimization complete")
            final = result["final"]
            if final:
                st.header("Final Prompt")
                st.code(final["improved"])
                st.download_button("Download final prompt", final["improved"], file_name="improved_prompt.txt")
                st.session_state["prompt_input"] = final["improved"]

if __name__ == "__main__":
    PromptEnhancer.run_streamlit()
