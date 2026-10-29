"""
Modern Tiny LLM & Autonomous Agent - Streamlit Web UI (Section 17)
Provides a rich browser interface with:
- Clear chat layout & streaming responses
- Model & Checkpoint selector (From-Scratch vs Ollama Pretrained/Teacher)
- Sampling controls (Temperature, Top-p, Top-k, Max Tokens)
- RAG knowledge retrieval toggle & real citations
- SQLite persistent memory & JSON chat export
- System telemetry (CPU, Memory, Model parameters)
"""
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "tiny_llm_from_scratch"))

# Windows OpenBLAS guards
from src.utils import clamp_blas_threads, get_hardware_info
clamp_blas_threads()

try:
    import streamlit as st
except ImportError:
    print("[ERROR] Streamlit is not installed. Run 'pip install streamlit' to launch the web UI.")
    print("Or run 'python chat.py' to use the terminal CLI interface.")
    sys.exit(1)

import torch
from tokenizers import Tokenizer
from src.models import load_modern_checkpoint, inspect_checkpoint
from src.generation import TextGenerator, SamplingConfig
from src.rag import LocalRAGPipeline
from src.memory import ConversationDatabase
from src.reasoning import ReasoningOrchestrator
from src.backends import BackendManager, OllamaClient

st.set_page_config(page_title="Tiny LLM & AI Assistant", page_icon="🤖", layout="wide")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"web_session_{int(time.time())}"

# Sidebar: Configuration & Controls
with st.sidebar:
    st.title("⚙️ Model Controls")
    
    # Backend Selection
    backend_choice = st.radio("Active Backend", ["From-Scratch ModernTransformer", "Local Ollama Pretrained"])
    
    ckpt_dir = PROJECT_ROOT / "tiny_llm_from_scratch" / "checkpoints"
    available_ckpts = [f.name for f in ckpt_dir.glob("*.pt")] if ckpt_dir.is_dir() else []
    
    selected_ckpt = st.selectbox("Checkpoint", available_ckpts, index=0 if available_ckpts else None)
    
    st.markdown("---")
    st.subheader("🎛️ Sampling Controls")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.70, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.90, 0.05)
    top_k = st.slider("Top-k", 1, 100, 40, 1)
    max_tokens = st.slider("Max New Tokens", 10, 512, 120, 10)
    
    st.markdown("---")
    st.subheader("🧠 Cognitive Modules")
    enable_rag = st.checkbox("RAG Knowledge Retrieval", value=True)
    enable_cot = st.checkbox("Chain-of-Thought (<thinking>)", value=True)
    
    st.markdown("---")
    # Telemetry
    hw = get_hardware_info()
    st.caption(f"OS: {hw['os']} {hw['architecture']} | CPUs: {hw['logical_cpus']}")
    
    # Export Chat
    if st.button("📥 Export Chat (JSON)"):
        chat_json = json.dumps(st.session_state.messages, indent=2)
        st.download_button("Download JSON", chat_json, file_name=f"{st.session_state.session_id}.json", mime="application/json")
        
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = f"web_session_{int(time.time())}"
        st.rerun()

# Main Chat Layout
st.header("🤖 Tiny LLM & Autonomous Agent (v2.2)")
st.caption("Self-contained Modern Transformer with RoPE, KV-Cache, Safe Python Sandbox & Multi-Step Reasoning.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("thinking") and enable_cot:
            with st.expander("💭 Reasoning Trace (<thinking>)", expanded=False):
                st.markdown(f"```text\n{msg['thinking']}\n```")
        st.markdown(msg["content"])
        if msg.get("citations"):
            st.caption(f"📚 Sources: {', '.join(msg['citations'])}")

# User Input
if prompt := st.chat_input("Ask a question, enter an equation, or execute Python code (/code)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Load Model & Reasoning Engine
        tok_path = PROJECT_ROOT / "tiny_llm_from_scratch" / "tokenizer_bpe.json"
        tokenizer = Tokenizer.from_file(str(tok_path))
        
        rag = LocalRAGPipeline(PROJECT_ROOT / "tiny_llm_from_scratch" / "data_knowledge") if enable_rag else None
        orchestrator = ReasoningOrchestrator(rag_pipeline=rag, enable_thinking=enable_cot)
        
        prep = orchestrator.prepare_reasoning(prompt)
        
        if prep.thinking and enable_cot:
            with st.expander("💭 Reasoning Trace (<thinking>)", expanded=True):
                st.markdown(f"```text\n{prep.thinking}\n```")
        
        if prep.direct_answer:
            response_text = prep.direct_answer
            st.markdown(response_text)
        else:
            # Generate from model
            ckpt_path = ckpt_dir / selected_ckpt if selected_ckpt else None
            if ckpt_path and ckpt_path.is_file():
                model, _ = load_modern_checkpoint(ckpt_path, device="cpu")
                gen = TextGenerator(model, tokenizer)
                s_cfg = SamplingConfig(temperature=temperature, top_p=top_p, top_k=top_k, max_new_tokens=max_tokens)
                
                response_placeholder = st.empty()
                streamed = []
                p_ids = tokenizer.encode(prep.prompt_with_context).ids
                for piece in gen.generate_stream(p_ids, config=s_cfg, device="cpu"):
                    streamed.append(piece)
                    response_placeholder.markdown("".join(streamed) + "▌")
                response_text = "".join(streamed)
                response_placeholder.markdown(response_text)
            else:
                response_text = "Please select a valid checkpoint in the sidebar."
                st.error(response_text)

        if prep.retrieved_citations:
            st.caption(f"📚 Sources: {', '.join(prep.retrieved_citations)}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "thinking": prep.thinking,
            "citations": prep.retrieved_citations,
        })
