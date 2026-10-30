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
from src.research import ResearchProfile, ProfileDefinition, PROFILES, ResearchModeManager

st.set_page_config(page_title="Tiny LLM & Advanced Research Platform", page_icon="🛡️", layout="wide")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"web_session_{int(time.time())}"
if "research_mgr" not in st.session_state:
    st.session_state.research_mgr = ResearchModeManager(ResearchProfile.ADVANCED_DEVELOPMENT)

# Sidebar: Configuration & Controls
with st.sidebar:
    st.title("🛡️ Research & Model Controls")

    # Advanced Research Execution Profile
    st.subheader("🔬 Execution Profile")
    profile_list = [p.value for p in ResearchProfile]
    current_prof_idx = profile_list.index(st.session_state.research_mgr.active_profile.value)
    chosen_profile_name = st.selectbox("Active Profile", profile_list, index=current_prof_idx)
    active_profile_defn = st.session_state.research_mgr.set_profile_by_name(chosen_profile_name)

    with st.expander("ℹ️ Profile Scope & Boundary", expanded=False):
        st.write(f"**Scope:** {active_profile_defn.scope}")
        st.write(f"**Safety Boundary:** {active_profile_defn.safety_boundary}")
        st.write(f"**Capabilities:** {', '.join(active_profile_defn.capabilities)}")
        st.write(f"**Allowed Tools:** {', '.join(active_profile_defn.allowed_tool_categories)}")

    st.markdown("---")
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
st.header("🛡️ Tiny LLM & Advanced Research Platform")
st.caption(f"Active Profile: **{active_profile_defn.title}** | Self-contained Modern Transformer with RoPE, KV-Cache & Cybersecurity Lab.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("provenance"):
            st.caption(f"🏷️ Source: **[{msg['provenance']}]**")
        if msg.get("thinking") and enable_cot:
            with st.expander("💭 Reasoning Trace (<thinking>)", expanded=False):
                st.markdown(f"```text\n{msg['thinking']}\n```")
        st.markdown(msg["content"])
        if msg.get("citations"):
            st.caption(f"📚 Sources: {', '.join(msg['citations'])}")

# User Input
if prompt := st.chat_input("Ask a technical question, analyze code, or run a security audit..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Load Model & Reasoning Engine
        tok_path = PROJECT_ROOT / "tokenizer_bpe_v2.json"
        if not tok_path.is_file():
            tok_path = PROJECT_ROOT / "tiny_llm_from_scratch" / "tokenizer_bpe.json"
        tokenizer = Tokenizer.from_file(str(tok_path))
        
        rag = LocalRAGPipeline(PROJECT_ROOT / "tiny_llm_from_scratch" / "data_knowledge") if enable_rag else None
        orchestrator = ReasoningOrchestrator(
            rag_pipeline=rag,
            enable_thinking=enable_cot,
            research_manager=st.session_state.research_mgr,
        )
        
        prep = orchestrator.prepare_reasoning(prompt)
        provenance = "Deterministic Tool"
        
        if prep.thinking and enable_cot:
            with st.expander("💭 Reasoning Trace (<thinking>)", expanded=True):
                st.markdown(f"```text\n{prep.thinking}\n```")
        
        if prompt.startswith("/browse"):
            url = prompt.split("/browse", 1)[1].strip()
            from src.tools.browser import WebBrowserEngine
            res = WebBrowserEngine.browse_url(url, max_chars=4000)
            response_text = WebBrowserEngine.format_browser_display(res)
            provenance = "Browser Engine"
            st.markdown(response_text)
        elif prompt.startswith("/learn"):
            learn_topic = prompt.split("/learn", 1)[1].strip()
            from src.tools.internet_learner import InternetLearnerEngine
            pkg = InternetLearnerEngine.learn(learn_topic, project_root=PROJECT_ROOT, rag_pipeline=rag, deep_browse=True)
            response_text = InternetLearnerEngine.format_learning_report(pkg)
            provenance = "Internet Learner"
            st.markdown(response_text)
        elif prompt.startswith("/agent"):
            goal = prompt.split("/agent", 1)[1].strip()
            from src.agents.autonomous_agent import AutonomousWorkAgent
            from src.tools.registry import ToolRegistry
            from src.tools.builtin import register_builtin_tools
            reg = ToolRegistry()
            register_builtin_tools(reg, PROJECT_ROOT)
            agent = AutonomousWorkAgent(reg, max_steps=6, verbose=False)
            mission_res = agent.execute_mission(goal)
            response_text = mission_res.synthesized_output
            provenance = "Autonomous Agent"
            st.markdown(response_text)
        elif prompt.startswith("/gh"):
            gh_args = prompt.split("/gh", 1)[1].strip().split()
            sub = gh_args[0].lower() if gh_args else "search"
            arg_tail = " ".join(gh_args[1:]).strip() if len(gh_args) > 1 else ""
            from src.tools.github_engine import GitHubEngine
            provenance = "GitHub Engine"
            if sub == "search":
                res = GitHubEngine.search_repositories(arg_tail, limit=5)
                response_text = GitHubEngine.format_repos_display(res)
            elif sub == "view":
                res = GitHubEngine.get_repo_info(arg_tail)
                if res.get("success"):
                    d = res.get("data", {})
                    response_text = f"### 📦 GitHub Repository: {res.get('repo')}\n\n- **Description**: {d.get('description')}\n- **Stars**: {d.get('stargazerCount', 0):,}\n- **Forks**: {d.get('forkCount', 0):,}\n- **Default Branch**: `{d.get('defaultBranch', 'main')}`\n- **URL**: {d.get('url')}"
                else:
                    response_text = f"GitHub Error: {res.get('error')}"
            elif sub == "file":
                parts = arg_tail.split()
                if len(parts) < 2:
                    response_text = "Usage: `/gh file <owner/repo> <file_path>`"
                else:
                    res = GitHubEngine.get_file_content(parts[0], parts[1])
                    if res.get("success"):
                        response_text = f"### 📄 File: `{parts[0]}/{parts[1]}` ({res.get('length'):,} bytes)\n```\n{res.get('content')[:3000]}\n```"
                    else:
                        response_text = f"GitHub Error: {res.get('error')}"
            elif sub == "issues":
                res = GitHubEngine.list_issues(arg_tail, limit=5)
                if res.get("success"):
                    lines = [f"### 📋 Issues: `{res.get('repo')}`"]
                    for iss in res.get("issues", []):
                        lines.append(f"- **#{iss.get('number')}**: [{iss.get('title')}]({iss.get('url')}) (`{iss.get('state')}`)")
                    response_text = "\n".join(lines)
                else:
                    response_text = f"GitHub Error: {res.get('error')}"
            else:
                response_text = "Available `/gh` subcommands: `search <query>`, `view <owner/repo>`, `file <owner/repo> <path>`, `issues <owner/repo>`"
            st.markdown(response_text)
        elif prompt.startswith("/git"):
            git_args = prompt.split("/git", 1)[1].strip().split()
            sub = git_args[0].lower() if git_args else "status"
            from src.tools.github_engine import GitHubEngine
            provenance = "Git Engine"
            if sub == "status":
                res = GitHubEngine.git_status(PROJECT_ROOT)
                if res.get("success"):
                    response_text = f"### 🌿 Git Status: `{res.get('branch')}`\n\n- **Modified Files ({res.get('changed_files_count', 0)})**:\n" + "\n".join(f"  - `{f}`" for f in res.get("files", []))
                else:
                    response_text = f"Git Error: {res.get('error')}"
            elif sub == "diff":
                res = GitHubEngine.git_diff(PROJECT_ROOT, max_lines=60)
                if res.get("success"):
                    response_text = f"### 📝 Git Diff\n```diff\n{res.get('patch')}\n```"
                else:
                    response_text = f"Git Error: {res.get('error')}"
            elif sub == "log":
                res = GitHubEngine.git_log(PROJECT_ROOT, limit=5)
                if res.get("success"):
                    response_text = f"### 📜 Git Commit Log\n" + "\n".join(f"- `{c['commit']}`: {c['message']}" for c in res.get("commits", []))
                else:
                    response_text = f"Git Error: {res.get('error')}"
            else:
                response_text = "Available `/git` subcommands: `status`, `diff`, `log`"
            st.markdown(response_text)
        elif prep.direct_answer:
            provenance = "Deterministic Tool"
            response_text = prep.direct_answer
            st.markdown(response_text)
        elif backend_choice == "Local Ollama Pretrained":
            provenance = "Qwen Teacher"
            from src.backends.ollama import OllamaClient
            client = OllamaClient()
            teacher_res = client.generate("qwen3.5:2b", prompt, system="You are an expert research and software engineer.")
            response_text = teacher_res or "Ollama teacher model not responding."
            st.markdown(response_text)
        else:
            provenance = "Scratch Transformer"
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

        source_info = f"🏷️ Source: **[{provenance}]**"
        if prep.retrieved_citations:
            source_info += f" | 📚 Sources: {', '.join(prep.retrieved_citations)}"
        st.caption(source_info)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "thinking": prep.thinking,
            "citations": prep.retrieved_citations,
            "provenance": provenance,
        })

