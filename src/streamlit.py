# src/streamlit.py
from __future__ import annotations

# --- Path fix (pour: poetry run python -m streamlit run src/streamlit.py)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

# --- Projet (même pipeline que la CLI)
from src.scripts.query_rag import answer
from src.scripts.build_index import main as rebuild_index
from src.config import settings


# ===========================
#    UI CONFIG
# ===========================
st.set_page_config(page_title="Puls-Events RAG", page_icon="🎟️", layout="wide")
st.title("🎟️ Puls-Events — Chat RAG")
st.caption("Chat (Mistral API ou Ollama local) • FAISS (recherche MMR) • Embeddings selon build_index")


# ===========================
#    SIDEBAR
# ===========================
with st.sidebar:
    st.subheader("Réglages")
    top_k = st.slider("k (passages à considérer)", 3, 10, 5, 1)
    temperature = st.slider("Température (créativité)", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Taille de réponse (tokens)", 50, 1000, 300, 50)

    st.write("---")
    st.subheader("Chat backend")
    # keep the label text so your existing code can map it
    chat_backend = st.radio(
        "Backend de génération",
        options=["Mistral API", "Ollama local"],
        index=0,
        help="Choisis entre l'API Mistral ou un modèle local via Ollama.",
    )

    # model pickers, keeping YOUR variable names
    mistral_model = st.text_input(
        "Modèle Mistral",
        value=getattr(settings, "mistral_chat_model", "mistral-medium-2508"),
        help="Ex: mistral-small-latest, mistral-medium-2508, etc.",
        disabled=(chat_backend != "Mistral API"),
    )

    # sensible defaults for Ollama
    default_ollama_base = "http://127.0.0.1:11434"
    ollama_base_url = st.text_input(
        "Ollama base URL",
        value=default_ollama_base,
        help="Astuce Windows: utilisez 127.0.0.1 plutôt que localhost.",
        disabled=(chat_backend != "Ollama local"),
    )

    # try to list local models; fallback to a small list
    ollama_models = ["mistral", "llama3.2", "llama2"]
    if chat_backend == "Ollama local":
        try:
            import requests
            resp = requests.get(f"{ollama_base_url.rstrip('/')}/api/tags", timeout=1.5)
            if resp.ok:
                tags = [t["name"].split(":")[0] for t in (resp.json().get("models") or [])]
                if tags:
                    ollama_models = sorted(set(tags))
        except Exception:
            pass

    ollama_chat_model = st.selectbox(
        "Modèle Ollama",
        options=ollama_models,
        index=min(ollama_models.index("mistral") if "mistral" in ollama_models else 0, len(ollama_models)-1),
        disabled=(chat_backend != "Ollama local"),
    )

    st.write("---")
    st.subheader("Retrieval (MMR)")
    use_mmr = st.checkbox("Activer MMR (diversité)", value=True)
    mmr_fetch_k = st.slider("fetch_k (pool initial)", 20, 400, 100, 10)
    mmr_lambda = st.slider("lambda_mult (0=diversité, 1=similarité)", 0.0, 1.0, 0.2, 0.05)

    st.write("---")
    if st.button("🔄 Rebuild index (local)", type="primary"):
        with st.spinner("Reconstruction de l'index FAISS..."):
            try:
                rebuild_index()
                st.success(f"Index reconstruit ✅\n➡ {settings.index_dir}")
            except Exception as e:
                st.error(f"Erreur rebuild: {e}")

    st.write("---")
    st.subheader("Configuration RAG")
    st.code(
        f"INDEX_DIR = {settings.index_dir}\n"
        f"BACKEND = {chat_backend}\n"
        f"MISTRAL_CHAT_MODEL = {mistral_model}\n"
        f"OLLAMA_CHAT_MODEL = {ollama_chat_model}\n"
        f"temperature = {temperature}\n"
        f"max_tokens = {max_tokens}\n"
        f"use_mmr = {use_mmr} | fetch_k = {mmr_fetch_k} | lambda = {mmr_lambda}\n",
        language="bash",
    )


# ===========================
#    SESSION STATE
# ===========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content}]
if "last_contexts" not in st.session_state:
    st.session_state.last_contexts = []
if "last_ranking" not in st.session_state:
    st.session_state.last_ranking = []


# ===========================
#    HELPERS
# ===========================
def ask_and_render(question: str, k: int) -> None:
    # message user
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # réponse + ranking
    with st.chat_message("assistant"):
        with st.spinner("Je cherche dans la base et je rédige..."):
            try:
                res = answer(
                    question,
                    k=k,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_mmr=use_mmr,
                    mmr_fetch_k=mmr_fetch_k,
                    mmr_lambda_mult=mmr_lambda,
                    # --- nouveautés : keep YOUR exact arg names
                    chat_backend=("mistral" if chat_backend == "Mistral API" else "ollama"),
                    mistral_model=mistral_model,
                    ollama_chat_model=ollama_chat_model,
                    ollama_base_url=ollama_base_url,
                )
                answer_text = res.get("answer", "Je ne sais pas.")
                contexts = res.get("contexts", [])
                ranking = res.get("ranking", [])
            except Exception as e:
                import traceback
                st.error(f"**{type(e).__name__}**: {e}")
                st.code("".join(traceback.format_exc()), language="text")
                answer_text, contexts, ranking = "Je ne sais pas.", [], []
        st.markdown(answer_text)
        st.session_state.messages.append({"role": "assistant", "content": answer_text})
        st.session_state.last_contexts = contexts
        st.session_state.last_ranking = ranking


# ===========================
#    CHAT HISTORY
# ===========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ===========================
#    CHAT INPUT
# ===========================
user_q = st.chat_input("Pose ta question (ex: Quels concerts à Toulouse en juin 2024 ?)")
if user_q and user_q.strip():
    ask_and_render(user_q.strip(), k=top_k)


# ===========================
#    PRESET BUTTONS
# ===========================
st.write("---")
st.write("#### Exemples rapides")
cols = st.columns(3)
examples = [
    "Quels concerts à Toulouse en juin 2024 ?",
    "Expositions à Montpellier en septembre 2024 ?",
    "Que faire à Paris cet été ?",
]
for c, ex in zip(cols, examples):
    if c.button(ex, key=f"ex_{ex}", help="Clique pour pré-remplir"):
        ask_and_render(ex, k=top_k)
        st.rerun()


# ===========================
#    CONTEXTS + RANKING
# ===========================
ranking = st.session_state.last_ranking or []
contexts = st.session_state.last_contexts or []

if ranking:
    st.write("### Passages utilisés")

    df_rank = pd.DataFrame(ranking)

    # tri par score de similarité décroissant
    df_sorted = (
        df_rank
        .assign(sim_score=df_rank["sim_score"].fillna(-1))
        .sort_values(by="sim_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    df_sorted.insert(0, "rank_by_score", df_sorted.index + 1)

    # ordre de colonnes lisible
    cols_order = [
        c for c in ["rank_by_score","pos_in_context","mmr_rank","sim_score",
                    "title","city","start","url","uid"]
        if c in df_sorted.columns
    ]
    st.dataframe(df_sorted[cols_order], width="stretch", hide_index=True)

    # réordonner les contextes selon le tri par sim_score
    ctx_sorted = []
    for _, r in df_sorted.iterrows():
        pos = int(r["pos_in_context"]) - 1  # index dans la liste contexts
        if 0 <= pos < len(contexts):
            ctx_sorted.append((r, contexts[pos]))

    # expanders dans l'ordre par score
    for r, ctx in ctx_sorted:
        subtitle = f"Passage {int(r['rank_by_score'])} — sim={r['sim_score']:.4f}"
        if pd.notna(r.get("mmr_rank")):
            subtitle += f" | MMR#{int(r['mmr_rank'])}"
        with st.expander(subtitle):
            lines = ctx.splitlines()
            if lines:
                st.code(lines[0], language="text")  # header “Titre | Ville | Début | URL”
                body = "\n".join(lines[1:]).strip()
                if body:
                    st.write(body)
            else:
                st.write(ctx)
