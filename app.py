import streamlit as st
import json
import os
from sampler import ModelInterface, get_available_models, load_config
from optimizer import compute_dpo_loss, compute_grpo_advantage

st.set_page_config(layout="wide", page_title="TriFetch AI Workbench")
st.title("🏥 TriFetch AI: RLHF Control Room")

# --- Load Config ---
config = load_config()
dpo_beta = config["dpo"]["beta"]

# --- Sidebar ---
st.sidebar.header("1. Model")
available_models = get_available_models()

selected_model_key = st.sidebar.selectbox(
    "Select Model:",
    list(available_models.keys()),
    index=list(available_models.keys()).index(config['default']),
    format_func=lambda x: available_models[x]
)

st.sidebar.header("2. Case")

# --- Load Data ---
@st.cache_resource(show_spinner="Loading model...")
def get_model(model_key):
    return ModelInterface(model_key)

@st.cache_data
def load_samples():
    samples = {}
    for i in range(1, 6):
        filename = f"sample{i}.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                samples[f"Patient Case {i}"] = json.load(f)
    return samples

try:
    model = get_model(selected_model_key)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

samples_data = load_samples()
if not samples_data:
    st.error("No sample files found!")
    st.stop()

# --- Session State ---
if "traces" not in st.session_state:
    st.session_state.traces = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "current_case" not in st.session_state:
    st.session_state.current_case = None
if "prompt" not in st.session_state:
    st.session_state.prompt = None

selected_case = st.sidebar.selectbox("Select Case:", list(samples_data.keys()))

if selected_case != st.session_state.current_case:
    st.session_state.traces = None
    st.session_state.logs = []
    st.session_state.current_case = selected_case

current_data = samples_data[selected_case]

# --- Clear All Button ---
st.sidebar.divider()
st.sidebar.header("3. Actions")

if st.sidebar.button("🗑️ Clear All", type="secondary"):
    st.session_state.traces = None
    st.session_state.logs = []
    st.session_state.prompt = None
    st.session_state.current_case = None
    st.rerun()

# Show current settings
st.sidebar.divider()
st.sidebar.caption(f"**DPO Beta:** {dpo_beta}")

# --- Main UI ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Patient Context")
    st.write(current_data["Questions"])
    st.success(f"**Answer:** {current_data['Answer']}")
    
    st.divider()
    st.write(f"**Model:** {available_models[selected_model_key]}")
    
    # Logs
    log_container = st.empty()
    def render_logs():
        with log_container.container():
            for t, msg in st.session_state.logs:
                if t == "success": st.success(msg)
                elif t == "error": st.error(msg)
                elif t == "warning": st.warning(msg)
                else: st.write(msg)
    
    if st.session_state.logs:
        render_logs()
    
    if st.button("Generate Traces", type="primary"):
        st.session_state.logs = []
        st.session_state.traces = None
        st.session_state.prompt = current_data["Questions"]
        
        for update_type, content in model.generate_verified_traces(
            current_data["Questions"], current_data["Answer"]
        ):
            if update_type == "result":
                st.session_state.traces = content
            else:
                st.session_state.logs.append((update_type, content))
                render_logs()
        st.rerun()

with col2:
    if st.session_state.traces:
        traces = st.session_state.traces
        
        st.subheader("📝 Traces")
        t1, t2, t3 = st.tabs(["Trace A", "Trace B", "Trace C"])
        with t1: st.write(traces[0] if len(traces) > 0 else "N/A")
        with t2: st.write(traces[1] if len(traces) > 1 else "N/A")
        with t3: st.write(traces[2] if len(traces) > 2 else "N/A")
        
        st.divider()
        st.subheader("👨‍⚕️ Rank Traces")
        
        c1, c2, c3 = st.columns(3)
        with c1: rank_best = st.selectbox("🥇 Best", ["Trace A", "Trace B", "Trace C"], index=0)
        with c2: rank_mid = st.selectbox("🥈 Middle", ["Trace A", "Trace B", "Trace C"], index=1)
        with c3: rank_worst = st.selectbox("🥉 Worst", ["Trace A", "Trace B", "Trace C"], index=2)
        
        if st.button("Update Model"):
            trace_map = {"Trace A": 0, "Trace B": 1, "Trace C": 2}
            best_idx, mid_idx, worst_idx = trace_map[rank_best], trace_map[rank_mid], trace_map[rank_worst]
            
            prompt = st.session_state.prompt
            
            with st.spinner("Computing..."):
                policy_chosen = model.get_sequence_log_prob(prompt, traces[best_idx])
                policy_rejected = model.get_sequence_log_prob(prompt, traces[worst_idx])
                ref_chosen = model.get_sequence_log_prob(prompt, traces[best_idx], use_reference=True)
                ref_rejected = model.get_sequence_log_prob(prompt, traces[worst_idx], use_reference=True)
            
            # GRPO
            rewards = [0.0, 0.0, 0.0]
            rewards[best_idx], rewards[mid_idx], rewards[worst_idx] = 1.0, 0.5, 0.0
            advantages = compute_grpo_advantage(rewards)
            
            # DPO - using beta from config
            dpo_loss = compute_dpo_loss(
                policy_chosen, 
                policy_rejected, 
                ref_chosen, 
                ref_rejected,
                beta=dpo_beta
            )
            
            st.success("✅ Done!")
            
            st.metric("DPO Loss", f"{dpo_loss.item():.4f}")
            
            st.write("**GRPO Advantages:**")
            a1, a2, a3 = st.columns(3)
            with a1: st.metric("Trace A", f"{advantages[0].item():.4f}")
            with a2: st.metric("Trace B", f"{advantages[1].item():.4f}")
            with a3: st.metric("Trace C", f"{advantages[2].item():.4f}")
            
            st.json({
                "Config": {
                    "Model": selected_model_key,
                    "DPO Beta": dpo_beta
                },
                "DPO Loss": f"{dpo_loss.item():.4f}",
                "GRPO Rewards": rewards,
                "GRPO Advantages": [f"{a.item():.4f}" for a in advantages],
                "Ranking": {"Best": rank_best, "Middle": rank_mid, "Worst": rank_worst}
            })
    else:
        st.info("👈 Generate traces first")