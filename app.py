import os
import streamlit as st
import pandas as pd
import plotly.express as px
import json
from agent_connectors import (
    get_groq_client,
    extract_text_from_pdf,
    run_document_agent,
    run_quiz_agent,
    run_performance_and_planner_agent
)

# Page Configuration
st.set_page_config(
    page_title="StudyLens AI - Student Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/graduation-cap.png", width=70)
    st.title("StudyLens AI")
    st.caption("Multi-Agent Learning Support System")
    st.divider()
    
    # API Key Handling (Reads from environment/secrets if set, otherwise prompts user)
    default_key = os.getenv("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        placeholder="Enter your Groq API key (gsk_...)",
        help="Enter your Groq API key to activate the agents."
    )
    
    st.divider()
    st.markdown("### Agent Architecture")
    st.markdown("""
    - **Doc Agent:** PDF Ingestion & Summarization
    - **Quiz Agent:** Automated MCQ Generation
    - **Performance Agent:** Weak-Spot Diagnostic
    - **Planner Agent:** Personalized Study Schedule
    """)

# State Management
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = None
if "planner_result" not in st.session_state:
    st.session_state.planner_result = None

# Header
st.title("🎓 StudyLens AI Portal")
st.markdown("Upload your study documents to generate instant summaries, interactive assessments, and customized learning schedules.")

# Navigation Tabs
tab_upload, tab_summary, tab_quiz, tab_performance, tab_planner = st.tabs([
    "📁 1. Document Upload",
    "📝 2. Summary & Concepts",
    "❓ 3. Interactive Quiz",
    "📊 4. Performance Analytics",
    "📅 5. Personalized Study Plan"
])

# -----------------------------------------------------------------------------
# TAB 1: DOCUMENT UPLOAD INTERFACE
# -----------------------------------------------------------------------------
with tab_upload:
    st.header("Upload Study Material")
    uploaded_file = st.file_uploader("Upload a PDF document to begin analysis", type=["pdf"])
    
    if uploaded_file is not None:
        st.info(f"File attached: **{uploaded_file.name}** ({round(uploaded_file.size / 1024, 2)} KB)")
        if st.button("🚀 Process Document with Agents", type="primary"):
            if not api_key_input:
                st.error("Please enter a valid Groq API Key in the sidebar before processing.")
            else:
                try:
                    with st.status("Agents operating on document...", expanded=True) as status:
                        st.write("📄 Extracting raw text from PDF...")
                        client = get_groq_client(api_key_input)
                        extracted_text = extract_text_from_pdf(uploaded_file)
                        st.session_state.pdf_text = extracted_text
                        
                        st.write("🤖 Running Document Agent (Summarization)...")
                        summary_res = run_document_agent(client, extracted_text)
                        st.session_state.summary_result = summary_res["summary"]
                        
                        st.write("❓ Running Quiz Agent (Generating MCQs)...")
                        quiz_res = run_quiz_agent(client, extracted_text, num_questions=5)
                        st.session_state.quiz_data = quiz_res
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_results = None
                        st.session_state.planner_result = None
                        
                        status.update(label="Document successfully processed across all agents!", state="complete", expanded=False)
                    st.success("Analysis Complete! Navigate to the next tabs to view your summary and test your knowledge.")
                except Exception as e:
                    st.error(f"Error executing agent pipeline: {str(e)}")

# -----------------------------------------------------------------------------
# TAB 2: SUMMARY & EXPLANATION RESULTS
# -----------------------------------------------------------------------------
with tab_summary:
    st.header("Document Summary & Core Concepts")
    if st.session_state.summary_result:
        st.markdown(st.session_state.summary_result)
    else:
        st.warning("No summary available. Please upload and process a PDF in Tab 1.")

# -----------------------------------------------------------------------------
# TAB 3: QUESTION / QUIZ INTERFACE
# -----------------------------------------------------------------------------
with tab_quiz:
    st.header("Interactive Assessment")
    if st.session_state.quiz_data:
        with st.form("quiz_form"):
            user_answers = {}
            for q in st.session_state.quiz_data:
                st.subheader(f"Q{q['id']}: {q['question']}")
                user_answers[q['id']] = st.radio(
                    "Select your answer:",
                    q['options'],
                    key=f"q_{q['id']}"
                )
                st.divider()
            
            submit_quiz = st.form_submit_button("Submit Quiz & Generate Analytics", type="primary")
            
            if submit_quiz:
                score = 0
                total = len(st.session_state.quiz_data)
                weak_areas = []
                
                for q in st.session_state.quiz_data:
                    selected = user_answers[q['id']]
                    if selected == q['answer']:
                        score += 1
                    else:
                        weak_areas.append({
                            "question": q['question'],
                            "your_answer": selected,
                            "correct_answer": q['answer'],
                            "explanation": q['explanation']
                        })
                
                pct = round((score / total) * 100, 2)
                st.session_state.quiz_results = {
                    "score": score,
                    "total": total,
                    "percentage": pct,
                    "weak_areas": weak_areas,
                    "user_answers": user_answers
                }
                st.session_state.quiz_submitted = True
                st.success("Quiz submitted! Your results and personalized plan have been triggered.")
    else:
        st.warning("No active quiz found. Please upload and process a PDF document first.")

# -----------------------------------------------------------------------------
# TAB 4: PERFORMANCE RESULTS DISPLAY
# -----------------------------------------------------------------------------
with tab_performance:
    st.header("Performance Analytics Dashboard")
    if st.session_state.quiz_submitted and st.session_state.quiz_results:
        res = st.session_state.quiz_results
        
        # Key Metrics Row
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Score", f"{res['score']} / {res['total']}")
        col2.metric("Accuracy Percentage", f"{res['percentage']}%")
        col3.metric("Incorrect Questions", len(res['weak_areas']))
        
        # Donut Visual Chart
        fig_data = pd.DataFrame({
            "Status": ["Correct", "Incorrect"],
            "Count": [res['score'], res['total'] - res['score']]
        })
        fig = px.pie(
            fig_data,
            values="Count",
            names="Status",
            hole=0.4,
            color="Status",
            color_discrete_map={"Correct": "#2ecc71", "Incorrect": "#e74c3c"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Weak Area Review
        if res['weak_areas']:
            st.subheader("⚠️ Review Required Concepts")
            for idx, item in enumerate(res['weak_areas']):
                with st.expander(f"Missed Item {idx+1}: {item['question']}"):
                    st.write(f"**Your Answer:** {item['your_answer']}")
                    st.write(f"**Correct Answer:** {item['correct_answer']}")
                    st.info(f"**Explanation:** {item['explanation']}")
    else:
        st.warning("Please complete and submit the quiz in Tab 3 to view performance analytics.")

# -----------------------------------------------------------------------------
# TAB 5: PERSONALIZED STUDY PLAN DISPLAY
# -----------------------------------------------------------------------------
with tab_planner:
    st.header("AI-Generated Study Schedule")
    if st.session_state.quiz_submitted and st.session_state.quiz_results:
        if not st.session_state.planner_result:
            if st.button("⚡ Generate Targeted Recovery Plan", type="primary"):
                if not api_key_input:
                    st.error("Please enter a valid Groq API Key in the sidebar.")
                else:
                    with st.spinner("Performance Analyst Agent drafting custom study roadmap..."):
                        client = get_groq_client(api_key_input)
                        plan_data = run_performance_and_planner_agent(
                            client,
                            st.session_state.quiz_results,
                            study_material_topic="Uploaded Study Material"
                        )
                        st.session_state.planner_result = plan_data["analysis_and_plan"]
                        st.rerun()
        else:
            st.markdown(st.session_state.planner_result)
    else:
        st.warning("Please submit your quiz in Tab 3 first so the agent can personalize your plan.")
