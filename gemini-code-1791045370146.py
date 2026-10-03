import os
import json
from groq import Groq
import pypdf

MODEL_NAME = "openai/gpt-oss-120b"

def get_groq_client(api_key: str):
    if not api_key:
        raise ValueError("Groq API Key is missing.")
    return Groq(api_key=api_key)

def extract_text_from_pdf(pdf_file) -> str:
    """Extracts raw text content from uploaded PDF file."""
    reader = pypdf.PdfReader(pdf_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text.strip()

def run_document_agent(client: Groq, text: str) -> dict:
    """Member 1 & 2 Agent: Generates comprehensive summary and key takeaways."""
    prompt = f"""
    You are an expert Study Lens Document & Summary Agent.
    Analyze the following study material and provide:
    1. Executive Summary (3-4 concise paragraphs)
    2. Key Concepts & Definitions (bullet points)
    3. Core Takeaways (numbered list)

    Study Material:
    {text[:8000]}
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return {"summary": response.choices[0].message.content}

def run_quiz_agent(client: Groq, text: str, num_questions: int = 5) -> list:
    """Member 3 & 4 Agent: Generates MCQs in strict JSON format."""
    prompt = f"""
    You are a Quiz & Assessment Agent. Generate {num_questions} Multiple Choice Questions (MCQs) based on this content.
    Return ONLY a valid JSON array of objects with no extra text or markdown code blocks outside.
    Each object must have:
    - "id": int
    - "question": string
    - "options": list of 4 strings
    - "answer": string (must match one of the exact strings in options)
    - "explanation": string

    Content:
    {text[:8000]}
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    raw_content = response.choices[0].message.content.strip()
    
    # Strip markdown fence blocks if present
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    if raw_content.startswith("```"):
        raw_content = raw_content[3:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]
        
    return json.loads(raw_content.strip())

def run_performance_and_planner_agent(client: Groq, quiz_results: dict, study_material_topic: str) -> dict:
    """Member 5 Agent: Analyzes test performance and generates a personalized study schedule."""
    prompt = f"""
    You are a Performance Analyst and Study Planner Agent.
    
    Student Evaluation Data:
    - Overall Score: {quiz_results['score']}/{quiz_results['total']} ({quiz_results['percentage']}%)
    - Missed/Weak Concepts: {json.dumps(quiz_results['weak_areas'])}
    - Core Subject Topic: {study_material_topic}

    Task:
    Provide a dual-part response formatted cleanly in Markdown:
    
    ### PART 1: Performance Diagnostics & Feedback
    - Strengths
    - Knowledge Gaps & Analysis
    
    ### PART 2: Personalized 5-Day Recovery Study Plan
    - Day 1 to Day 5 breakdown (Modules, Allocated Time, Focus Topic, Action Items)
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return {"analysis_and_plan": response.choices[0].message.content}