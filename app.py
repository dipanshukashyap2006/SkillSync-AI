import streamlit as st
import spacy
import fitz  # PyMuPDF
import requests
from sentence_transformers import SentenceTransformer, util

# 1. LOAD MODELS (Outside the main loop so they only load once)
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    # This is the "Brain" for the Gap Engine
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return nlp, embed_model

nlp, embed_model = load_models()

st.set_page_config(page_title="SkillSync AI", page_icon="🚀", layout="wide")
st.title("🚀 SkillSync AI")
st.subheader("Analyze your gap to the industry standards.")

# 2. SKILL DATABASE
skill_db = ["Python", "Java", "SQL", "Machine Learning", "C++", "React", "DBMS", "Operating Systems"]

# SIDEBAR
st.sidebar.header("User Profile")
name = st.sidebar.text_input("Enter your Name")

# 3. MAIN LOGIC 
resume_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

if resume_file:
    # --- SECTION A: RESUME PARSING 
    with st.spinner("Analyzing your resume..."):
        
        doc = fitz.open(stream=resume_file.read(), filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        found_skills = [skill for skill in skill_db if skill.lower() in full_text.lower()]
    
    # Display Results 
    st.success("Analysis Complete!")
    st.write(f"### Hello {name if name else 'Candidate'}, here are the skills we found:")
    if found_skills:
        st.write(", ".join([f"`{s}`" for s in found_skills]))
    else:
        st.warning("No skills found from our database.")

    # --- SECTION B: GAP ENGINE  ---
    st.divider()
    st.subheader("🎯 Step 2: Industry Gap Analysis")
    
    job_desc = st.text_area("Paste the Job Description here:", height=200, placeholder="Paste requirements from LinkedIn...")

    if job_desc:
        with st.status("Calculating Similarity using RTX 3050...", expanded=True):
            # Convert text into vectors (embeddings)
            resume_vec = embed_model.encode(full_text, convert_to_tensor=True)
            jd_vec = embed_model.encode(job_desc, convert_to_tensor=True)
            
            # Compute Cosine Similarity
            score = util.cos_sim(resume_vec, jd_vec)
            match_pct = round(float(score) * 100, 2)
            
        # Display the result
        st.metric(label="Overall Match Score", value=f"{match_pct}%")
        
        # Logic for the "Gap"
        if match_pct < 50:
            st.error("⚠️ Significant Gap Found: Your resume needs more keywords from this job.")
        elif match_pct < 75:
            st.warning("⚡ Moderate Match: You're close! Adding a few more details could help.")
        else:
            st.success("🔥 High Match! You're ready to apply.")

    # --- SECTION C: GITHUB VERIFIER
    st.divider()
    st.subheader("🛡️ Step 3: GitHub Verification")
    github_user = st.text_input("Enter GitHub Username to verify skills:")
    
    if github_user:
        with st.status(f"Scanning {github_user}'s GitHub for proof...", expanded=True):
            # 1. Hit the GitHub API
            url = f"https://api.github.com/users/{github_user}/repos"
            response = requests.get(url)
            
            if response.status_code == 200:
                repos = response.json()
                # 2. Extract the main language from each repo
                languages = [r['language'] for r in repos if r['language']]
                unique_langs = list(set(languages))
                
                st.write(f"✅ Found **{len(repos)}** public repositories!")
                st.write(f"**Top Tech Found:** {', '.join(unique_langs)}")
                
                # 3. THE TRUTH TEST: Cross-reference GitHub with Resume
                verified_skills = [lang for lang in unique_langs if lang in found_skills]
                
                if verified_skills:
                    st.success(f"**Verified Proof Found for:** {', '.join(verified_skills)} 🛡️")
                    st.balloons() # Victory lap!
                else:
                    st.warning("We found your projects, but they don't match the skills on your resume yet.")
            else:
                st.error("GitHub profile not found. Check the username and try again!")