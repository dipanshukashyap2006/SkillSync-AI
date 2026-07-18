import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import spacy
import fitz  # PyMuPDF
import requests
from sentence_transformers import SentenceTransformer, util
import traceback
import google.generativeai as genai

# --- INITIALIZE GLOBAL VARIABLES ---
# This ensures the app doesn't crash when switching tabs
if 'missing_skills' not in st.session_state:
    st.session_state['missing_skills'] = []

if 'job_desc' not in st.session_state:
    st.session_state['job_desc'] = ""
# This creates a local "shortcut" to the data in the locker
missing_skills = st.session_state['missing_skills']
job_desc = st.session_state['job_desc']

st.set_page_config(page_title="SkillSync AI", page_icon="🚀", layout="wide")
st.title("🚀 SkillSync AI")
st.subheader("Analyze your gap to the industry standards.")


# 1. LOAD MODELS (Optimized for speed)
@st.cache_resource(show_spinner="🚀 Powering up SkillSync AI... please wait.")
def load_models():
    import spacy
    from sentence_transformers import SentenceTransformer
    
    # Load NLP for skill extraction
    nlp = spacy.load("en_core_web_sm")
    
    # Load the "Brain" (SentenceTransformer)
    # This model is small (420MB) but very smart
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    
    return nlp, embed_model

# Initialize the models
nlp, embed_model = load_models()



# 2. SKILL DATABASE
skill_db = [
    "Python", "Java", "C++", "SQL", "Machine Learning", "Deep Learning", 
    "Operating Systems", "Data Structures", "Algorithms", "Git", "GitHub", 
    "API", "REST API", "JSON", "XML", "Docker", "AWS", "Cloud Computing",
    "Data Transformation", "Data Cleaning", "Tableau", "PowerBI", "FastAPI"
]
# --- SIDEBAR (Clean & Professional) ---
# --- SIDEBAR (Dynamic Profile) ---
# --- 1. SIDEBAR (Manual Entry) ---
with st.sidebar:
    st.header("👤 User Profile")
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80) 
    
    name = st.text_input("Candidate Name:", placeholder="Enter your name...")
    status = st.text_input("Academic Status:", placeholder="e.g., 4th Sem VTU")
    
    st.divider()
    st.markdown(f"**Engine:** 💻 MSI Thin 15\n\n🚀 RTX 3050 (CUDA)")
    st.caption("Theme: Social Impact, Equal Opportunity towards tech careers & Skill Development.")

# --- 2. MAIN DASHBOARD ---
resume_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

if resume_file:
    # --- SECTION A: RESUME PARSING ---
    with st.spinner("Analyzing your resume..."):
        import fitz  # PyMuPDF
        # Reset file pointer and read the PDF content
        resume_file.seek(0)
        doc = fitz.open(stream=resume_file.read(), filetype="pdf")
        full_text = ""
        for page in doc:
            # Get text as blocks (x0, y0, x1, y1, "text", block_no, block_type)
            blocks = page.get_text("blocks")
            
            # Sort blocks: First by y-coordinate (top-to-bottom), 
            # then by x-coordinate (left-to-right)
            blocks.sort(key=lambda b: (b[1], b[0]))
            
            for b in blocks:
                if b[4].strip():  # Check if the block actually contains text
                    full_text += b[4] + "\n"
        
        # Identify skills from your skill_db
        found_skills = [s for s in skill_db if s.lower() in full_text.lower()]
    # Header that uses your manual inputs
    display_name = name if name else "Candidate"
    st.title(f"📊 {display_name}'s Skill Analysis")
    if status:
        st.subheader(f"Targeting: {status}")

    tab1, tab2, tab3 = st.tabs(["🎯 Gap Analysis", "✍️ AI Optimizer", "🛡️ GitHub & Export"])

    with tab1:
        st.subheader("Industry Gap Analysis")
        job_desc = st.text_area("Paste Job Description:", height=150)
        
        if job_desc:
            try:
                with st.spinner("Calculating Match Score on RTX 3050..."):
                    # GPU embedding logic
                    job_embed = embed_model.encode(job_desc, convert_to_tensor=True)
                    resume_embed = embed_model.encode(full_text, convert_to_tensor=True)
                    score = util.cos_sim(resume_embed, job_embed)
                    match_pct = round(float(score[0][0]) * 100, 2)

                # --- VISUAL SECTION ---
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.metric("Match Score", f"{match_pct}%")
                    if match_pct > 70:
                        st.success("🔥 Strong Match!")
                    else:
                        st.warning("⚠️ Improvement Needed")
                
                with col2:
                    if found_skills:
                        st.write("**💎 Identified Skill Strengths:**")
                        badges_html = " ".join([
                            f'<span style="background-color: #1E3A8A; color: #93C5FD; padding: 6px 12px; margin: 4px; border-radius: 16px; font-weight: 500; display: inline-block;">{skill}</span>' 
                            for skill in found_skills
                        ])
                        st.markdown(badges_html, unsafe_allow_html=True)

                # --- GAP ANALYSIS (Inside Column 2) ---
                st.divider()
                st.write("🚩 **Industry Skill Gaps:**")
                
                # Identify missing keywords
                missing_skills = [s for s in skill_db if s.lower() in job_desc.lower() and s.lower() not in full_text.lower()]
                
                if missing_skills:
                    gap_cols = st.columns(min(len(missing_skills), 4)) 
                    for i, skill in enumerate(missing_skills[:4]):
                        gap_cols[i].error(f"{skill}")
                else:
                    st.success("🎯 No major gaps found!")

                # --- END OF COLUMN 2 ---

            except Exception as e: # <--- PULL THIS BACK to align with 'try' on Line 89
                st.error(f"🛑 Engine Error: {e}")
                import traceback
                traceback.print_exc()

    with tab2:
        st.subheader("✍️ AI Resume Optimizer")
        st.write("Let Gemini rewrite your experience to bridge the identified gaps.")
        # 1. Check the vault first
        user_api_key = st.secrets.get("GEMINI_API_KEY")
        
        # 2. Only show the input box if the vault is empty
        if not user_api_key:
            user_api_key = st.text_input("Enter Gemini API Key:", type="password")

        # Check if we have an API key and missing skills
        if user_api_key and missing_skills:
            if st.button("Generate Optimization Strategy"):
                with st.spinner("Consulting Gemini for career insights..."):
                    try:
                        # Configure Gemini
                        import google.generativeai as genai
                        genai.configure(api_key=user_api_key)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        
                        # The Prompt: Ask for a bullet point rewrite
                        prompt = f"""
                        A candidate is missing these skills: {', '.join(missing_skills)}. 
                        The Job Description requires: {job_desc[:500]}...
                        
                        Please provide:
                        1. Three optimized resume bullet points that integrate these missing skills naturally.
                        2. A 'Wildcard' project idea they can build this weekend to prove they know these skills.
                        """
                        
                        response = model.generate_content(prompt)
                        st.markdown("### 🚀 Suggested Improvements")
                        st.write(response.text)
                        
                    except Exception as e:
                        if "429" in str(e):
                            st.error("🚦 **Traffic Jam!** The Free Tier limit was hit. Please wait 30 seconds before clicking again.")
                        else:
                            st.error(f"❌ API Error: {e}")
                        st.error(f"❌ Gemini Error: {e}")
        elif not user_api_key:
            st.warning("🔑 Please enter your Gemini API Key in the sidebar or Tab 2 to use this feature.")
        else:
            st.info("💡 Upload a resume and Job Description first to see optimization tips.")

    with tab3:
        st.subheader("📊 GitHub Skill Pulse")
        
        # 1. THE MISSING INPUT FIELD
        gh_username = st.text_input("Enter your GitHub Username to analyze proficiency:", 
                                placeholder="e.g., your-github-id")
        
        if gh_username:
            with st.spinner(f"Scouting {gh_username}'s repositories..."):
                try:
                    from github import Github
                    # Uses secrets for the token to avoid friction/leaks
                    g = Github(st.secrets.get("GITHUB_TOKEN", None))
                    user = g.get_user(gh_username)
                    
                    lang_stats = {}
                    for repo in user.get_repos():
                        if not repo.fork: # Only count your original code
                            # 1. Get the dictionary of languages
                            langs = repo.get_languages()
                            
                            # 2. Safety check: Iterate and only count if 'b' is a number
                            for lang_name, b in langs.items():
                                try:
                                    # Only add if it's actually a number (int or float)
                                    if isinstance(b, (int, float)):
                                        lang_stats[lang_name] = lang_stats.get(lang_name, 0) + b
                                    elif str(b).isdigit():
                                        lang_stats[lang_name] = lang_stats.get(lang_name, 0) + int(b)
                                except:
                                    continue # Skip anything that isn't a number (like URLs)
                    
                    if lang_stats:
                        # 2. THE LEVEL MAPPING (Bytes to Professional Tiers)
                        level_data = []
                        for lang, b in lang_stats.items():
                            if b > 150000: level, rank = "Expert", 4
                            elif b > 50000: level, rank = "Advanced", 3
                            elif b > 10000: level, rank = "Intermediate", 2
                            else: level, rank = "Beginner", 1
                            
                            level_data.append({"Lang": lang, "Rank": rank, "Level": level})
                        
                        # Sort so the best skills are first
                        level_data = sorted(level_data, key=lambda x: x['Rank'], reverse=True)

                        # 3. THE VISUALIZATION (Professional Bar Chart)
                        import plotly.express as px
                        fig = px.bar(level_data, x="Lang", y="Rank", color="Rank",
                                    text="Level", color_continuous_scale="Tealgrn",
                                    labels={"Rank": "Proficiency", "Lang": "Language"})
                        
                        fig.update_layout(yaxis=dict(tickvals=[1,2,3,4], 
                                        ticktext=['Beginner', 'Intermediate', 'Advanced', 'Expert']),
                                        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", 
                                        plot_bgcolor="rgba(0,0,0,0)")
                        
                        st.plotly_chart(fig, use_container_width=True)

                        # --- NEW HOLISTIC ROADMAP (Replaces Lines 256-274) ---
                        st.divider()
                        st.subheader("🚀 Your Full-Stack Growth Roadmap")
                        
                        # 1. Group skills by rank
                        summary = {
                            "Expert": [i['Lang'] for i in level_data if i['Level'] == "Expert"],
                            "Advanced": [i['Lang'] for i in level_data if i['Level'] == "Advanced"],
                            "Intermediate": [i['Lang'] for i in level_data if i['Level'] == "Intermediate"],
                            "Beginner": [i['Lang'] for i in level_data if i['Level'] == "Beginner"]
                        }

                        # 2. Display a quick summary of what the AI is looking at
                        all_beg = ", ".join(summary['Beginner']) if summary['Beginner'] else "None"
                        st.info(f"💡 Found **{len(summary['Beginner'])}** Beginner skills to level up: {all_beg}")

                        if st.button("✨ Generate Unified Level-Up Plan"):
                            with st.spinner("Analyzing your entire tech stack..."):
                                # Create a clear picture of the user's brain for Gemini
                                profile_context = "\n".join([f"{rank}: {', '.join(langs)}" for rank, langs in summary.items() if langs])
                                
                                prompt = f"""
                                You are a senior tech lead. Analyze this developer's current GitHub profile levels:
                                {profile_context}

                                Based on this entire stack, suggest:
                                1. A 'Bridge Project' that uses their Advanced/Expert skills to help them practice their Beginner skills.
                                2. A combined learning roadmap to get all 'Beginner' skills to 'Intermediate' this month.
                                3. Which specific skill they should prioritize next for their career.
                                """
                                
                                try:
                                    # Using the global 'genai' we imported earlier
                                    model_rec = genai.GenerativeModel('gemini-2.5-flash')
                                    res_rec = model_rec.generate_content(prompt)
                                    st.markdown(res_rec.text)
                                except Exception as ai_err:
                                    st.error(f"AI Roadmap failed: {ai_err}")
                        # --- END OF INSERTION ---
                    else:
                        st.warning("No original repositories found for this user.")
                except Exception as e:
                    st.error(f"GitHub Error: {e}")

        st.divider()
        
        # Keep your original Export & GitHub Link buttons at the bottom
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("### 🐙 Project Source")
            st.link_button("View on GitHub", "https://github.com/YOUR_USERNAME/SkillSync-AI")
        with col_b:
            st.write("### 📄 Save Results")
            if 'response' in locals():
                st.download_button("Download Report", response.text, "Report.txt")
# Final build