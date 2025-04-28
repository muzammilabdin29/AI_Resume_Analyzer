import streamlit as st
import pdfplumber
import pandas as pd
import re
import nltk
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load NLP Models
try:
    nlp = spacy.load("en_core_web_trf")
except:
    nlp = spacy.load("en_core_web_sm")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Job roles
job_roles = {
    "Data Scientist": "Python, Machine Learning, Deep Learning, SQL, Data Analysis",
    "Software Engineer": "Java, Python, C++, Software Development, OOP, Cloud",
    "Web Developer": "HTML, CSS, JavaScript, React, Node.js, UI/UX",
    "DevOps Engineer": "Docker, Kubernetes, AWS, CI/CD, Linux",
    "AI Engineer": "Deep Learning, NLP, PyTorch, TensorFlow, Generative AI, LLMs",
    "Backend Developer": "Node.js, Python, Django, Flask, REST APIs, SQL, PostgreSQL, MongoDB",
    "Frontend Developer": "React, Angular, Vue.js, JavaScript, TypeScript, CSS",
    "Full Stack Developer": "MERN Stack, MEAN Stack, PHP, Django, GraphQL",
    "Machine Learning Engineer": "Python, TensorFlow, PyTorch, Deep Learning, ML Algorithms, Scikit-Learn"
}
job_df = pd.DataFrame(job_roles.items(), columns=["Job Role", "Required Skills"])


# --------- FUNCTIONS ---------

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def clean_text(text):
    text = re.sub(r"\W", " ", text).lower()
    return " ".join([word for word in text.split() if word not in stop_words])


def extract_basic_details(text):
    name = text.strip().split("\n")[0]
    email = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phone = re.findall(r'\b\d{10}\b|\+\d{1,2}\s?\d{10}', text)

    return {
        "Name": name.strip(),
        "Email": email[0] if email else "Not Found",
        "Phone": phone[0] if phone else "Not Found"
    }


def extract_entities(text):
    doc = nlp(text)
    skills, projects = [], []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            skills.append(ent.text.lower())
        elif ent.label_ == "WORK_OF_ART":
            projects.append(ent.text)
    return list(set(skills)), list(set(projects))


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


def suggest_skills(extracted_skills, job_role):
    required_skills = job_roles.get(job_role, "").lower().split(", ")
    return [skill for skill in required_skills if skill not in extracted_skills]


def generate_bar_chart(job_matches):
    plt.figure(figsize=(6, 4))
    sns.barplot(y=job_matches["Job Role"], x=job_matches["Match Score"], palette="coolwarm")
    plt.xlabel("Match Score")
    plt.ylabel("Job Role")
    plt.title("Job Role Match Scores")
    st.pyplot(plt)


def generate_pie_chart(score):
    labels = ['Matched', 'Unmatched']

    sizes = [score * 100, 100 - score * 100]
    colors = ['#4CAF50', '#FF6347']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)


def generate_reason(extracted_skills, required_skills_str, job_role):
    required_skills = [skill.strip().lower() for skill in required_skills_str.split(",")]
    matched = [skill for skill in required_skills if skill in extracted_skills]
    unmatched = [skill for skill in required_skills if skill not in extracted_skills]

    if matched and unmatched:
        reason = f"Based on your resume, you already have strengths in {', '.join(matched)}. However, to be a strong candidate for a {job_role}, consider gaining more experience in {', '.join(unmatched)}."
    elif matched:
        reason = f"You have all the core skills required for a {job_role}, such as {', '.join(matched)}. Great match!"
    elif unmatched:
        reason = f"To be suitable for a {job_role}, you should work on developing skills like {', '.join(unmatched)}."
    else:
        reason = f"No overlapping skills were detected for the {job_role}. Consider tailoring your resume or gaining experience in key areas."

    return reason


def resume_improvement_tips(resume_text, extracted_skills):
    skill_score = min(100, len(extracted_skills) * 10) if extracted_skills else 20

    cert_keywords = ['certified', 'certification', 'certificate', 'course']
    cert_score = 70 if any(kw in resume_text.lower() for kw in cert_keywords) else 40

    project_keywords = ['project', 'developed', 'built', 'created']
    project_score = 70 if any(kw in resume_text.lower() for kw in project_keywords) else 40

    format_score = 80 if ('•' in resume_text or '-' in resume_text[:300]) else 50

    all_required = ', '.join(job_roles.values()).lower()
    keyword_matches = [word for word in resume_text.lower().split() if word in all_required]
    keyword_score = min(100, len(set(keyword_matches)) * 5) if keyword_matches else 30

    labels = ['Skills', 'Projects', 'Certifications', 'Format', 'Keywords']
    scores = [skill_score, project_score, cert_score, format_score, keyword_score]

    st.subheader("📈 Resume Improvement Analysis")
    for tip in [
        "✅ Add more relevant skills based on job requirements.",
        "✅ Mention technical or academic projects clearly.",
        "✅ List certifications, courses, or training completed.",
        "✅ Use clean bullet-point format and consistent sections.",
        "✅ Include job-specific keywords from listings."
    ]:
        st.markdown(f"- {tip}")




# --------- STREAMLIT UI ---------

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer & Job Suggestion")
st.markdown("🚀 Upload your resume and get insights + job suggestions!")

uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.write("📜 **Extracted Resume Preview:**")
    st.text_area("", resume_text[:600], height=150)

    if st.button("🔍 Analyze Resume"):
        cleaned_resume = clean_text(resume_text)
        details = extract_basic_details(resume_text)
        extracted_skills, _ = extract_entities(resume_text)
        resume_embedding = get_bert_embedding(cleaned_resume)
        job_embeddings = [get_bert_embedding(skills) for skills in job_df["Required Skills"]]
        job_df["Match Score"] = [cosine_similarity(resume_embedding, je)[0][0] for je in job_embeddings]
        recommended_jobs = job_df.sort_values(by="Match Score", ascending=False).head(3)

        st.subheader("📌 Basic Resume Details")
        for key, value in details.items():
            st.markdown(f"- **{key}:** {value}")

        st.subheader("🛠 Extracted Skills")
        st.success(", ".join(extracted_skills) if extracted_skills else "No skills detected")

        best_match = recommended_jobs.iloc[0]
        st.markdown(
            f"### 🎯 Best Role: **{best_match['Job Role']}** with **{best_match['Match Score'] * 100:.2f}%** Match")
        generate_pie_chart(best_match["Match Score"])

        suggested_skills = suggest_skills(extracted_skills, best_match["Job Role"])
        if suggested_skills:
            st.warning("⚠️ Skills Missing for 100% Match:")
            st.write(", ".join(suggested_skills))

        st.subheader("🔝 Top 3 Suggested Job Roles with Reasons")

        for index, row in recommended_jobs.iterrows():
            reason = generate_reason(extracted_skills, row["Required Skills"], row["Job Role"])
            with st.container():
                st.markdown(f"""
                    <div style="border-left: 6px solid #4CAF50; background-color: #ffffff; padding: 15px 20px; border-radius: 10px; margin-bottom: 15px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
                        <h4 style="color:#000000; margin-bottom: 0;">🚀 {row['Job Role']}</h4>
                        <p style="margin: 0.5em 0; color:#000;"><strong>Match Score:</strong> <span style="color:#ff9933;">{row['Match Score'] * 100:.2f}%</span></p>
                        <p style="margin: 0.5em 0; color:#000;"><strong>Why this role?</strong><br>
                        <span style="white-space: pre-wrap; color:#000000;">{reason}</span></p>
                    </div>
                """, unsafe_allow_html=True)

        st.subheader("📊 Job Role Match Comparison")
        generate_bar_chart(recommended_jobs)

        # ✅ Fixed function call
        resume_improvement_tips(resume_text, extracted_skills)

# Sidebar Content
st.sidebar.title("🧠 How It Works")

st.sidebar.markdown("""
This app uses **AI and NLP** to analyze your resume and suggest the most suitable job roles.

---

### 📌 Steps:
1. **Upload your resume** (PDF format)  
2. **AI extracts** name, email, phone, and skills  
3. Compares your profile with roles using **BERT embeddings**  
4. Suggests **Top Job Roles** and missing skills  
5. Provides **Resume Improvement Tips**  

---

### 💼 Built With:
- **Python**, **Streamlit**
- **spaCy**, **BERT**, **Transformers**
- **pdfplumber**, **Pandas**, **Matplotlib**, **Seaborn**

---

### 💡 Tips:
- Include relevant **keywords** from job listings  
- Highlight **projects** and **certifications**  
- Use clean formatting with **bullet points**

---

Built with ❤️ by Md Muzammil Abdin
""")
