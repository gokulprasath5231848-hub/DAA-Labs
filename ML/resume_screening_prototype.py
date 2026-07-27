"""
AI Powered Resume Screening and Ranking System — Simple Working Prototype
Review-II (25% Implementation)

What this demonstrates:
1. Loading resume dataset + job description
2. Baseline matching using plain TF-IDF (lexical / keyword-only)
3. Hybrid matching using synonym-expanded TF-IDF (semantic-aware) — proxy
   for the Sentence-Transformer stage, showing WHY hybrid > keyword-only
4. Explainable AI layer: matched skills vs missing skills per candidate
5. Final ranked output (the "novelty" comparison table)
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------
# 1. Master skills list + synonym map (simulates semantic understanding
#    without needing a downloaded embedding model — good enough for a
#    Review-II working prototype)
# ---------------------------------------------------------------------
SKILL_SYNONYMS = {
    "python": ["python"],
    "machine learning": ["machine learning", "ml"],
    "natural language processing": ["natural language processing", "nlp"],
    "artificial intelligence": ["artificial intelligence", "ai"],
    "deep learning": ["deep learning", "dl"],
    "sql": ["sql", "structured query language"],
    "data analysis": ["data analysis", "data analytics"],
    "cloud": ["cloud", "aws", "azure"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "text classification": ["text classification"],
}

JOB_REQUIRED_SKILLS = [
    "python", "machine learning", "natural language processing",
    "artificial intelligence", "data analysis", "sql", "cloud",
    "scikit-learn", "tensorflow", "pytorch", "text classification"
]


def normalize_text(text):
    """Replace synonyms with a canonical skill token so TF-IDF treats
    'ML' and 'Machine Learning' as the same term — this is the
    hybrid/semantic-aware step."""
    text_lower = text.lower()
    for canonical, synonyms in SKILL_SYNONYMS.items():
        for syn in synonyms:
            text_lower = re.sub(r"\b" + re.escape(syn) + r"\b",
                                 canonical.replace(" ", "_"), text_lower)
    return text_lower


def extract_matched_skills(resume_text):
    text_lower = resume_text.lower()
    matched, missing = [], []
    for skill in JOB_REQUIRED_SKILLS:
        synonyms = SKILL_SYNONYMS.get(skill, [skill])
        found = any(re.search(r"\b" + re.escape(s) + r"\b", text_lower) for s in synonyms)
        (matched if found else missing).append(skill)
    return matched, missing


def main():
    resumes = pd.read_csv("resumes.csv")
    with open("job_description.txt") as f:
        job_desc = f.read()

    # ---------------- Baseline: plain TF-IDF (keyword-only) ----------------
    corpus_plain = [job_desc] + resumes["resume_text"].tolist()
    tfidf_plain = TfidfVectorizer(stop_words="english")
    matrix_plain = tfidf_plain.fit_transform(corpus_plain)
    baseline_scores = cosine_similarity(matrix_plain[0:1], matrix_plain[1:]).flatten()

    # ---------------- Hybrid: synonym-normalized TF-IDF ----------------
    corpus_hybrid = [normalize_text(job_desc)] + [normalize_text(t) for t in resumes["resume_text"]]
    tfidf_hybrid = TfidfVectorizer(stop_words="english")
    matrix_hybrid = tfidf_hybrid.fit_transform(corpus_hybrid)
    hybrid_scores = cosine_similarity(matrix_hybrid[0:1], matrix_hybrid[1:]).flatten()

    results = []
    for i, row in resumes.iterrows():
        matched, missing = extract_matched_skills(row["resume_text"])
        results.append({
            "Candidate": row["candidate_name"],
            "Baseline_TFIDF_%": round(baseline_scores[i] * 100, 1),
            "Hybrid_Match_%": round(hybrid_scores[i] * 100, 1),
            "Matched_Skills": ", ".join(matched) if matched else "None",
            "Missing_Skills": ", ".join(missing) if missing else "None",
        })

    results_df = pd.DataFrame(results).sort_values("Hybrid_Match_%", ascending=False)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))
    results_df.to_csv("results.csv", index=False)

    # ---------------- Detailed, explained per-candidate report ----------------
    print("\n" + "=" * 70)
    print(" AI POWERED RESUME SCREENING AND RANKING SYSTEM — DETAILED REPORT")
    print("=" * 70)
    print(f"\nJob Description Loaded: {len(job_desc.split())} words")
    print(f"Resumes Screened: {len(resumes)}")
    print(f"Required Skills for this Role ({len(JOB_REQUIRED_SKILLS)}): "
          f"{', '.join(JOB_REQUIRED_SKILLS)}")

    # sort candidates for the detailed section in the same rank order as results_df
    ranked_names = results_df["Candidate"].tolist()
    resumes_indexed = resumes.set_index("candidate_name")

    for rank, name in enumerate(ranked_names, start=1):
        idx = resumes[resumes["candidate_name"] == name].index[0]
        matched, missing = extract_matched_skills(resumes_indexed.loc[name, "resume_text"])
        base_pct = baseline_scores[idx] * 100
        hybrid_pct = hybrid_scores[idx] * 100
        diff = hybrid_pct - base_pct

        print("\n" + "-" * 70)
        print(f" RANK #{rank} — {name}")
        print("-" * 70)
        print(f" Baseline (TF-IDF only) Match Score : {base_pct:5.1f}%")
        print(f" Hybrid (Synonym-Aware) Match Score : {hybrid_pct:5.1f}%")
        if diff > 0.01:
            print(f" Score Change                       : +{diff:.1f}% "
                  f"(hybrid model caught synonym matches TF-IDF alone missed)")
        elif diff < -0.01:
            print(f" Score Change                       : {diff:.1f}% "
                  f"(fewer exact keyword overlaps than synonym-normalized terms diluted)")
        else:
            print(f" Score Change                       : no change (no synonym gaps detected)")

        print(f"\n Matched Skills ({len(matched)}/{len(JOB_REQUIRED_SKILLS)}):")
        print("   " + (", ".join(matched) if matched else "None"))

        print(f"\n Missing Skills ({len(missing)}/{len(JOB_REQUIRED_SKILLS)}):")
        print("   " + (", ".join(missing) if missing else "None"))

        # simple explainable verdict
        coverage = len(matched) / len(JOB_REQUIRED_SKILLS)
        if coverage >= 0.6:
            verdict = "STRONG MATCH — recommended for shortlisting"
        elif coverage >= 0.3:
            verdict = "MODERATE MATCH — may be considered with training/upskilling"
        else:
            verdict = "WEAK MATCH — does not meet most core requirements"
        print(f"\n Verdict: {verdict}")

    # ---------------- Summary table ----------------
    print("\n" + "=" * 70)
    print(" SUMMARY — Ranked Candidates (Hybrid Model)")
    print("=" * 70 + "\n")
    print(results_df.to_string(index=False))

    # ---------------- Line-by-line score comparison ----------------
    print("\n" + "=" * 70)
    print(" LINE-BY-LINE SCORE COMPARISON (Baseline vs Hybrid)")
    print("=" * 70 + "\n")
    print(f"{'Candidate':<15}{'Baseline %':>12}{'Hybrid %':>12}{'Change':>10}   Bar (Hybrid %)")
    print("-" * 70)
    for _, r in results_df.iterrows():
        name = r["Candidate"]
        base = r["Baseline_TFIDF_%"]
        hyb = r["Hybrid_Match_%"]
        change = hyb - base
        change_str = f"{'+' if change >= 0 else ''}{change:.1f}%"
        bar = "#" * int(round(hyb / 2))  # 1 block per 2%
        print(f"{name:<15}{base:>11.1f}%{hyb:>11.1f}%{change_str:>10}   {bar}")

    # ---------------- Novelty demonstration ----------------
    print("\n" + "=" * 70)
    print(" WHY HYBRID > BASELINE (Novelty Demonstration)")
    print("=" * 70)
    print(" Plain TF-IDF only matches exact keywords. Our hybrid model")
    print(" normalizes synonyms (e.g. 'ML' <-> 'Machine Learning') BEFORE")
    print(" vectorizing, so semantically equivalent terms are correctly")
    print(" recognized as matches. This is demonstrated below:\n")
    for i, row in resumes.iterrows():
        diff = hybrid_scores[i] - baseline_scores[i]
        if diff > 0.01:
            print(f" - {row['candidate_name']}: Baseline TF-IDF={baseline_scores[i]*100:.1f}% "
                  f"-> Hybrid={hybrid_scores[i]*100:.1f}%  (caught synonym matches like ML/AI/NLP)")

    print("\nFull results also saved to results.csv\n")


if __name__ == "__main__":
    main()