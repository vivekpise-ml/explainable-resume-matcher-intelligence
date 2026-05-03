from src.data_loader import create_pairs
from src.skill_extraction import load_skill_dict
from src.inference import run_inference, load_ner_model, load_skill_graph

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------------
# Config
# -----------------------------
data_dir = "data/raw"
skill_dict_path = "data/annotations/skill_dict.json"
skill_graph_path = "data/annotations/skill_graph.json"
# model_name = "distilbert-base-uncased"
model_path = "models/matcher_model"

# -----------------------------
# Step 1: Load data
# -----------------------------
pairs = create_pairs(data_dir)
print("Total pairs:", len(pairs))


# -----------------------------
# Step 2: Load skill dictionary
# -----------------------------
skill_dict = load_skill_dict(skill_dict_path)


# -----------------------------
# Step 3: Load skill graph (NEW)
# -----------------------------
skill_graph = load_skill_graph(skill_graph_path)


# -----------------------------
# Step 4: Load NER model (optional)
# -----------------------------
ner_model = load_ner_model()


# -----------------------------
# Step 5: Load transformer model
# -----------------------------
# tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_path)
matcher_model = AutoModelForSequenceClassification.from_pretrained(
    #model_name,
    model_path,
    num_labels=2
)


# -----------------------------
# Step 6: Run inference (sample)
# -----------------------------
for i, pair in enumerate(pairs[:3]):

    resume_text = pair["resume_text"]
    jd_text = pair["jd_text"]

    result = run_inference(
        resume_text=resume_text,
        jd_text=jd_text,
        matcher_model=matcher_model,
        tokenizer=tokenizer,
        skill_dict=skill_dict,
        skill_graph=skill_graph,
        ner_model=ner_model,
        row_data=None
    )

    print(f"\n--- Result {i+1} ---")
    print("Match Score:", result["match_score"])

    print("Matched Skills:", result["skill_analysis"]["matched"])
    print("Missing Skills:", result["skill_analysis"]["missing"])

    # OPTIONAL (VERY USEFUL DEBUG)
    print("Graph Matches:", result["skill_analysis"].get("graph_matches", {}))