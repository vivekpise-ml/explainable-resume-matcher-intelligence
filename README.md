# explainable-resume-matcher-nlp
This is for the NLP mini project 

Resume–JD Matcher with Explainability
Overview
This project matches resumes with job descriptions using:
•	NLP-based skill extraction
•	Transformer-based semantic matching
•	Skill gap analysis
•	Explainability using structured features
Features
•	Hybrid skill extraction (dictionary + NER)
•	Technical vs soft skill separation
•	Skill gap identification
•	Structured feature matching (experience, location, etc.)
Project Structure
•	src/ → core modules
•	data/ → datasets and annotations
•	main.py → entry point
Setup
pip install -r requirements.txt
Run
python main.py


------------------------

Project structure looks like this

project/
│
├── src/
│   ├── data_loader.py
│   ├── skill_extraction.py
│   ├── feature_extraction.py
│   ├── generate_skill_dict.py
│   ├── inference.py
│
├── data/
│   ├── raw/                  ← DO NOT push heavy files
│   ├── annotations/
│   │   ├── skill_dict.json
│   │   ├── skill_graph.json (later)
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore


-------------------

Things to do now.


First — Big Picture (VERY IMPORTANT)
We have 4 types of data/tasks:
1. Raw Data → JD + Resumes
2. Skill Knowledge → skill_dict.json (+ skill_graph.json)
3. Training Labels → match scores (Data.csv)
4. Structured Features → extracted from Data.csv
________________________________________
What We need to do (clear breakdown)
________________________________________
1: Resume + JD (RAW DATA)
- As discussed in last meeting Ashish and we all will populate the JD and resumes according to the format below:
data/raw/
   JD_1/
      jd.pdf
      resume1.pdf
      resume2.pdf
Our task:
We Ensure:
•	JD is clean (clear requirements)
•	Resumes are readable (PDF/DOCX extraction works)
No annotation needed here
________________________________________
2: Skill Dictionary (skill_dict.json - Have a look at this which is already generated)
We have already generated it (initially automatically using generate_skill_dict.py and stored in data/annotations/skill_dict.json)
Now We must:
Manually work on skill_dict.json (VERY IMPORTANT)
Open:
data/annotations/skill_dict.json
Step A — CLEAN
Remove junk if they are there, else it is fine. Or else, we can add more skills in Data.sv (which Ashish had created and use the same automated script namely python src/generate_skill_dict.py to create the updated skill_dict.json):
"data": ❌
"model": ❌
"work": ❌
________________________________________
Step B — ADD important skills if anythiong more needs to be added
As an Example --- Add manually in that skill_dict.json file:
"python": "TECH_SKILL",
"machine learning": "TECH_SKILL",
"deep learning": "TECH_SKILL",
"tensorflow": "TECH_SKILL",
"pytorch": "TECH_SKILL",
"aws": "TECH_SKILL",
"docker": "TECH_SKILL",

"communication": "SOFT_SKILL",
"teamwork": "SOFT_SKILL",
"leadership": "SOFT_SKILL"
________________________________________
Target
I felt that ~50–120 clean skills is enough or may be whatever we have could be also enough with some more little additions - This is for Ashish, and Swapnesh to review and update. I will also add some more depending on the JDs we get from Ashish
________________________________________
3: Skill Graph (skill_graph.json) — NEW (IMPORTANT)
- Create:
data/annotations/skill_graph.json
________________________________________
Manual creation
Example:
{
  "machine learning": ["deep learning"],
  "deep learning": ["cnn", "rnn", "lstm"],
  "deep learning frameworks": ["tensorflow", "pytorch"],
  "cloud": ["aws", "azure"]
}
________________________________________
Purpose of this :
Handle cases like:
JD → deep learning
Resume → CNN

Above example will still be considered match
________________________________________
4: Data.csv (MOST IMPORTANT PART)
- This is our training dataset
________________________________________
I felt this is work in progress on how to assign the weightage to matching score. but initially I did was to have 0, 1, 2 --- -where 0 - Poorly matched, 1 - Moderately matched, 2 - Good match (VERY IMPORTANT)

Hence as below
________________________________________
Use classification labels
0 → Poor
1 → Moderate
2 → Good
________________________________________
What We should do
________________________________________
Step 1 — For each JD
Open JD:
JD_1 → requires Python, DL, AWS
________________________________________
Step 2 — For each resume in that folder
Check:
Resume	Match?
Resume1	YES
Resume2	NO
________________________________________
Step 3 — Update Data.csv
Example:
JD	Resume	Label	Remark
JD_1	resume1	1	Has Python & DL
JD_1	resume2	0	Missing DL
________________________________________
Use "Remark" for explainability
Example:
"Has Python but missing deep learning"
VERY IMPORTANT for our project for explainability
________________________________________
- As THIS IS OUR TRAINING DATA
Used for:
Transformer training
________________________________________
Feature usage (Data.csv columns)
We already have:
experience
location
qualification
tenure
________________________________________
We do NOT manually change these
Just ensure they are correct
They will be used in feature extraction
________________________________________
Final Data Workflow (VERY IMPORTANT)
RAW DATA
(JD + Resume)
        ↓
create_pairs()
        ↓
Manual labeling (Data.csv)
        ↓
Training dataset
        ↓
Transformer model
        ↓
Inference
________________________________________
- Hence I feel we shouldnt have eventually too too many labels
- We should write remarks properly. Each one of us should go over the JD and the resumes we have and update the remarks properly --- This is Very Important
- Though automatically initially skill_dict.json is generated using generate_skill_dict.py using Data.csv, we should also update more mannually on this with JDs we see and also some of the common skill sets matching those JDs and update it in skill_dict.json
- we should all together take part of JDs and their respective Resumes and update the skill_graph.json like below
Manual creation
Example:
{
  "machine learning": ["deep learning"],
  "deep learning": ["cnn", "rnn", "lstm"],
  "deep learning frameworks": ["tensorflow", "pytorch"],
  "cloud": ["aws", "azure"]
}
- But I fill we keep it simple or in other words dont make it overcomplicated

________________________________________
To summarise
We need to do:
________________________________________
Resume + JD
•	keep as is
And add more JDs and their respective resumes
________________________________________
skill_dict.json
•	auto-generate + manually clean + add skills
________________________________________
skill_graph.json
•	manually create small relationship map
________________________________________
Data.csv
•	manually label:
o	match (0/1/2)
o	remarks (VERY IMPORTANT)


_____________________________________________

What I have done:
- Initial code made available
- automatically created skill_dict.json (using python src/generate_skill.dict.py)
- Have not compiled and run the entire workflow from main.py -- -This will be done after cleaning and updating the Data.csv, skill_dict.json, skill_graph.json, and adding resumes in data/raw folder by runnnig main.py. Folder I have not created yet - namely data/raw
- data/raw, *.pdf, *.docx are kept in .gitignore.
- JDs and resume, we will add the versioning using DVC - this also not done yet
________________________________________


For git cloning.

— Basic Git workflow

git clone https://github.com/vivekpise-ml/explainable-resume-matcher-nlp
git checkout -b feature-branch
git add .
git commit -m "feature added"
git push origin feature-branch



