import pandas as pd


import random
job_roles ={
    "data scientist": ["python", "pandas", "machine learning", "tensorflow", "keras", "sql", "statistics", "numpy", "nltk"],
    "data analyst": ["excel", "sql", "tableau", "power bi", "data visualization", "pandas"],
    "software engineer": ["c++", "java", "python", "git", "dsa", "system design"],
    "frontend developer": ["html", "css", "javascript", "react", "bootstrap"],
    "backend developer": ["python", "django", "flask", "node.js", "api", "sql"],
    "devops engineer": ["aws", "docker", "kubernetes", "linux", "jenkins"],
    "cloud engineer": ["azure", "aws", "gcp", "terraform", "cloud architecture"],
    "java developer": ["java", "spring", "hibernate", "maven", "rest api"],
    "android developer": ["java", "kotlin", "android studio", "xml", "firebase"],
    "qa engineer": ["manual testing", "automation testing", "selenium", "jira"],
}
data=[]
for role, skills in job_roles.items():
    for i in range(50):  


        selected = random.sample(skills, min(4, len(skills)))
        resume_text=(
            f"I have experience in {selected[0]} and {selected[1]}. "

            f"I've worked with {selected[2]} and completed projects using {selected[3] if len(selected) > 3 else selected[0]}. "
            f"My key skills include: {', '.join(selected)}."


        )


        data.append({"resume_text": resume_text, "role": role})
df=pd.DataFrame(data)
df.to_csv("resume_dataset.csv", index=False)


print("resume_dataset.csv created with", len(df), "rows.")
