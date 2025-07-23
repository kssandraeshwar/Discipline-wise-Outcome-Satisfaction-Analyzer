AI Assistant Usage Analysis in Student Life
This project takes a deep dive into how students across various disciplines interact with AI assistants — whether for coding, writing, studying, or brainstorming. It uncovers patterns in satisfaction, explores statistical differences, and builds a machine learning model to predict whether students will return to the AI assistant again.

Objectives
Understand how satisfaction levels vary across disciplines and tasks

Use statistical tests like ANOVA and Tukey HSD to verify if the differences are meaningful

Build a predictive model to estimate whether a student will use the AI again, based on their session behavior

Dataset
File: ai_assistant_usage_student_life.csv
The dataset contains 10,000 records representing student-AI interactions. Each record includes:

Discipline, StudentLevel

TaskType (e.g., Coding, Studying, Writing)

SessionLengthMin, TotalPrompts

AI_AssistanceLevel, FinalOutcome

 SatisfactionRating (scale 1–5)

UsedAgain (boolean — will the student return?)

Visualizations
This project includes insightful visualizations to support interpretation:

Average Satisfaction Rating by Discipline

Average Satisfaction by Task Type

Final Outcome by Task Type

Feature Importance (from the predictive model)

These plots help visualize the consistency and variation in AI usage patterns.

Statistical Analysis
To assess if satisfaction varies meaningfully:

ANOVA revealed no significant differences in satisfaction across disciplines or task types

Tukey HSD confirmed that satisfaction is fairly consistent, regardless of task or background

Insight: Students across the board seem to engage similarly with the AI assistant — a sign of its broad utility.

Predictive Modeling
We also built a classification model to predict if a student will return to use the assistant.

Model Used: Random Forest Classifier

Target: UsedAgain

Accuracy: 72%

ROC AUC Score: 0.67

Top Influential Features:

SatisfactionRating

SessionLengthMin

FinalOutcome

This model helps identify what encourages students to rely on AI — valuable for product or academic tool developers.

Project Files
File	Description
Discipline-wise Outcome & Satisfaction Analyzer.py	Complete analysis & modeling script
ai_assistant_usage_student_life.csv	Dataset file
Feature Importance.png	Feature impact chart from the model
Average Satisfaction by Task Type.png	Task-based satisfaction analysis
Final Outcome by task type.png	Task vs outcome comparison
Average Satisfaction Rating by Discipline.png	Satisfaction across disciplines



How to Use This Project
Clone the repository to your local system

Run the Python script (.py) in Jupyter or VS Code

Explore the visualizations and predictions for insights


