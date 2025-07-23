import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score


# Load the dataset
df = pd.read_csv('ai_assistant_usage_student_life.csv')

# Convert SessionDate to datetime (optional)
df['SessionDate'] = pd.to_datetime(df['SessionDate'])

# 1. Mean satisfaction across disciplines
discipline_satisfaction = df.groupby('Discipline')['SatisfactionRating'].mean().sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x=discipline_satisfaction.index, y=discipline_satisfaction.values, palette='viridis')
plt.xticks(rotation=45)
plt.title('Average Satisfaction Rating by Discipline')
plt.ylabel('Mean Satisfaction Rating')
plt.xlabel('Discipline')
plt.tight_layout()
plt.show()

# 2. Final outcome breakdown by task type
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='TaskType', hue='FinalOutcome', palette='Set2')
plt.title('Task Type vs Final Outcome')
plt.xlabel('Task Type')
plt.ylabel('Count')
plt.legend(title='Final Outcome')
plt.tight_layout()
plt.show()

# 3. ANOVA: Does satisfaction vary by discipline?
anova_model = ols('SatisfactionRating ~ C(Discipline)', data=df).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print("ANOVA Table:\n", anova_table)

# 4. Tukey’s HSD for pairwise comparison
tukey = pairwise_tukeyhsd(endog=df['SatisfactionRating'],
                          groups=df['Discipline'],
                          alpha=0.05)
print("\nTukey HSD Results:\n", tukey.summary())


# 5. Mean satisfaction rating per task type
task_satisfaction = df.groupby('TaskType')['SatisfactionRating'].mean().sort_values(ascending=False)
print("Mean Satisfaction by Task Type:\n", task_satisfaction)

# Bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x=task_satisfaction.index, y=task_satisfaction.values, palette='coolwarm')
plt.title("Average Satisfaction by Task Type")
plt.ylabel("Mean Satisfaction")
plt.xlabel("Task Type")
plt.tight_layout()
plt.show()

# 6. Outcome distribution across task types
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='TaskType', hue='FinalOutcome', palette='Set2')
plt.title("Final Outcome by Task Type")
plt.xlabel("Task Type")
plt.ylabel("Count")
plt.legend(title="Outcome")
plt.tight_layout()
plt.show()

# 7. ANOVA: Does satisfaction differ by TaskType?
anova_task = ols('SatisfactionRating ~ C(TaskType)', data=df).fit()
anova_result = sm.stats.anova_lm(anova_task, typ=2)
print("\nANOVA Table:\n", anova_result)

# 8. Tukey HSD: Pairwise comparison of satisfaction between task types
tukey_task = pairwise_tukeyhsd(endog=df['SatisfactionRating'],
                               groups=df['TaskType'],
                               alpha=0.05)
print("\nTukey HSD Results:\n", tukey_task.summary())


