import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from datetime import datetime

# Set visual style
plt.style.use('default')
sns.set_palette("husl")

# -----------------------------
# STEP 1: Load Data
# -----------------------------
try:
    df = pd.read_csv("C:/Users/utkar/Desktop/Anurag Yadav/Yadav/dice_com-job_us_sample.csv")
    print("✅ Data loaded successfully!")
    print(f"Found {len(df)} job postings!")
    
except FileNotFoundError:
    print("❌ File not found. Please check the file path.")
    exit()

# -----------------------------
# STEP 2: Initial Data Exploration
# -----------------------------
print("=" * 50)
print("DATA EXPLORATION")
print("=" * 50)

print(f"Shape of dataset: {df.shape}")
print(f"\nColumn names ({len(df.columns)}): {list(df.columns)}")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# FIXED: Handle the case where there are no numerical columns
print("\nSummary Stats for Categorical Columns:")
print(df.describe(include=['object']))

print("\nNo numerical columns found in this dataset!")

# -----------------------------
# STEP 3: Data Quality Assessment
# -----------------------------
print("=" * 50)
print("DATA QUALITY ASSESSMENT")
print("=" * 50)

# Missing values analysis
missing_data = pd.DataFrame({
    'Missing Values': df.isnull().sum(),
    'Percentage': (df.isnull().sum() / len(df)) * 100
}).sort_values('Percentage', ascending=False)

print("Missing values per column:")
print(missing_data[missing_data['Missing Values'] > 0])

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# -----------------------------
# STEP 4: Enhanced Data Cleaning
# -----------------------------
print("=" * 50)
print("DATA CLEANING")
print("=" * 50)

# Create a copy for cleaning
df_clean = df.copy()

# Handle missing values based on column type
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].dtype == 'object':
            df_clean[col].fillna('Not Specified', inplace=True)
        else:
            # For numerical columns, fill with median (if any exist)
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

print("Missing values after cleaning:")
print(df_clean.isnull().sum().sum(), "missing values remaining")

# -----------------------------
# STEP 5: Univariate Analysis
# -----------------------------
print("=" * 50)
print("UNIVARIATE ANALYSIS")
print("=" * 50)

# Top Job Titles
print("\nTop 15 Job Titles:")
top_titles = df_clean["jobtitle"].value_counts().head(15)
print(top_titles)

# Top Companies
print("\nTop 15 Companies:")
top_companies = df_clean["company"].value_counts().head(15)
print(top_companies)

# Top Locations
print("\nTop 15 Locations:")
top_locations = df_clean["joblocation_address"].value_counts().head(15)
print(top_locations)

# Visualizations for top job titles
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Horizontal bar chart
top_titles.sort_values().plot(kind='barh', ax=axes[0], color='skyblue')
axes[0].set_title('Top 15 Job Titles', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Count')

# Pie chart for top 10
top10_titles = df_clean["jobtitle"].value_counts().head(10)
axes[1].pie(top10_titles.values, labels=top10_titles.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Top 10 Job Titles Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Word cloud for job titles
text = ' '.join(title for title in df_clean['jobtitle'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Job Titles', fontsize=16, fontweight='bold')
plt.show()

# -----------------------------
# STEP 6: Bivariate/Multivariate Analysis
# -----------------------------
print("=" * 50)
print("BIVARIATE ANALYSIS")
print("=" * 50)

# Top Companies visualization
plt.figure(figsize=(12, 8))
top_companies.sort_values().plot(kind='barh', color='lightgreen')
plt.title('Top 15 Companies Posting Jobs', fontsize=14, fontweight='bold')
plt.xlabel('Number of Job Postings')
plt.tight_layout()
plt.show()

# Employment Type analysis
if 'employmenttype_jobstatus' in df_clean.columns:
    print("\nEmployment Types:")
    employment_types = df_clean["employmenttype_jobstatus"].value_counts()
    print(employment_types)
    
    plt.figure(figsize=(10, 6))
    employment_types.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Employment Types Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('')  # Hide y-label
    plt.tight_layout()
    plt.show()

# Skills analysis (let's look at the most common skills)
if 'skills' in df_clean.columns:
    print("\nLet's analyze the skills mentioned in job postings...")
    
    # Get all skills as one big text
    all_skills = ' '.join(df_clean['skills'].dropna().astype(str))
    
    # Count skills (simple approach)
    skills_list = all_skills.lower().split()
    from collections import Counter
    skill_counts = Counter(skills_list).most_common(20)
    
    print("\nTop 20 Skills Mentioned:")
    for skill, count in skill_counts:
        print(f"  {skill}: {count} times")
    
    # Plot top skills
    skills, counts = zip(*skill_counts)
    plt.figure(figsize=(12, 8))
    plt.barh(skills, counts, color='orange')
    plt.title('Top 20 Skills in Job Postings', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency')
    plt.gca().invert_yaxis()  # Display highest at top
    plt.tight_layout()
    plt.show()

# -----------------------------
# STEP 7: Save Cleaned Data
# -----------------------------
print("=" * 50)
print("SAVING RESULTS")
print("=" * 50)

# Save cleaned dataset
df_clean.to_csv("C:/Users/utkar/Desktop/Anurag Yadav/Yadav/clean_job_posts.csv", index=False)
print("✅ Cleaned data saved to 'clean_job_posts.csv'")

print("\n🎉 EDA process completed successfully!")
print("Check your folder for the cleaned CSV file and all the charts!")