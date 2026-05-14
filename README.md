# basic-details-of-indian-schools
"""Analyze the distribution of indian schools


Original file is located at
https://colab.research.google.com/drive/1UZImM9TNlXgW5Ds6nXRSj8DGtwHSWVUZ?usp=sharing
""""


# Library importing
# Data manipulation and analysis
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib settings
plt.figure(figsize=(8, 5))

# seaborn settings
sns.set_theme(style="whitegrid")

# Getting data from google drive
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/AI Driven DA/Final project file/basic-details-of-schools.csv')
df.head()

# Data overview rows and columns
df.shape

# Date overview
df.info()

# Available columns
df.columns

# Missing value analysis
missing_df = (
    df.isnull()
      .sum()
      .to_frame(name='missing_count')
      .assign(missing_percentage=lambda x: (x['missing_count'] / len(df)) * 100)
      .sort_values(by='missing_percentage', ascending=False)
)

missing_df

# Checking data types
df.dtypes

# Data overview
df.describe()

# Data overview All columns and rows
df.describe(include='all')

# Unwanted columns
drop_columns = [
    'id',
    'state_code',
    'district_code',
    'subdistrict_code',
    'udise_village_code',
    'ward',
    'cluster_name'
]


# Dropping unwanted columns
cleaned_df = df.drop(columns=drop_columns)

# Checking updated values
cleaned_df.shape

# Creating a copy will help backup
cleaned_df_copy = cleaned_df.copy()


# Selecting Specific columns for analysis
selected_columns = [
    # Location
    'state_name',
    'district_name',
    'location_type',

    # School identity & type
    'school_name',
    'udise_school_code',
    'school_category',
    'school_type',
    'management',
    'year_of_establishment',
    'pre_primary',
    'status',

    # Infrastructure
    'class_rooms',
    'other_rooms',

    # Teachers
    'total_teachers',

    # Student enrollment (by level)
    'pre_primary_students',
    'i_students', 'ii_students', 'iii_students', 'iv_students', 'v_students',
    'vi_students', 'vii_students', 'viii_students',
    'ix_students', 'x_students', 'xi_students', 'xii_students',

    # Aggregates
    'class_students',
    'class_with_pre_primary_students'
]

# Selected Column
cleaned_df = cleaned_df[selected_columns]

# Optimise low-cardinality string columns using categorical dtype for improved memory efficiency and analytical performance.
categorical_cols = [
    'state_name',
    'district_name',
    'location_type',
    'school_category',
    'school_type',
    'management',
    'status'
]

for col in categorical_cols:
    cleaned_df[col] = cleaned_df[col].astype('category')


# Enrollment and infrastructure count variables to integer types to reflect their discrete nature and reduce memory footprint.
count_cols = [
    'pre_primary',
    'class_rooms',
    'other_rooms',
    'total_teachers',
    'pre_primary_students',
    'i_students', 'ii_students', 'iii_students', 'iv_students', 'v_students',
    'vi_students', 'vii_students', 'viii_students',
    'ix_students', 'x_students', 'xi_students', 'xii_students',
    'class_students',
    'class_with_pre_primary_students'
]

for col in count_cols:
    cleaned_df[col] = cleaned_df[col].astype('int32')

# Downcasting year_of_establishment to int16 to optimise memory since year values fall within a small numeric range.
cleaned_df['year_of_establishment'] = cleaned_df['year_of_establishment'].astype('int16')

# Finding missing values
cleaned_df.isna().sum()

# Imputing missing values with the most repeated value location type
for col in ['location_type']:
    mode_value = cleaned_df[col].mode()[0]
    cleaned_df[col] = cleaned_df[col].fillna(mode_value)


# standardising school category naming
category_mapping = {
    'Primary': 'Primary',
    'Primary with Upper Primary': 'Primary_UpperPrimary',
    'Upper Primary only': 'UpperPrimary_Only',
    'Pri. Upper Pri. and Secondary Only': 'Primary_UpperPrimary_Secondary',
    'Pri. with Upper Pri. Sec. and H.Sec.': 'Primary_to_HigherSecondary',
    'Upper Pri. and Secondary': 'UpperPrimary_Secondary',
    'Upper Pri. Secondary and Higher Sec': 'UpperPrimary_to_HigherSecondary',
    'Secondary Only': 'Secondary_Only',
    'Secondary with Higher Secondary': 'Secondary_HigherSecondary',
    'Higher Secondary only/Jr. College': 'HigherSecondary_Only'
}

cleaned_df['school_category_std'] = cleaned_df['school_category'].map(category_mapping)

# Total students calculation
student_cols = [
    'pre_primary_students',
    'i_students', 'ii_students', 'iii_students', 'iv_students', 'v_students',
    'vi_students', 'vii_students', 'viii_students',
    'ix_students', 'x_students', 'xi_students', 'xii_students'
    ]

cleaned_df['total_students'] = cleaned_df[student_cols].sum(axis=1)


cleaned_df['total_students']


# Student-Teacher Ratio calculation
cleaned_df['student_teacher_ratio'] = (
    cleaned_df['total_students'] /
    cleaned_df['total_teachers'].replace(0, np.nan)
).round(1)

cleaned_df['student_teacher_ratio']


# Classroom utilisation analysis
cleaned_df['classroom_utilization'] = (
    cleaned_df['total_students'] /
    cleaned_df['class_rooms'].replace(0, np.nan)
).round(1)

cleaned_df['classroom_utilization']


# Finding school-age
CURRENT_YEAR = 2026
cleaned_df['school_age'] = CURRENT_YEAR - cleaned_df['year_of_establishment']
cleaned_df['school_age']


# Finding education level span
level_span_map = {
    'Primary': 1,
    'Primary_UpperPrimary': 2,
    'UpperPrimary_Only': 2,
    'Primary_UpperPrimary_Secondary': 3,
    'UpperPrimary_Secondary': 3,
    'Secondary_Only': 3,
    'Secondary_HigherSecondary': 4,
    'Primary_to_HigherSecondary': 4,
    'UpperPrimary_to_HigherSecondary': 4,
    'HigherSecondary_Only': 4
}

cleaned_df['education_level_span'] = cleaned_df['school_category_std'].map(level_span_map)

cleaned_df['education_level_span']


# schools with no teachers
cleaned_df['no_teacher_flag'] = (cleaned_df['total_teachers'] == 0).astype(int)

cleaned_df['no_teacher_flag']

# checking calculated columns
cleaned_df[
    [
        'total_students',
        'student_teacher_ratio',
        'classroom_utilization',
        'school_age',
        'education_level_span'
    ]
].describe()

# student teacher ratio overview
cleaned_df['student_teacher_ratio'].dropna().describe()

# class room utilization overview
cleaned_df['classroom_utilization'].dropna().describe()

# category based school's
cleaned_df.groupby('school_category_std')[
    ['student_teacher_ratio', 'classroom_utilization']
].mean().sort_values('student_teacher_ratio', ascending=False)


# management types
cleaned_df.groupby('management')[
    ['student_teacher_ratio', 'classroom_utilization']
].mean().sort_values('student_teacher_ratio', ascending=False)


# top 10 states by Student–Teacher Ratio
cleaned_df.groupby('state_name')['student_teacher_ratio'].mean().sort_values(ascending=False).head(10)


# bottom 10 states by Student–Teacher Ratio
cleaned_df.groupby('state_name')['student_teacher_ratio'].mean().sort_values().head(10)

# School Age vs Enrollment
cleaned_df[['school_age', 'total_students']].corr()

# Classrooms vs Students
cleaned_df[['class_rooms', 'total_students']].corr()


# schools with teachers and non-teachers
cleaned_df['no_teacher_flag'].value_counts()


# School counts by state
state_school_counts = cleaned_df['state_name'].value_counts()

top_10_states = state_school_counts.head(10)

plt.figure(figsize=(10, 6))
top_10_states.sort_values().plot(kind='barh')
plt.title('Top 10 States by Number of Schools')
plt.xlabel('Number of Schools')
plt.ylabel('State')
plt.show()


# Distribution of School Category
plt.figure(figsize=(10, 6))

school_category_counts = cleaned_df['school_category'].value_counts()

school_category_counts.plot(kind='bar')

plt.title('Distribution of School Categories')
plt.xlabel('School Category')
plt.ylabel('Number of Schools')
plt.xticks(rotation=45, ha='right')
plt.show()


# Rural vs Urban schools
location_counts = cleaned_df['location_type'].value_counts()

plt.figure(figsize=(6, 6))

location_counts.plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=90
)

plt.title('Percentage Distribution of Rural vs Urban Schools')
plt.ylabel('')
plt.show()


# School management type distribution
plt.figure(figsize=(10, 6))

management_counts = cleaned_df['management'].value_counts()

sns.barplot(
    x=management_counts.values,
    y=management_counts.index
)

plt.title('Distribution of Schools by Management Type')
plt.xlabel('Number of Schools')
plt.ylabel('Management Type')
plt.show()


# student teacher ratio rural vs urban
plt.figure(figsize=(8, 6))

sns.boxplot(
    data=cleaned_df,
    x='location_type',
    y='student_teacher_ratio'
)

plt.title('Student–Teacher Ratio: Rural vs Urban Schools')
plt.xlabel('Location Type')
plt.ylabel('Student–Teacher Ratio')
plt.show()


# state vs school category(top 10 state)

# filter to top 10 states
top_states = cleaned_df['state_name'].value_counts().head(10).index

# removing unused categories
filtered_df = cleaned_df[cleaned_df['state_name'].isin(top_states)].copy()

filtered_df['school_category_std'] = (
    filtered_df['school_category_std']
    .cat.remove_unused_categories()
)

state_category = (
    filtered_df
    .groupby(['state_name', 'school_category_std'], observed=True)
    .size()
    .unstack(fill_value=0)
)

# heatmap creation
plt.figure(figsize=(10, 5))

sns.heatmap(
    state_category,
    cmap='Blues',
    linewidths=0.5
)

plt.title('State vs School Category (Top 10 States)')
plt.xlabel('School Category')
plt.ylabel('State')

plt.tight_layout()
plt.show()


# Rural vs Urban – Student strength comparison
location_students_df = (
    cleaned_df
    .groupby('location_type')['total_students']
    .sum()
    .reset_index()
)

plt.figure(figsize=(6, 5))

sns.barplot(
    data=location_students_df,
    x='location_type',
    y='total_students'
)

plt.title('Total Student Strength: Rural vs Urban')
plt.xlabel('Location Type')
plt.ylabel('Total Number of Students')

plt.tight_layout()
plt.show()


# Location Type vs Student–Teacher Ratio
ratio_by_location = (
    cleaned_df.groupby('location_type')['student_teacher_ratio']
      .mean()
      .reset_index()
)

# Plot
plt.figure()
plt.bar(
    ratio_by_location['location_type'],
    ratio_by_location['student_teacher_ratio']
)
plt.title('Average Student–Teacher Ratio by Location Type')
plt.xlabel('Location Type')
plt.ylabel('Student–Teacher Ratio')
plt.show()


# State vs Average Students per School
state_avg_students = (
    cleaned_df
    .groupby('state_name')['total_students']
    .mean()
    .reset_index()
)

top10_states = state_avg_students.sort_values(
    by='total_students',
    ascending=False
).head(10)

plt.figure(figsize=(10, 5))

plt.barh(
    top10_states['state_name'],
    top10_states['total_students']
)

plt.title('Top 10 States by Average Students per School')
plt.xlabel('Average Number of Students per School')
plt.ylabel('State')

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# School Category vs Total Students
category_students_df = (
    cleaned_df
    .groupby(['education_level_span', 'school_category_std'])['total_students']
    .sum()
    .reset_index()
)

category_students_df = category_students_df.sort_values(
    'education_level_span'
)

plt.figure(figsize=(9, 6))

plt.bar(
    category_students_df['school_category_std'],
    category_students_df['total_students']
)

plt.title('Total Students by School Category (Ordered by Education Span)')
plt.xlabel('School Category')
plt.ylabel('Total Number of Students')

plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()


# Location Type vs Classroom Utilisation
location_utilization_df = (
    cleaned_df
    .groupby('location_type')['classroom_utilization']
    .median()
    .reset_index()
)

plt.figure(figsize=(6, 5))

plt.bar(
    location_utilization_df['location_type'],
    location_utilization_df['classroom_utilization']
)

plt.title('Classroom Utilization by Location Type')
plt.xlabel('Location Type')
plt.ylabel('Students per Classroom')

plt.tight_layout()
plt.show()


# School Category vs Classroom Utilisation
valid_df = cleaned_df[
    (cleaned_df['class_rooms'] > 0) &
    (cleaned_df['classroom_utilization'].notna())
]

category_util_df = (
    valid_df
    .groupby('school_category_std')['classroom_utilization']
    .mean()
    .reset_index()
)

plt.figure(figsize=(9, 5))

plt.bar(
    category_util_df['school_category_std'],
    category_util_df['classroom_utilization']
)

plt.title('Average Classroom Utilization by School Category')
plt.xlabel('School Category')
plt.ylabel('Average Students per Classroom')
plt.xticks(rotation=30, ha='right')

plt.tight_layout()
plt.show()


# Infrastructure vs Enrollment Balance

# distribution of students per classroom

plt.figure(figsize=(8, 5))
sns.histplot(
    cleaned_df['classroom_utilization'],
    bins=50,
    kde=True
)
plt.title('Distribution of Students per Classroom')
plt.xlabel('Students per Classroom')
plt.ylabel('Count')
plt.show()


# Category × Location
plt.figure(figsize=(11, 6))
sns.boxplot(
    data=cleaned_df,
    x='school_category_std',
    y='classroom_utilization',
    hue='location_type'
)
plt.title('Infrastructure vs Enrollment by Category and Location')
plt.xlabel('School Category')
plt.ylabel('Students per Classroom')
plt.xticks(rotation=30, ha='right')
plt.legend(title='Location')
plt.show()


# Infrastructure vs Enrollment Scatter
plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=cleaned_df.sample(10000),  # sampling for performance
    x='class_rooms',
    y='total_students',
    hue='location_type',
    alpha=0.4
)
plt.title('Infrastructure vs Enrollment Scatter')
plt.xlabel('Number of Classrooms')
plt.ylabel('Total Students')
plt.show()


# Correlation Heatmap
numeric_cols = [
    'total_students',
    'total_teachers',
    'class_rooms',
    'student_teacher_ratio',
    'classroom_utilization',
    'school_age'
]

corr = cleaned_df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()


















