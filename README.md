# ML Data Cleaning Task

This repository contains the implementation of data cleaning and preprocessing steps performed on the Titanic dataset as part of an internship assignment.

---

## 📁 Repository Structure

```
ml-data-cleaning-task/
├── data_preprocessing.py
├── data_preprocessing_part2.py
├── data_preprocessing_part3.py
├── data_preprocessing_part4.py
├── titanic.csv
├── titanic_cleaned.csv
├── titanic_preprocessing.ipynb
├── encoding_results.txt
├── exploration_results.txt
├── outlier_analysis_results.txt
├── scaling_results.txt
├── todo.md
└── README.md
```

---

## 📊 Dataset

- **Source**: [Kaggle - Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **Files**:
  - `titanic.csv`: Original dataset.
  - `titanic_cleaned.csv`: Dataset after preprocessing.

---

## 🛠️ Preprocessing Steps

1. **Data Exploration**
   - Analyzed data types, null values, and statistical summaries.
   - Visualized distributions and relationships using plots.

2. **Handling Missing Values**
   - Imputed missing numerical values with mean or median.
   - Filled missing categorical values with mode or a placeholder.

3. **Encoding Categorical Variables**
   - Applied One-Hot Encoding to variables like `Sex` and `Embarked`.
   - Utilized Label Encoding where appropriate.

4. **Feature Scaling**
   - Normalized numerical features using Min-Max Scaling.
   - Standardized features using Z-score normalization.

5. **Outlier Detection and Removal**
   - Identified outliers using IQR and Z-score methods.
   - Removed or capped outliers to mitigate their impact.

---

## 📈 Results

- **Cleaned Dataset**: `titanic_cleaned.csv` is ready for machine learning modeling.
- **Analysis Reports**:
  - `encoding_results.txt`: Details on encoding transformations.
  - `exploration_results.txt`: Insights from initial data exploration.
  - `outlier_analysis_results.txt`: Findings from outlier detection.
  - `scaling_results.txt`: Summary of feature scaling effects.

---

## 📌 Additional Notes

- The preprocessing steps are modularized across multiple Python scripts for clarity.
- The Jupyter notebook `titanic_preprocessing.ipynb` provides an interactive walkthrough of the entire process.
- `todo.md` outlines pending tasks and future improvements.

---

## 👤 Author

- **Name**: Chaitanya
- **Role**: 3rd Year B.Tech Student
- **Interests**: AI, Cybersecurity, Coding, Startups, Internships, Hackathons

---

## 📎 Repository Link

Access the full project here: [ml-data-cleaning-task](https://github.com/Ncn914491/ml-data-cleaning-task)

---

Feel free to explore the repository and provide feedback or suggestions.
