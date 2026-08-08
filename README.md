# Skill-based Career Path Modeling and Recommendation

This repository contains the codebase and methodology for an **Early Intervention Career Guidance System**. It analyzes academic performance data (such as Ordinary Level exams) to discover latent "Aptitude Profiles" and recommends optimal career pathways.

The theoretical framework is heavily inspired by the **Monotonic Nonlinear State-Space (MNSS)** model (Ghosh et al.), adapting its advanced longitudinal deep learning concepts (State-Space Models, Variational Inference, and Skill Gap Optimization) to educational data.

## 📂 Project Structure

### `DATA/`
Contains the raw data and the extraction scripts.
* `analyze_data.py` & `combine_csv.py`: Scripts used to evaluate schema drift across yearly datasets.
* *Note: The raw CSVs and the final 800,000+ row `cleaned_skill_profiles.csv` are excluded via `.gitignore` to prevent repository bloat.*

### `datamanupulation/`
The core Artificial Intelligence and Machine Learning pipeline.
* **`data_cleaning_pipeline.ipynb`**: The ETL pipeline. Handles schema standardization, Min-Max score normalization, and engineers quantitative/scientific skill composites. (Run this first to generate the dataset).
* **`career_recommender_model.ipynb`**: The cross-sectional recommendation engine. Uses K-Means clustering and the Elbow Method to discover natural student profiles, mapping them to career families.
* **`mnss_model_architecture.ipynb`**: The advanced Deep Learning PyTorch implementation. Contains the Monotonic GRU cell, ELBO loss function, and Projected Gradient Descent Skill Gap Optimizer for longitudinal career tracking.
* **`data_storytelling_visualizations.ipynb`**: Generates simple, intuitive presentation materials (Radar charts, Bar charts, Skill Gap analysis) for non-technical audiences.

### `report/`
Contains the formal thesis documentation.
* **`report.tex`**: A comprehensive LaTeX report detailing the data engineering process, the K-Means mathematics, and a deep theoretical literature review of the MNSS state-space mathematical framework.

## 🚀 How to Use

1. Ensure your raw datasets are placed in the `DATA/` folder.
2. Run **`data_cleaning_pipeline.ipynb`** top-to-bottom. It will automatically clean the data and generate `cleaned_skill_profiles.csv`.
3. You can then run either **`career_recommender_model.ipynb`** to train the clustering model or **`data_storytelling_visualizations.ipynb`** to generate charts for your presentation.
4. To view the advanced longitudinal architecture, refer to the PyTorch code in **`mnss_model_architecture.ipynb`**.
