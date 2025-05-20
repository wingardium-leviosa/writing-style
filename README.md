# ICLR Writing Style Analysis

This project aims to help researchers receive features feedback on their academic writing style by analyzing the language used in different sections of their papers (e.g., Abstract, Introduction, Related Work, Methods, Results, Conclusion). Inspired by prior ICLR submissions and reviews, we extract linguistic, syntactic, and stylistic features from papers and use trained models to predict writing effectiveness. Users can interactively input their text into a web interface and receive automated analysis based on previous accepted ICLR papers.

---

## Project Overview

This is a full-stack application consisting of:

- A **Python backend** for feature extraction, modeling, and data processing.
- A **React + Vite frontend** for user interaction and visualization.
- Data pipelines to ingest and clean historical ICLR paper content and reviews.
- Feature extraction modules that compute over a dozen writing style metrics.

---

## Folder and File Structure

```plaintext
.
├── README.md                  # Project documentation and usage guide
├── data_raw                   # Raw OpenReview paper/review data
├── feature_extraction         # Extracted features & scripts for syntactic/semantic metrics
├── Merged_model               # Model training scripts and evaluation notebooks
├── Frontend                   # Web app frontend (React + Vite)
```
