# OMR Evaluation App

Automated system for evaluating multiple-choice answer sheets. Detects filled bubbles, 
compares with answer key, calculates score, generates CSV, and produces annotated images. 
Streamlit frontend provides interactive evaluation.

---

## Project Overview

- Detect filled bubbles automatically  
- Compare answers with provided key  
- Calculate total score and question-wise correctness  
- Generate annotated OMR images  
- Export detailed CSV results  
- Interactive evaluation via Streamlit  

---

## Workflow of the Code

1. Load answer key from text file  
2. Preprocess image: grayscale → Gaussian blur → adaptive threshold → morphological closing  
3. Detect contours and filter bubbles based on size, aspect ratio, area, and circularity  
4. Sort bubbles into rows and columns corresponding to questions and options  
5. Measure filled pixels in each bubble; select highest intensity as answer  
6. Compare detected answers with key to calculate score  
7. Generate CSV with question, student answer, correct answer, and result  
8. Annotate image with question numbers and selected options  

---

## Streamlit Interface Features

- Upload OMR image and answer key  
- Run evaluation with one click  
- Display final score in styled card  
- View original and processed images side by side  
- Display question-wise results table (green = correct, red = wrong)  
- Download results CSV  

---

## Features Summary

- Fully automated bubble detection with adaptive filtering  
- Dynamic grouping of bubbles into questions and options  
- Accurate answer selection using pixel intensity  
- Score computation with question-wise evaluation  
- Annotated images for verification  
- Interactive frontend with image display, results table, and CSV download  
- Handles variations in bubble size, layout, and lighting  
- Modular backend pipeline for integration or batch processing  

---

## Project Structure

OMR-Evaluation-App
│
├── omr_pipeline.py  # Core OMR evaluation functions
├── app.py  # Streamlit frontend
├── answer_key.txt  # Sample answer key
├── README.md  # Project description and workflow
├── processed_omr.jpg  # Annotated OMR sheet output
└── omr_results.csv  # CSV file with evaluation results

