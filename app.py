import streamlit as st
import cv2
import numpy as np
import pandas as pd
import csv

# =========================
# 🎨 THEME COLORS 
# =========================
PAGE_BG = "#93EB87C7"    
TITLE_COLOR = "#663399"    
BUTTON_COLOR = "#338B00DF"   
BUTTON_HOVER = "#000000"   
SCORE_BG = "#FFFFFF6C"    
SCORE_BORDER = "#00008B"   
SCORE_TEXT = "#663399"  

st.set_page_config(page_title="OMR Evaluation App", layout="centered")

# =========================
# 🎨 Custom CSS Styling
# =========================
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {PAGE_BG};
    }}

    h1 {{
        color: {TITLE_COLOR};
        text-align: center;
    }}

    div.stButton > button:first-child {{
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
        background-color: {BUTTON_COLOR};
        color: white;
        border: none;
    }}
    div.stButton > button:hover {{
        background-color: {BUTTON_HOVER};
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# 📝 Page Layout
# =========================
st.title("📝 OMR Evaluator")

uploaded_omr = st.file_uploader("📷 Upload OMR Sheet Image", type=["jpg", "png", "jpeg"])
uploaded_key = st.file_uploader("📝 Upload Answer Key (TXT)", type=["txt"])

# =========================
# 🚀 Processing Pipeline
# =========================
if st.button("Run OMR Evaluation") and uploaded_omr and uploaded_key:
    with open("temp_omr.jpg", "wb") as f:
        f.write(uploaded_omr.getbuffer())
    with open("answer_key.txt", "wb") as f:
        f.write(uploaded_key.getbuffer())

    with open("answer_key.txt", "r") as f:
        answer_key = [line.strip().upper() for line in f if line.strip()]

    image = cv2.imread("temp_omr.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_bubbles = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(c)
        if not (20 < w < 50 and 20 < h < 50): continue
        if not (0.8 <= aspect_ratio <= 1.2): continue
        if not (300 < area < 2000): continue
        circularity = 4 * np.pi * area / (cv2.arcLength(c, True) ** 2 + 1e-5)
        if circularity < 0.7: continue
        valid_bubbles.append((x, y, w, h))

    valid_bubbles = sorted(valid_bubbles, key=lambda b: b[1])
    questions = []
    for i in range(0, len(valid_bubbles), 4):
        row = valid_bubbles[i:i+4]
        row = sorted(row, key=lambda b: b[0])
        questions.append(row)

    selected_answers = []
    output_grouped = image.copy()
    for q_idx, row in enumerate(questions):
        bubble_values = []
        for opt_idx, (x, y, w, h) in enumerate(row):
            roi = thresh[y:y+h, x:x+w]
            total = cv2.countNonZero(roi)
            bubble_values.append(total)
            cx, cy = x + w // 2, y + h // 2
            cv2.putText(output_grouped, f"Q{q_idx+1}{chr(65+opt_idx)}",
                        (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(output_grouped, (x, y), (x + w, y + h), (255, 0, 0), 1)

        filled_index = np.argmax(bubble_values)
        selected_option = chr(65 + filled_index)
        selected_answers.append(selected_option)

    score = 0
    results = []
    for student_ans, correct_ans in zip(selected_answers, answer_key):
        if student_ans == correct_ans:
            score += 1
            results.append("Correct")
        else:
            results.append("Wrong")

    results_path = "omr_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Student Answer", "Correct Answer", "Result"])
        for i, (student_ans, correct_ans, res) in enumerate(zip(selected_answers, answer_key, results)):
            writer.writerow([f"Q{i+1}", student_ans, correct_ans, res])

    # Save session state
    st.session_state["score"] = score
    st.session_state["answer_len"] = len(answer_key)
    st.session_state["results_path"] = results_path
    st.session_state["original_image_bytes"] = uploaded_omr.getvalue()
    _, buffer = cv2.imencode(".jpg", output_grouped)
    st.session_state["processed_image_bytes"] = buffer.tobytes()

# =========================
# 🎯 Show Results
# =========================
if "score" in st.session_state:
    score = st.session_state["score"]
    total = st.session_state["answer_len"]

    st.markdown(
        f"""
        <div style="
            background-color:{SCORE_BG};
            padding:20px;
            border-radius:10px;
            border:2px solid {SCORE_BORDER};
            text-align:center;
            font-size:22px;
            color:{SCORE_TEXT};
            font-weight:bold;
        ">
        ✅ Final Score: <b>{score}/{total}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("📷 Show Original & Processed Images"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state["original_image_bytes"], caption="Original OMR", use_column_width=True)
        with col2:
            st.image(st.session_state["processed_image_bytes"], caption="Processed OMR", use_column_width=True)

    # ✅ Highlight correct/wrong answers
    def highlight_result(row):
        style = 'background-color: green; color: white; font-weight: bold;' if row["Result"] == "Correct" else 'background-color: red; color: white; font-weight: bold;'
        return [style] * len(row)

    if st.button("📑 Show Results Table"):
        df = pd.read_csv(st.session_state["results_path"])
        st.dataframe(df.style.apply(highlight_result, axis=1))
        with open(st.session_state["results_path"], "rb") as f:
            st.download_button("⬇️ Download Results CSV", f, file_name="omr_results.csv", mime="text/csv")
