
import cv2
import numpy as np
import csv

def evaluate_omr(image_path,
                 answer_key_path="answer_key.txt",
                 csv_filename="omr_results.csv",
                 output_image_path="omr_marked.png"):
    """
    Runs the v5 pipeline on image_path using answer_key_path.
    Writes CSV to csv_filename (same format as original).
    Writes a marked output image to output_image_path.
    Returns: (score, total_questions, csv_filename, output_image_path)
    """

    # -------------------------
    # Step 0: Load answer key
    # -------------------------
    with open(answer_key_path, "r") as f:
        answer_key = [line.strip().upper() for line in f if line.strip()]

    # -------------------------
    # Step 1: Load image & preprocess
    # -------------------------
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"{image_path} not found in working directory.")

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

    # -------------------------
    # Step 2: Find contours (collect candidate bubble contours)
    # -------------------------
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        arc = cv2.arcLength(c, True)
        boxes.append((x, y, w, h, area, arc))

    if len(boxes) == 0:
        raise RuntimeError("No contours found. Check the image path or threshold parameters.")

    # -------------------------
    # Step 2a: Choose dynamic area threshold
    # -------------------------
    areas = np.array([b[4] for b in boxes])
    expected_bubbles = 400  # 100 questions * 4 options (used as a target for auto-selection)
    best = None
    for p in range(99, 49, -1):
        thr = np.percentile(areas, p)
        cand = [b for b in boxes if b[4] >= thr]
        cand = [b for b in cand if 5 <= b[2] <= 200 and 5 <= b[3] <= 200]
        cnt = len(cand)
        if best is None or abs(cnt - expected_bubbles) < abs(best[1] - expected_bubbles):
            best = (thr, cnt, p)
    area_thr = best[0]

    # -------------------------
    # Step 2b: Final filtering using that area threshold plus shape/circularity checks
    # -------------------------
    candidate_contours = [b for b in boxes if b[4] >= area_thr]
    ws = np.array([b[2] for b in candidate_contours]) if candidate_contours else np.array([30])
    hs = np.array([b[3] for b in candidate_contours]) if candidate_contours else np.array([30])
    areas_cand = np.array([b[4] for b in candidate_contours]) if candidate_contours else np.array([400.0])

    median_w = float(np.median(ws))
    median_h = float(np.median(hs))
    median_area = float(np.median(areas_cand))

    min_w = max(6, int(0.5 * median_w))
    max_w = int(1.8 * median_w)
    min_h = max(6, int(0.5 * median_h))
    max_h = int(1.8 * median_h)
    min_area = max(30, 0.35 * median_area)
    max_area = 3.0 * median_area

    valid_bubbles = []
    for (x, y, w, h, area, arc) in candidate_contours:
        aspect = w / float(h) if h > 0 else 0
        circularity = 4 * np.pi * area / (arc ** 2 + 1e-10)
        if not (min_w <= w <= max_w and min_h <= h <= max_h):
            continue
        if not (0.7 <= aspect <= 1.3):
            continue
        if not (min_area <= area <= max_area):
            continue
        if circularity < 0.55:
            continue
        valid_bubbles.append((x, y, w, h))

    # fallback
    if len(valid_bubbles) < 0.6 * expected_bubbles or len(valid_bubbles) > 1.6 * expected_bubbles:
        valid_bubbles = []
        for (x, y, w, h, area, arc) in boxes:
            aspect = w / float(h) if h > 0 else 0
            circularity = 4 * np.pi * area / (arc ** 2 + 1e-10)
            if not (int(0.5 * median_w) <= w <= int(1.8 * median_w)): continue
            if not (int(0.5 * median_h) <= h <= int(1.8 * median_h)): continue
            if not (0.65 <= aspect <= 1.35): continue
            if circularity < 0.5: continue
            valid_bubbles.append((x, y, w, h))

    valid_bubbles = sorted(valid_bubbles, key=lambda b: b[1])

    # Debug visualization saved later
    output = image.copy()
    for (x, y, w, h) in valid_bubbles:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # -------------------------
    # Step 3: Group bubbles into questions (4 per row)
    # -------------------------
    if len(valid_bubbles) == 0:
        raise RuntimeError("No valid bubbles found after filtering — adjust thresholds or check image.")

    centers_x = np.array([x + w / 2.0 for (x, y, w, h) in valid_bubbles])
    q25, q50, q75 = np.percentile(centers_x, [25, 50, 75])
    thresholds = [q25, q50, q75]

    cols = [[] for _ in range(4)]
    for (x, y, w, h) in valid_bubbles:
        cx = x + w / 2.0
        col_idx = int(np.searchsorted(thresholds, cx))
        cols[col_idx].append({'x': x, 'y': y, 'w': w, 'h': h, 'cx': cx, 'cy': y + h / 2.0})

    for i in range(4):
        cols[i] = sorted(cols[i], key=lambda r: r['cy'])

    median_w = int(np.median([c['w'] for c in (cols[0] + cols[1] + cols[2] + cols[3]) if c]))
    median_h = int(np.median([c['h'] for c in (cols[0] + cols[1] + cols[2] + cols[3]) if c]))

    col_median_x = []
    for i in range(4):
        if len(cols[i]) > 0:
            col_median_x.append(int(np.median([c['cx'] for c in cols[i]])))
        else:
            col_median_x.append(int((i + 0.5) * image.shape[1] / 4.0))

    num_rows = max(len(c) for c in cols)
    questions = []
    for r in range(num_rows):
        row_boxes = []
        present_ys = []
        for c in range(4):
            if r < len(cols[c]):
                present_ys.append(cols[c][r]['cy'])
        est_y_center = float(np.median(present_ys)) if present_ys else (r + 0.5) * median_h * 1.6
        for c in range(4):
            if r < len(cols[c]):
                cell = cols[c][r]
                row_boxes.append((int(cell['x']), int(cell['y']), int(cell['w']), int(cell['h'])))
            else:
                cx = col_median_x[c]
                x = max(0, int(cx - median_w // 2))
                y = max(0, int(est_y_center - median_h // 2))
                row_boxes.append((x, y, median_w, median_h))
        questions.append(row_boxes)

    output_grouped = image.copy()
    for q_idx, row in enumerate(questions):
        for opt_idx, (x, y, w, h) in enumerate(row):
            cx, cy = x + w // 2, y + h // 2
            cv2.putText(output_grouped, f"Q{q_idx+1}{chr(65+opt_idx)}",
                        (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(output_grouped, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # -------------------------
    # Step 4: Identify filled bubble in each question
    # -------------------------
    selected_answers = []
    for row in questions:
        bubble_values = []
        for (x, y, w, h) in row:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(thresh.shape[1], x + w)
            y2 = min(thresh.shape[0], y + h)
            roi = thresh[y1:y2, x1:x2]
            total = cv2.countNonZero(roi)
            bubble_values.append(total)

        filled_index = int(np.argmax(bubble_values))
        selected_option = chr(65 + filled_index)
        selected_answers.append(selected_option)

    # -------------------------
    # Step 5 & 6: Compare with answer key + Save results
    # -------------------------
    score = 0
    results = []

    for student_ans, correct_ans in zip(selected_answers, answer_key):
        if student_ans == correct_ans:
            score += 1
            results.append("Correct")
        else:
            results.append("Wrong")

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Student Answer", "Correct Answer", "Result"])
        for i, (student_ans, correct_ans, res) in enumerate(zip(selected_answers, answer_key, results)):
            writer.writerow([f"Q{i+1}", student_ans, correct_ans, res])

    # Save marked output image (so Streamlit can display)
    cv2.imwrite(output_image_path, output_grouped)

    # Print final score (same text as original)
    print(f"Final Score: {score}/{len(answer_key)}")

    return score, len(answer_key), csv_filename, output_image_path


if __name__ == "__main__":
    pass
