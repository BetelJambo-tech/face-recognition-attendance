import streamlit as st
import cv2
import numpy as np
import csv
import pandas as pd
from datetime import datetime
from PIL import Image
import os

st.set_page_config(page_title="Face Recognition Attendance", page_icon="📸", layout="wide")

# ----------------------------
# Load model and labels
# ----------------------------
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

label_map = {}
with open("trainer/labels.txt", "r") as f:
    for line in f:
        label_id, person_name = line.strip().split(",")
        label_map[int(label_id)] = person_name


# ----------------------------
# Attendance functions
# ----------------------------
def mark_attendance(name):
    file_exists = os.path.exists("attendance.csv")

    if not file_exists or os.path.getsize("attendance.csv") == 0:
        with open("attendance.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    with open("attendance.csv", "a+", newline="") as f:
        f.seek(0)
        reader = csv.reader(f)
        rows = list(reader)

        recorded_names_today = []
        today = datetime.now().strftime("%Y-%m-%d")

        for row in rows[1:]:
            if len(row) >= 3 and row[1] == today:
                recorded_names_today.append(row[0])

        if name not in recorded_names_today:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str])


def load_attendance():
    if os.path.exists("attendance.csv") and os.path.getsize("attendance.csv") > 0:
        return pd.read_csv("attendance.csv")
    return pd.DataFrame(columns=["Name", "Date", "Time"])


def clear_attendance():
    with open("attendance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("📸 Attendance App")
st.sidebar.markdown("""
### About
This app detects and recognizes faces from uploaded images and records attendance automatically.

### Features
- Face detection
- Face recognition
- Attendance logging
- Download attendance records

### Tech Stack
- Python
- OpenCV
- Streamlit
- Pandas
""")

# ----------------------------
# Main UI
# ----------------------------
st.title("📸 Face Recognition Attendance System")
st.caption("Upload an image to recognize a known face and automatically mark attendance.")

st.info("Upload a face image. If the person is recognized, their attendance will be recorded for today.")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

camera_image = st.camera_input("Or take a picture")

result_col, table_col = st.columns([1.4, 1])

# ----------------------------
# Recognition section
# ----------------------------
with result_col:
    image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

elif camera_image is not None:
    image = Image.open(camera_image).convert("RGB")

if image is not None:
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        st.subheader("Recognition Result")

        if len(faces) == 0:
            st.error("No face detected in the uploaded image.")
        else:
            recognized_names = []

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                face = cv2.equalizeHist(face)

                label_id, confidence = recognizer.predict(face)

                if confidence < 90:
                    name = label_map.get(label_id, "Unknown").upper()
                    mark_attendance(name)
                    recognized_names.append(name)

                    display_text = f"{name} | Match Score: {round(100 - confidence, 1)}%"
                    color = (0, 255, 0)
                    st.success(f"Recognized: {name}")
                else:
                    display_text = "UNKNOWN"
                    color = (255, 0, 0)
                    st.warning("Face not recognized.")

                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(
                    img,
                    display_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            st.image(img, caption="Processed Image", use_container_width=True)

            if recognized_names:
                st.markdown("### Detected Names")
                for person in recognized_names:
                    st.write(f"- {person}")

# ----------------------------
# Attendance table section
# ----------------------------
with table_col:
    st.subheader("📋 Attendance Records")

    attendance_df = load_attendance()

    total_records = len(attendance_df)
    unique_people = attendance_df["Name"].nunique() if not attendance_df.empty else 0

    today_str = datetime.now().strftime("%Y-%m-%d")
    today_count = len(attendance_df[attendance_df["Date"] == today_str]) if not attendance_df.empty else 0

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Total", total_records)
    metric2.metric("People", unique_people)
    metric3.metric("Today", today_count)

    st.markdown("---")

    if attendance_df.empty:
     st.info("No attendance records yet.")
    else:
     st.markdown("### Filter Records")


filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    name_options = ["All"] + sorted(attendance_df["Name"].dropna().unique().tolist())
    selected_name = st.selectbox("Filter by Name", name_options)

with filter_col2:
    date_options = ["All"] + sorted(attendance_df["Date"].dropna().astype(str).unique().tolist())
    selected_date = st.selectbox("Filter by Date", date_options)

search_name = st.text_input("Search Name")
today_only = st.checkbox("Show only today's attendance")

sort_order = st.radio(
    "Sort by Time",
    ["Newest First", "Oldest First"],
    horizontal=True
)

filtered_df = attendance_df.copy()

if selected_name != "All":
    filtered_df = filtered_df[filtered_df["Name"] == selected_name]

if selected_date != "All":
    filtered_df = filtered_df[filtered_df["Date"].astype(str) == selected_date]

if search_name:
    filtered_df = filtered_df[
        filtered_df["Name"].astype(str).str.contains(search_name, case=False, na=False)
    ]

if today_only:
    today_str = datetime.now().strftime("%Y-%m-%d")
    filtered_df = filtered_df[filtered_df["Date"].astype(str) == today_str]

if not filtered_df.empty:
    filtered_df["DateTime"] = pd.to_datetime(
        filtered_df["Date"].astype(str) + " " + filtered_df["Time"].astype(str),
        errors="coerce"
    )

    if sort_order == "Newest First":
        filtered_df = filtered_df.sort_values("DateTime", ascending=False)
    else:
        filtered_df = filtered_df.sort_values("DateTime", ascending=True)

    filtered_df = filtered_df.drop(columns=["DateTime"])

    st.write(f"Showing {len(filtered_df)} record(s)")
    st.dataframe(filtered_df, use_container_width=True, height=350)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Current View",
        data=csv_data,
        file_name="filtered_attendance_records.csv",
        mime="text/csv"
    )

    if st.button("🗑 Clear Attendance"):
        clear_attendance()
        st.success("Attendance cleared.")
        st.rerun()

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with Python, OpenCV, Streamlit, and Pandas")
st.caption("Built by Betel | Face Recognition Attendance Project")