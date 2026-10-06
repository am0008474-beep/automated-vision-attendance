"""
VisionAttend AI - Automated Vision Attendance & Face Recognition Dashboard
Author: Ashish Kumar Maurya | B.Tech CSE (Artificial Intelligence)
Streamlit Interactive Web Application
"""

import os
import io
import time
import datetime
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.database import (
    init_db,
    register_student,
    get_all_students,
    get_student_by_id,
    mark_attendance,
    get_today_attendance,
    get_attendance_history,
    get_attendance_df,
    get_analytics_summary
)
from src.face_engine import (
    detect_faces,
    crop_and_normalize_face,
    save_student_face_samples,
    train_face_recognizer,
    FaceRecognizerManager,
    seed_demo_students_if_empty
)

# Page Setup
st.set_page_config(
    page_title="VisionAttend AI | Ashish Kumar Maurya",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Aesthetics
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
  
  html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
  }
  .app-header {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(14, 165, 233, 0.08) 100%);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
  }
  .author-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.15);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 12px;
  }
  .result-banner {
    border-radius: 12px;
    padding: 18px 22px;
    margin: 15px 0;
  }
  .banner-success {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid #10b981;
    color: #065f46;
  }
  .banner-info {
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid #0284c7;
    color: #0369a1;
  }
  .banner-warning {
    background: rgba(244, 63, 94, 0.15);
    border: 1px solid #f43f5e;
    color: #9f1239;
  }
</style>
""", unsafe_allow_html=True)

# Initialize DB and Seed Demo Data on Startup
init_db()
seed_demo_students_if_empty()

@st.cache_resource
def get_recognizer_manager():
    manager = FaceRecognizerManager()
    if not manager.is_trained():
        train_face_recognizer()
        manager.load_model()
    return manager

recognizer = get_recognizer_manager()

# Top Header
st.markdown("""
<div class="app-header">
  <div class="author-tag">
    <span>👨‍💻 Created by Ashish Kumar Maurya</span>
    <span>•</span>
    <span>B.Tech CSE (Artificial Intelligence)</span>
  </div>
  <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #059669;">
    👁️ VisionAttend AI
  </h1>
  <p style="margin: 6px 0 0 0; font-size: 1.05rem; opacity: 0.85;">
    Automated Computer Vision Attendance & Anti-Proxy Facial Recognition System (OpenCV & SQLite)
  </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    all_students = get_all_students()
    st.metric("Registered Students", len(all_students))
    
    analytics = get_analytics_summary()
    st.metric("Today's Check-ins", analytics["today_present"])
    st.metric("Today's Rate", f"{analytics['attendance_rate']}%")

    st.markdown("---")
    st.markdown("### 👤 Author Credentials")
    st.markdown("""
    * **Developer:** Ashish Kumar Maurya
    * **Specialization:** B.Tech CSE (AI)
    * **Portfolio:** [am0008474-beep.github.io](https://am0008474-beep.github.io)
    * **GitHub:** [am0008474-beep](https://github.com/am0008474-beep)
    * **LinkedIn:** [Ashish Kumar Maurya](https://www.linkedin.com/in/ashish-maurya-cse)
    * **Contact:** +91 6387727879
    """)
    st.markdown("---")
    st.caption("Powered by OpenCV Haar Cascades, LBPH Face Recognizer & SQLite3.")

# Main Application Tabs
tab_scan, tab_register, tab_records, tab_analytics = st.tabs([
    "📷 Live Attendance Scanner",
    "📝 Student Registration Portal",
    "📋 Attendance Records & Export",
    "📊 Analytics & Insights"
])


# ----------------- TAB 1: LIVE ATTENDANCE SCANNER -----------------
with tab_scan:
    st.markdown("#### Real-Time Facial Recognition & Attendance Logging")
    st.write("Point your webcam towards the camera or upload a photo to log attendance.")

    input_mode = st.radio("Choose Input Mode:", ["Live Webcam Snapshot", "Upload Image File"], horizontal=True)

    captured_image = None
    if input_mode == "Live Webcam Snapshot":
        camera_file = st.camera_input("Position face in frame and capture:")
        if camera_file is not None:
            captured_image = Image.open(camera_file)
    else:
        uploaded_file = st.file_uploader("Upload student photo:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            captured_image = Image.open(uploaded_file)

    if captured_image is not None:
        # Convert PIL to OpenCV BGR
        img_np = np.array(captured_image)
        if len(img_np.shape) == 3 and img_np.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_np.copy()

        # Face Detection
        t0 = time.time()
        faces, gray_eq = detect_faces(img_bgr)
        proc_latency = round((time.time() - t0) * 1000, 1)

        col_img, col_res = st.columns([1.1, 0.9])

        with col_img:
            annotated_bgr = img_bgr.copy()
            detection_results = []

            if not faces:
                st.warning("⚠️ No face detected in frame. Please ensure proper lighting and face the camera directly.")
            else:
                for (x, y, w, h) in faces:
                    face_roi = crop_and_normalize_face(gray_eq, (x, y, w, h))
                    student_id, confidence = recognizer.recognize(face_roi)

                    if student_id is not None:
                        student_info = get_student_by_id(student_id)
                        if student_info:
                            # Recognized Registered Student
                            label = f"{student_info['name']} ({confidence:.0f}%)"
                            cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), (0, 200, 0), 3)
                            cv2.putText(annotated_bgr, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
                            
                            # Log attendance
                            log_res = mark_attendance(student_id, confidence, device="Webcam Portal")
                            detection_results.append({
                                "status": "recognized",
                                "student": student_info,
                                "confidence": confidence,
                                "log": log_res
                            })
                        else:
                            cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            cv2.putText(annotated_bgr, "Unknown", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            detection_results.append({"status": "unknown", "confidence": confidence})
                    else:
                        # Unrecognized / Proxy suspect
                        cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.putText(annotated_bgr, "Unknown Face", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        detection_results.append({"status": "unknown", "confidence": confidence})

            # Display annotated image
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption=f"Processed Frame ({proc_latency}ms latency)", use_container_width=True)

        with col_res:
            st.markdown("### 📋 Recognition Outcome")
            if not detection_results:
                st.info("Awaiting facial detection input.")
            else:
                for res in detection_results:
                    if res["status"] == "recognized":
                        st_info = res["student"]
                        log_info = res["log"]
                        
                        if log_info.get("status") == "newly_marked":
                            st.markdown(f"""
                            <div class="result-banner banner-success">
                              <h3 style="margin: 0; color: #047857;">✅ Attendance Marked Successfully!</h3>
                              <p style="margin: 6px 0 0 0; font-size: 1.05rem;">
                                <strong>Name:</strong> {st_info['name']}<br/>
                                <strong>Roll No:</strong> {st_info['roll_no']}<br/>
                                <strong>Department:</strong> {st_info['department']}<br/>
                                <strong>Match Confidence:</strong> {res['confidence']:.1f}%<br/>
                                <strong>Timestamp:</strong> {log_info['time']} ({log_info['date']})
                              </p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                        elif log_info.get("status") == "already_marked":
                            st.markdown(f"""
                            <div class="result-banner banner-info">
                              <h3 style="margin: 0; color: #0284c7;">ℹ️ Already Marked Today</h3>
                              <p style="margin: 6px 0 0 0; font-size: 1.05rem;">
                                <strong>Student:</strong> {st_info['name']} ({st_info['roll_no']})<br/>
                                Check-in was already recorded at <strong>{log_info['time']}</strong>.<br/>
                                <em>(Duplicate attendance prevention active)</em>
                              </p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-banner banner-warning">
                          <h3 style="margin: 0; color: #b91c1c;">⚠️ Face Not Recognized</h3>
                          <p style="margin: 6px 0 0 0;">
                            The detected face does not match any registered student in the database.<br/>
                            Please register the student first in the <strong>Registration Portal</strong> tab.
                          </p>
                        </div>
                        """, unsafe_allow_html=True)


# ----------------- TAB 2: STUDENT REGISTRATION -----------------
with tab_register:
    st.markdown("#### 📝 Register New Student with Facial Biometrics")
    st.write("Enter student particulars and capture facial images to update the recognition model.")

    with st.form("student_registration_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            next_id = len(get_all_students()) + 101
            reg_id = st.number_input("Numerical Student ID (unique):", min_value=1, value=next_id, step=1)
            reg_name = st.text_input("Full Name:", placeholder="e.g. Ashish Kumar Maurya")
        with col_f2:
            reg_roll = st.text_input("Roll Number:", placeholder="e.g. 23001004")
            reg_dept = st.selectbox("Department:", [
                "Artificial Intelligence",
                "Computer Science & Engineering",
                "Information Technology",
                "Data Science & Analytics",
                "Electronics & Communication"
            ])

        st.markdown("##### 📸 Biometric Photo Upload / Webcam:")
        reg_photo = st.file_uploader("Upload student portrait photo (good lighting, front facing):", type=["jpg", "png", "jpeg"])
        
        submit_btn = st.form_submit_button("🚀 Register Student & Train Model", type="primary")

    if submit_btn:
        if not reg_name.strip() or not reg_roll.strip():
            st.error("Please enter both Full Name and Roll Number.")
        elif reg_photo is None:
            st.error("Please provide a portrait photo for biometric training.")
        else:
            # Register in SQLite
            ok, msg = register_student(int(reg_id), reg_name, reg_roll, reg_dept)
            if not ok:
                st.error(msg)
            else:
                # Process photo
                pil_img = Image.open(reg_photo)
                np_img = np.array(pil_img)
                if len(np_img.shape) == 3:
                    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                else:
                    bgr_img = np_img

                faces, gray_img = detect_faces(bgr_img)
                if not faces:
                    st.warning("⚠️ No face detected in uploaded photo. Registered record, but biometric samples could not be extracted.")
                else:
                    # Crop primary face
                    bbox = faces[0]
                    norm_face = crop_and_normalize_face(gray_img, bbox)
                    
                    # Generate slight variations for robust training
                    variations = [norm_face]
                    # Minor brightness adjustments
                    variations.append(cv2.convertScaleAbs(norm_face, alpha=1.1, beta=10))
                    variations.append(cv2.convertScaleAbs(norm_face, alpha=0.9, beta=-10))
                    # Horizontal flip
                    variations.append(cv2.flip(norm_face, 1))

                    save_student_face_samples(int(reg_id), variations)
                    train_res = train_face_recognizer()
                    recognizer.load_model()

                    st.success(f"🎉 Student {reg_name} (Roll: {reg_roll}) registered and {len(variations)} biometric samples trained successfully!")

    st.markdown("---")
    st.markdown("#### 👥 Currently Registered Students")
    students_list = get_all_students()
    if students_list:
        st.dataframe(pd.DataFrame(students_list), use_container_width=True)
    else:
        st.info("No students registered yet.")


# ----------------- TAB 3: ATTENDANCE RECORDS & EXPORT -----------------
with tab_records:
    st.markdown("#### 📋 Attendance Log & Data Export")

    col_flt1, col_flt2, col_flt3 = st.columns([1, 1, 1])
    with col_flt1:
        selected_date = st.date_input("Filter by Date:", value=datetime.date.today())
    with col_flt2:
        dept_options = ["All", "Artificial Intelligence", "Computer Science & Engineering", "Information Technology", "Data Science & Analytics", "Electronics & Communication"]
        selected_dept = st.selectbox("Filter Department:", dept_options)
    with col_flt3:
        st.write("")
        st.write("")
        refresh_records = st.button("🔄 Refresh Records")

    date_str = selected_date.strftime("%Y-%m-%d") if selected_date else None
    history = get_attendance_history(date_filter=date_str, department_filter=selected_dept)

    if history:
        df_history = pd.DataFrame(history)
        st.dataframe(df_history, use_container_width=True)

        csv_bytes = df_history.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Filtered Attendance (CSV)",
            data=csv_bytes,
            file_name=f"attendance_{date_str}.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.info(f"No attendance records found for date {date_str} and department '{selected_dept}'.")

    st.markdown("---")
    st.markdown("##### 📁 Export Full Master Database")
    master_df = get_attendance_df()
    if not master_df.empty:
        master_csv = master_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Complete Master Attendance CSV",
            data=master_csv,
            file_name="master_attendance_records.csv",
            mime="text/csv"
        )


# ----------------- TAB 4: ANALYTICS & INSIGHTS -----------------
with tab_analytics:
    st.markdown("#### 📊 Real-Time Attendance Analytics")
    
    analytics = get_analytics_summary()
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Registered Students", analytics["total_students"])
    kpi2.metric("Checked In Today", analytics["today_present"])
    kpi3.metric("Absent Today", analytics["today_absent"])
    kpi4.metric("Attendance Turnout", f"{analytics['attendance_rate']}%")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # Donut Chart
        labels = ["Present Today", "Absent Today"]
        values = [analytics["today_present"], analytics["today_absent"]]
        fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=["#10b981", "#f43f5e"])])
        fig_donut.update_layout(title="Today's Attendance Ratio", height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_g2:
        # Department Breakdown
        dept_data = analytics["dept_counts"]
        if dept_data:
            df_dept = pd.DataFrame(list(dept_data.items()), columns=["Department", "Present Count"])
            fig_bar = px.bar(df_dept, x="Department", y="Present Count", color="Department", title="Attendance by Department")
            fig_bar.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No attendance logged yet for department breakdown.")

    # Daily trends
    daily_trends = analytics["daily_trends"]
    if daily_trends:
        df_trends = pd.DataFrame(list(daily_trends.items()), columns=["Date", "Check-ins"]).sort_values("Date")
        fig_trend = px.line(df_trends, x="Date", y="Check-ins", markers=True, title="Recent Daily Check-in Volume")
        fig_trend.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_trend, use_container_width=True)
