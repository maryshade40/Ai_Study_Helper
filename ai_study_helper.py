import streamlit as st
import openai
import cv2
import numpy as np
import time
import pygame
import threading

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("API key:", api_key)


# Initialize pygame
pygame.mixer.init()

def play_beep():
    pygame.mixer.music.load("Beep.wav")  # Ensure you have a "beep.wav" file
    pygame.mixer.music.play()

# OpenAI GPT Homework Helper
def get_ai_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Camera Monitor - Sleep detection
def monitor_camera():
    st.warning("Webcam monitoring started. A new window will open. Close it to stop.")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    cap = cv2.VideoCapture(0)

    no_eye_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        eyes_detected = False
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 1:
                eyes_detected = True
                no_eye_count = 0
            else:
                no_eye_count += 1

        if no_eye_count > 20:
            play_beep()
            no_eye_count = 0

        cv2.imshow('Monitoring (Press Q to stop)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Background reading reminder (simple timer simulation)
def start_reminder(minutes):
    while True:
        time.sleep(minutes * 60)
        st.toast("⏰ Time to read your scheduled material!", icon="📚")

# --- Streamlit UI ---
st.set_page_config(page_title="AI Study Helper", layout="centered")

st.title("📘 AI Study Helper")
st.markdown("Boost your focus and productivity with homework help, reminders, and sleep detection.")

# Homework helper
st.subheader("🧠 Homework Helper")
user_input = st.text_area("Ask your question:")

if st.button("Ask AI"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            response = get_ai_response(user_input)
        st.success("Here's what AI says:")
        st.write(response)
    else:
        st.warning("Please enter a question.")

# Reading Reminder
st.subheader("⏰ Reading Reminder")
reminder_time = st.number_input("Remind me every (minutes):", min_value=1, max_value=120, value=60, step=1)

if st.button("Start Reading Reminder"):
    st.success(f"Reminder set every {reminder_time} minute(s)")
    threading.Thread(target=start_reminder, args=(reminder_time,), daemon=True).start()

# Camera Monitor
st.subheader("🎥 Sleep Monitor")
if st.button("Start Camera Sleep Monitor"):
    threading.Thread(target=monitor_camera, daemon=True).start()
