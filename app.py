import numpy as np
import cv2
import pandas as pd
import datetime
import streamlit as st
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# Konfigurasi Halaman
st.set_page_config(page_title="Moodify", layout="wide")

# Custom CSS - Minimalist & Smooth
st.markdown("""
<style>
    .stApp {
        background-color: #fafafa;
    }
    h1, h2, h3, h4 {
        color: #333333;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .mood-box {
        padding: 24px;
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        border-left: 4px solid #4A90E2;
        margin-top: 16px;
        color: #444444;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi Session State
if 'mood_history' not in st.session_state:
    st.session_state['mood_history'] = pd.DataFrame(columns=['Tanggal', 'Waktu', 'Emosi', 'Rekomendasi'])

# Load Model
@st.cache_resource
def load_emotion_model():
    emotion_dict = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad', 4: 'Surprise'}
    with open('./models/emotion_model1.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    classifier = model_from_json(loaded_model_json)
    classifier.load_weights("./models/emotion_model1.h5")
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    return classifier, face_cascade, emotion_dict

classifier, face_cascade, emotion_name = load_emotion_model()

# Dictionary Rekomendasi Kegiatan
rekomendasi_kegiatan = {
    'Angry': "Tarik napas perlahan. Biar kepala lebih dingin, coba seduh Teh Poci atau order es Mamamatcha favoritmu sebelum lanjut aktivitas.",
    'Happy': "Kondisi lagi optimal! Waktu yang pas buat ngelanjutin desain konten feed Instagram atau nyelesaiin task MSIB kamu hari ini.",
    'Neutral': "Fokus dan stabil. Sangat ideal untuk ngerjain tugas Statistika atau mulai menyusun skema database buat project selanjutnya.",
    'Sad': "Waktunya istirahat sebentar. Nontonin konten Asahi TREASURE atau self-reward jajan seblak prasmanan dan cireng hangat pasti bikin mood kamu mendingan.",
    'Surprise': "Ada kejadian tak terduga hari ini? Apapun itu, bawa santai aja. Nanti malam bisa rileks pakai greentea mask biar pikiran kembali fresh."
}

# Callback Webcam
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        
    for (x, y, w, h) in faces:
        cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(170, 170, 170), thickness=2)
        roi_gray = img_gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            prediction = classifier.predict(roi)[0]
            maxindex = int(np.argmax(prediction))
            finalout = emotion_name[maxindex]
            
            cv2.putText(img, finalout, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI Utama
st.title("Moodify")
st.markdown("Emotion tracker minimalis. Scan wajah, catat emosi, dan temukan rekomendasi aktivitas harianmu.")

tab1, tab2 = st.tabs(["Scanner & Rekomendasi", "Kalender Mood"])

with tab1:
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("Webcam Scanner")
        webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun4.l.google.com:19302"]},
                ]
            },
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
    with col2:
        st.subheader("Catat Mood Hari Ini")
        st.write("Emosi apa yang paling dominan terdeteksi?")
        
        selected_mood = st.selectbox("Pilih Mood", list(rekomendasi_kegiatan.keys()), label_visibility="collapsed")
        
        if st.button("Simpan & Dapatkan Saran"):
            now = datetime.datetime.now()
            tgl = now.strftime("%Y-%m-%d")
            waktu = now.strftime("%H:%M:%S")
            saran = rekomendasi_kegiatan[selected_mood]
            
            new_data = pd.DataFrame({'Tanggal': [tgl], 'Waktu': [waktu], 'Emosi': [selected_mood], 'Rekomendasi': [saran]})
            st.session_state['mood_history'] = pd.concat([new_data, st.session_state['mood_history']], ignore_index=True)
            
            st.markdown(f"""
            <div class="mood-box">
                <strong>Saran Kegiatan:</strong><br>
                <span style="font-size: 15px;">{saran}</span>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.subheader("Riwayat Mood")
    if st.session_state['mood_history'].empty:
        st.info("Belum ada data mood. Silakan scan wajah dan simpan hasil emosimu di tab sebelah.")
    else:
        st.dataframe(st.session_state['mood_history'], use_container_width=True)
