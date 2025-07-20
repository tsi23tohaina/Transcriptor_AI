import streamlit as st
import av
import numpy as np
import soundfile as sf
import tempfile
import os
from scipy.signal import butter, lfilter
import noisereduce as nr
from pydub import AudioSegment, silence
from faster_whisper import WhisperModel


st.set_page_config(
    page_icon="🗣",
    page_title="Transcriber"
)

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown(
        "<div style='border-bottom: 1px solid #eee; padding-bottom: 1rem; margin-bottom: 1.5rem;'>"
        "<h1 style='color: #4b4276; font-size: 1.8rem;'>Processing Settings</h1>"
        "</div>", 
        unsafe_allow_html=True
    )
    
    # Paramètres de transcription
    st.subheader("Transcription Settings")
    model_choice = st.selectbox(
        "Transcription Model",
        options=["small", "base", "medium", "large"],
        index=0,
        help="Larger models offer higher accuracy but may take longer to process."
    )
    
    # Nouveaux paramètres audio interactifs
    st.subheader("Audio Settings")
    lowcut = st.slider("Low Frequency Cutoff (Hz)", 100, 1000, 300)
    highcut = st.slider("High Frequency Cutoff (Hz)", 2000, 8000, 3400)
    amplification_factor = st.slider("Amplification Factor", 1.0, 5.0, 1.5, step=0.1)
    min_silence_len = st.slider("Minimum Silence Duration (ms)", 100, 1000, 500)
    silence_thresh = st.slider("Silence Threshold (dB)", -60, -20, -40)
    
    st.markdown(
        "<div style='margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 8px;'>"
        "<p style='font-size: 0.9rem; color: #6c757d;'>Changes will apply to your next audio processing task.</p>"
        "</div>",
        unsafe_allow_html=True
    )

# --- Main Page ---
st.markdown(
    "<div class='header-container'>"
    "<div class='title'>Audio & Video Transcriber</div>"
    "<div class='subtitle'>Crystal-clear transcription with smart audio cleanup</div>"
    "</div>", 
    unsafe_allow_html=True
)

# File uploader
st.markdown(
    "<div class='upload-container'>"
    "<h3 style='color: #4b4276; margin-bottom: 1.5rem;'>Upload a Media File</h3>"
    "</div>",
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    " ",  # Empty string to hide default label
    type=["mp3", "wav", "mp4", "avi", "mkv"],
    label_visibility="collapsed"
)

if not uploaded:
    st.markdown(
        "<div style='text-align: center; padding: 2rem; color: #6c757d;'>"
        "<p style='font-size: 1.1rem;'>Please upload an audio or video file to start processing</p>"
        "</div>",
        unsafe_allow_html=True
    )
    st.stop()

# Processing steps
st.markdown(
    "<div class='process-steps'>"
    "<h3 style='color: #4b4276; margin-bottom: 1.5rem;'>Audio Processing Pipeline</h3>"
    f"<div class='step-item'><div class='step-number'>1</div><div>Voice filtering bandpass ({lowcut}-{highcut} Hz)</div></div>"
    f"<div class='step-item'><div class='step-number'>2</div><div>Noise reduction (amplified ×{amplification_factor})</div></div>"
    f"<div class='step-item'><div class='step-number'>3</div><div>Silence removal ({min_silence_len}ms, {silence_thresh}dB)</div></div>"
    
    "</div>",
    unsafe_allow_html=True
)

# Temp save
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=uploaded.name)
tmp.write(uploaded.read())
tmp.flush()
input_path = tmp.name
base = os.path.splitext(os.path.basename(input_path))[0]

# Progress container
st.markdown("<div class='progress-container'>", unsafe_allow_html=True)
status = st.empty()
progress = st.progress(0)
st.markdown("</div>", unsafe_allow_html=True)

# 1) Décodage
status.markdown(
    "<div class='status-message'>"
    "<span class='status-icon'>🔄</span> Decoding the uploaded media file..."
    "</div>", 
    unsafe_allow_html=True
)
progress.progress(10)

try:
    container = av.open(input_path)
    stream = container.streams.audio[0]
    frames = [frame.to_ndarray() for frame in container.decode(stream)]
    audio_np = np.concatenate(frames, axis=1).T
    rate = stream.rate
except Exception as e:
    status.error(f"Erreur lors du décodage : {str(e)}")
    st.stop()

# 2) Traitement audio
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

# Filtre passe-bande
status.markdown(
    "<div class='status-message'>"
    "<span class='status-icon'>🔊</span> Applying vocal filter..."
    "</div>", 
    unsafe_allow_html=True
)
progress.progress(25)
mono = audio_np.mean(axis=1)
filtered = bandpass_filter(mono, lowcut, highcut, rate)  # Utilisation des paramètres

# Réduction du bruit
status.markdown(
    "<div class='status-message'>"
    "<span class='status-icon'>🧹</span> Reducing background noise..."
    "</div>", 
    unsafe_allow_html=True
)
progress.progress(40)
cleaned = nr.reduce_noise(y=filtered, sr=rate, stationary=True)

# Amplification
status.markdown(
    "<div class='status-message'>"
    "<span class='status-icon'>📈</span> Adjusting volume levels..."
    "</div>", 
    unsafe_allow_html=True
)
progress.progress(55)
amplified = np.clip(cleaned * amplification_factor, -1.0, 1.0)  # Utilisation du paramètre

# Suppression des silences
status.markdown(
    "<div class='status-message'>"
    "<span class='status-icon'>✂️</span> Removing silent segments..."
    "</div>", 
    unsafe_allow_html=True
)
progress.progress(70)
tmp_wav = input_path + "_tmp.wav"
sf.write(tmp_wav, amplified, rate)
audio = AudioSegment.from_wav(tmp_wav)
chunks = silence.split_on_silence(
    audio, 
    min_silence_len=min_silence_len,  # Utilisation du paramètre
    silence_thresh=silence_thresh     # Utilisation du paramètre
)
final = AudioSegment.empty()
for c in chunks:
    final += c

# 3) Export nettoyé
status.markdown(
    "<div class='status-message'>"
    "<span class='status-icon'>🙂</span> Audio processing completed"
    "</div>", 
    unsafe_allow_html=True
)
progress.progress(90)

output_wav = os.path.join(os.path.dirname(input_path), f"{base}_voice_only.wav")
final.export(output_wav, format="wav")
progress.progress(100)

# Résultats audio
st.markdown("<div class='result-card'>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #4b4276; margin-bottom: 1.5rem;'>Processed Audio Output</h3>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])
with col1:
    st.audio(output_wav, format='audio/wav')
    
with col2:
    st.download_button(
        "Télécharger l'audio nettoyé", 
        data=open(output_wav, "rb").read(),
        file_name=f"{base}_voice_only.wav",
        mime="audio/wav",
        use_container_width=True,
        key="download_audio"
    )

st.markdown("</div>", unsafe_allow_html=True)

# 4) Transcription
st.markdown("<div class='result-card' style='margin-top: 2rem;'>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #4b4276; margin-bottom: 1.5rem;'>Transcription Service</h3>", unsafe_allow_html=True)

if st.button("Start Transcription", use_container_width=True, key="transcribe_btn"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    transcription_box = st.empty()
    
    status_text.markdown(
        "<div class='status-message'>"
        " Initializing transcription engine..."
        "</div>", 
        unsafe_allow_html=True
    )
    
    try:
        model = WhisperModel(model_choice, device="cpu")
        status_text.markdown(
            "<div class='status-message'>"
            "<span class='status-icon'>🔍</span> Analyzing and transcribing..."
            "</div>", 
            unsafe_allow_html=True
        )
        
        segments, info = model.transcribe(output_wav, beam_size=5)
        duration = getattr(info, 'duration', None) or 60
        full_text = ""
        last_progress = 0
        
        for i, seg in enumerate(segments):
            full_text += seg.text + " "
            
            # Mise à jour dynamique de la transcription
            if i % 3 == 0:  # Mise à jour tous les 3 segments pour performance
                transcription_box.markdown(
                    f"<div class='transcription-box'>{full_text} [...]</div>", 
                    unsafe_allow_html=True
                )
            
            # Mise à jour de la barre de progression
            if duration:
                pct = min(seg.end / duration, 1.0)
                if pct - last_progress > 0.05:  # Mise à jour par paliers de 5%
                    progress_bar.progress(pct)
                    last_progress = pct
                    status_text.markdown(
                        f"<div class='status-message'>"
                        f"<span class='status-icon'>🔍</span> "
                        f"Transcribing: {pct*100:.1f}%"
                        "</div>", 
                        unsafe_allow_html=True
                    )
        
        progress_bar.progress(1.0)
        status_text.markdown(
            "<div class='status-message'>"
            f"<span class='status-icon'>😀</span> Transcription completed (Langue: {info.language})"
            "</div>", 
            unsafe_allow_html=True
        )
        
        st.markdown("<h4 style='margin-top: 1.5rem; margin-bottom: 1rem;'>Full Transcription</h4>", unsafe_allow_html=True)
        st.markdown(f"<div class='transcription-box'>{full_text}</div>", unsafe_allow_html=True)
        
        # Export TXT
        txt_data = f"Langue détectée : {info.language}\n\n{full_text}"
        st.download_button(
            "Download Transcription",
            data=txt_data,
            file_name=f"{base}_transcript.txt",
            mime="text/plain",
            use_container_width=True
        )
        
    except Exception as e:
        status_text.error(f"Erreur lors de la transcription : {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)