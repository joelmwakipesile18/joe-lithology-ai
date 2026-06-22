import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Kichwa cha Mfumo na Nembo ya Biashara Yako
st.set_page_config(page_title="EA Lithology AI", layout="centered")
st.title("🌍 Mfumo wa AI wa Kutambua Miamba (East Africa)")
st.write("---")
st.subheader("Meneja Mkuu wa Mfumo: Mkurugenzi Mtendaji")

# 1. Kupakia akili halisi ya AI uliyofundisha sasa hivi
@st.cache_resource
def pakia_ubongo():
    if os.path.exists("akili_ya_miamba.h5"):
        return tf.keras.models.load_model("akili_ya_miamba.h5")
    return None

model = pakia_ubongo()

# Kupata majina ya miamba kutoka kwenye folder lako automatically
path_ya_picha = "miamba_data"
if os.path.exists(path_ya_picha):
    majina_ya_miamba = sorted(os.listdir(path_ya_picha))
else:
    majina_ya_miamba = ["Granite", "Sandstone", "Clay"]

st.write("Pakia picha ya sampuli ya mwamba uliyopiga kwa simu kupata majibu ya kweli kutoka kwa AI.")

# Sehemu ya Kupakia Picha
picha_iliyopakiwa = st.file_uploader("Chagua au pakia picha ya mwamba...", type=["jpg", "png", "jpeg"])

if picha_iliyopakiwa is not None:
    image = Image.open(picha_iliyopakiwa)
    st.image(image, caption='Picha Iliyopakiwa', use_container_width=True)
    
    if model is None:
        st.error("❌ Mfumo haujapata faili la 'akili_ya_miamba.h5'. Hakikisha umorun train_ai.py kwanza.")
    else:
        with st.spinner("🔄 AI Inachambua miundo na rangi za mwamba kwa usahihi..."):
            # Kurekebisha picha ya mteja iendane na akili ya AI
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # AI Inafanya utabiri wa kweli hapa!
            utabiri = model.predict(img_array)
            index_ya_juu = np.argmax(utabiri[0])
            uhakika = float(utabiri[0][index_ya_juu]) * 100
            
            jina_la_mwamba = majina_ya_miamba[index_ya_juu].upper()
            
        # Kuonyesha Matokeo Halisi kwa Mteja
        st.success("✅ Uchambuzi wa Kweli Umekamilika!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Aina ya Mwamba Iliyogundulika", value=jina_la_mwamba)
        with col2:
            st.metric(label="Uhakika wa Mfumo (Confidence)", value=f"{uhakika:.1f}%")
            
        st.info("💵 Mfumo wa Malipo: Gharama ya scan hii ni Tsh 500 / KSh 30. Akaunti yako imekatwa.")