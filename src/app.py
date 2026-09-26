"""
Aplicación Web de Control de Cascos de Moto en Nayón.
Desarrollada con Streamlit, Teachable Machine (Keras), OpenCV y WebRTC.
Soporta reconocimiento continuo en vivo por video streaming, fotos instantáneas y subida de archivos.
"""
import time
from datetime import datetime
from PIL import Image
import pandas as pd
import streamlit as st
import cv2
import av
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
    RTCConfiguration,
)

import urllib.request
import json
import threading

from model_service import load_prediction_model, load_labels, predict_helmet

DEFAULT_MAKE_WEBHOOK_URL = "https://hook.us1.make.com/wjuvrfeufsdem77y7kxnongs96u4kx93"

def notify_make_webhook(webhook_url: str, payload: dict):
    """Envía el reporte de inspección de forma asíncrona a Make sin ralentizar la interfaz."""
    def _send():
        try:
            req = urllib.request.Request(
                webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Control de Cascos - Nayón (En Vivo)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para feedback de alto impacto
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card-green {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-card-red {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-card-amber {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .verdict-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
    }
    .verdict-text {
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 6px;
    }
    .live-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #EF4444;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Servidores STUN públicos para garantizar conexión WebRTC fluida
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)


# Carga de recursos en caché
@st.cache_resource(show_spinner="Iniciando modelo de Teachable Machine...")
def get_model_and_labels():
    model = load_prediction_model()
    labels = load_labels()
    return model, labels


# Procesador de video WebRTC para reconocimiento continuo cuadro a cuadro
class HelmetVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.labels = None
        self.threshold = 0.75
        self.frame_count = 0
        self.last_result = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Procesar 1 de cada 3 fotogramas para balancear alta tasa de FPS y respuesta en tiempo real
        if (self.frame_count % 3 == 0 or self.last_result is None) and self.model is not None:
            try:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                self.last_result = predict_helmet(
                    self.model,
                    self.labels,
                    pil_img,
                    confidence_threshold=self.threshold
                )
            except Exception:
                pass

        # Superponer franja semafórica de veredicto sobre el video
        if self.last_result:
            verdict = self.last_result["verdict"]
            confidence = self.last_result["confidence_percent"]
            class_name = self.last_result["class_name"]

            if verdict == "PERMITIDO":
                color_bgr = (46, 184, 92)    # Verde
                status_txt = f"ACCESO PERMITIDO | {class_name.upper()} ({confidence}%)"
            elif verdict == "DENEGADO":
                color_bgr = (60, 60, 220)    # Rojo
                status_txt = f"ACCESO DENEGADO | {class_name.upper()} ({confidence}%)"
            else:
                color_bgr = (0, 165, 255)    # Ámbar
                status_txt = f"NO CONCLUYENTE ({confidence}%) - ENFOCAR CABEZA"

            h, w, _ = img_bgr.shape

            # Franja superior semafórica
            cv2.rectangle(img_bgr, (0, 0), (w, 52), color_bgr, -1)
            cv2.putText(
                img_bgr,
                status_txt,
                (16, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            # Marco delimitador de color
            cv2.rectangle(img_bgr, (0, 0), (w - 1, h - 1), color_bgr, 5)

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


# Inicialización del estado de sesión
if "total_inspections" not in st.session_state:
    st.session_state.total_inspections = 0
if "approved_count" not in st.session_state:
    st.session_state.approved_count = 0
if "denied_count" not in st.session_state:
    st.session_state.denied_count = 0
if "inconclusive_count" not in st.session_state:
    st.session_state.inconclusive_count = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "last_processed_id" not in st.session_state:
    st.session_state.last_processed_id = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_image" not in st.session_state:
    st.session_state.last_image = None


# Carga del modelo
try:
    model, labels = get_model_and_labels()
    model_ready = True
except Exception as e:
    model_ready = False
    model_error = str(e)


# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/motorcycle-helmet.png", width=64)
    st.title("Puesto Nayón")
    st.caption("Sistema de Control y Seguridad Vial")
    st.markdown("---")

    st.subheader("⚙️ Configuración Operativa")

    threshold_val = st.slider(
        "Umbral de Confianza Requerido",
        min_value=50,
        max_value=95,
        value=75,
        step=5,
        help="Nivel mínimo de certeza exigido para validar una decisión de paso."
    )
    confidence_threshold = threshold_val / 100.0

    st.markdown("---")
    st.subheader("Estado del Sistema")
    if model_ready:
        st.success("✅ Modelo Keras Activo")
        st.caption(f"Clases: {', '.join([l.split(' ', 1)[-1] for l in labels])}")
    else:
        st.error(f"❌ Error al cargar modelo: {model_error}")

    if st.button("🔄 Reiniciar Métricas de Sesión", use_container_width=True):
        st.session_state.total_inspections = 0
        st.session_state.approved_count = 0
        st.session_state.denied_count = 0
        st.session_state.inconclusive_count = 0
        st.session_state.history = []
        st.session_state.last_processed_id = None
        st.session_state.last_result = None
        st.session_state.last_image = None
        st.rerun()

    st.markdown("---")
    st.subheader("📲 Notificaciones (Make / Telegram)")
    enable_make_webhook = st.toggle("Activar Envío a Make", value=True, help="Envía cada inspección en tiempo real al Webhook de Make para Google Sheets y Telegram.")
    if enable_make_webhook:
        st.caption("🟢 **Conectado:** Enlace oficial de Make activo.")
        if st.button("🚀 Enviar Notificación de Prueba", use_container_width=True, help="Envía un registro de prueba instantáneo a Make para verificar Google Sheets y Telegram."):
            test_payload = {
                "fecha": datetime.now().strftime("%d/%m/%Y"),
                "hora_minuto": datetime.now().strftime("%H:%M"),
                "estado": "Con casco",
                "certeza": "98.5%",
                "veredicto": "ACCESO PERMITIDO",
                "mensaje": "Prueba manual de enlace emitida desde el panel de control Nayón hacia Google Sheets y Telegram."
            }
            try:
                req = urllib.request.Request(
                    DEFAULT_MAKE_WEBHOOK_URL,
                    data=json.dumps(test_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=5) as res:
                    if res.status == 200:
                        st.success("✅ ¡Prueba enviada a Make! Revisa tu Google Sheet y Telegram.")
                    else:
                        st.warning(f"Respuesta inesperada: HTTP {res.status}")
            except Exception as e:
                st.error(f"❌ Error al conectar con Make: {e}")

    st.markdown("---")
    st.info("💡 **Consejo:** Para reconocimiento en vivo óptimo, asegúrese de que la cámara apunte a la altura de la cabeza del motociclista.")


# --- ENCABEZADO Y MÉTRICAS SUPERIORES ---
st.markdown('<div class="main-title">🛡️ Control de Cascos de Motocicleta</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Garita de Seguridad y Acceso — Nayón | Reconocimiento en Tiempo Real e Inteligencia Artificial</div>', unsafe_allow_html=True)

# Tarjetas superiores de KPIs
total = st.session_state.total_inspections
compliance_rate = (st.session_state.approved_count / total * 100) if total > 0 else 0.0

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("📊 Total Inspecciones", total)
col_m2.metric("🟢 Accesos Permitidos", st.session_state.approved_count, f"{compliance_rate:.1f}% tasa de uso")
col_m3.metric("🔴 Accesos Denegados", st.session_state.denied_count, "Infracciones")
col_m4.metric("🟡 No Concluyentes", st.session_state.inconclusive_count, "Reintentos")

st.markdown("---")


# --- ÁREA PRINCIPAL: CAPTURA Y EVALUACIÓN ---
col_left, col_right = st.columns([1.15, 0.95], gap="large")

image_to_evaluate = None
image_id = None
input_mode = None

with col_left:
    st.subheader("📸 Entrada de Inspección")

    input_tab_live, input_tab_photo, input_tab_upload = st.tabs([
        "🔴 Reconocimiento en Vivo",
        "📷 Foto Instantánea",
        "📁 Cargar Fotografía"
    ])

    # 1. PESTAÑA: VIDEO STREAMING EN VIVO
    with input_tab_live:
        st.caption("Transmisión continua: La IA analiza el video en tiempo real y superpone el dictamen sobre la imagen.")
        webrtc_ctx = webrtc_streamer(
            key="helmet-live-detector",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=HelmetVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.model = model
            webrtc_ctx.video_processor.labels = labels
            webrtc_ctx.video_processor.threshold = confidence_threshold

            # Si el stream está activo, actualizar el resultado más reciente
            if webrtc_ctx.state.playing and webrtc_ctx.video_processor.last_result:
                st.session_state.last_result = webrtc_ctx.video_processor.last_result

                # Botón para registrar el hallazgo actual en la bitácora
                if st.button("💾 Registrar Detección Actual en Historial de Sesión", use_container_width=True):
                    current = webrtc_ctx.video_processor.last_result
                    st.session_state.total_inspections += 1
                    if current["verdict"] == "PERMITIDO":
                        st.session_state.approved_count += 1
                    elif current["verdict"] == "DENEGADO":
                        st.session_state.denied_count += 1
                    else:
                        st.session_state.inconclusive_count += 1

                    st.session_state.history.insert(0, {
                        "Hora": datetime.now().strftime("%H:%M:%S"),
                        "Modo": "Video en Vivo",
                        "Dictamen": current["verdict"],
                        "Certeza": f"{current['confidence_percent']}%",
                        "Clase": current["class_name"]
                    })
                    if enable_make_webhook:
                        notify_make_webhook(DEFAULT_MAKE_WEBHOOK_URL, {
                            "fecha": datetime.now().strftime("%d/%m/%Y"),
                            "hora_minuto": datetime.now().strftime("%H:%M"),
                            "estado": current["class_name"],
                            "certeza": f"{current['confidence_percent']}%",
                            "veredicto": current["verdict"],
                            "mensaje": current["message"]
                        })
                    st.success("✅ Detección en vivo archivada y enviada a Make.")
                    time.sleep(0.5)
                    st.rerun()

    # 2. PESTAÑA: FOTO CON WEBCAM (ESTÁTICA)
    with input_tab_photo:
        camera_img = st.camera_input("Capturar foto del motociclista en garita", key="static_webcam")
        if camera_img is not None:
            image_to_evaluate = Image.open(camera_img)
            image_id = f"cam_{camera_img.name}_{camera_img.size}"
            input_mode = "Foto Instantánea"

    # 3. PESTAÑA: SUBIDA DE ARCHIVO
    with input_tab_upload:
        uploaded_file = st.file_uploader(
            "Subir imagen para auditoría o prueba",
            type=["jpg", "jpeg", "png"],
            key="file_uploader_input"
        )
        if uploaded_file is not None:
            image_to_evaluate = Image.open(uploaded_file)
            image_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"
            input_mode = "Archivo Subido"

    # Mostrar imagen evaluada en modos estáticos
    if image_to_evaluate is not None:
        st.image(image_to_evaluate, caption=f"Captura evaluada ({input_mode})", use_container_width=True)


# Procesamiento e inferencia para fotos fijas (pestañas 2 y 3)
if image_to_evaluate is not None and model_ready:
    is_new_image = (st.session_state.last_processed_id != image_id)

    result = predict_helmet(
        model=model,
        labels=labels,
        image=image_to_evaluate,
        confidence_threshold=confidence_threshold
    )

    if is_new_image:
        st.session_state.total_inspections += 1
        if result["verdict"] == "PERMITIDO":
            st.session_state.approved_count += 1
        elif result["verdict"] == "DENEGADO":
            st.session_state.denied_count += 1
        else:
            st.session_state.inconclusive_count += 1

        st.session_state.history.insert(0, {
            "Hora": datetime.now().strftime("%H:%M:%S"),
            "Modo": input_mode,
            "Dictamen": result["verdict"],
            "Certeza": f"{result['confidence_percent']}%",
            "Clase": result["class_name"]
        })
        if enable_make_webhook:
            notify_make_webhook(DEFAULT_MAKE_WEBHOOK_URL, {
                "fecha": datetime.now().strftime("%d/%m/%Y"),
                "hora_minuto": datetime.now().strftime("%H:%M"),
                "estado": result["class_name"],
                "certeza": f"{result['confidence_percent']}%",
                "veredicto": result["verdict"],
                "mensaje": result["message"]
            })
        st.session_state.last_processed_id = image_id
        st.session_state.last_result = result
        st.session_state.last_image = image_to_evaluate
    else:
        st.session_state.last_result = result
        st.session_state.last_image = image_to_evaluate


# --- PANEL DERECHO: DICTAMEN DE ACCESO ---
with col_right:
    st.subheader("🚦 Dictamen de Acceso")

    current_result = st.session_state.last_result

    if current_result is not None:
        verdict = current_result["verdict"]
        confidence = current_result["confidence_percent"]
        msg = current_result["message"]

        # Tarjeta visual tipo semáforo
        if verdict == "PERMITIDO":
            st.markdown(f"""
            <div class="status-card-green">
                <div class="verdict-title" style="color: #059669;">🟢 ACCESO PERMITIDO</div>
                <div class="verdict-text" style="color: #065F46;">{msg}</div>
            </div>
            """, unsafe_allow_html=True)
        elif verdict == "DENEGADO":
            st.markdown(f"""
            <div class="status-card-red">
                <div class="verdict-title" style="color: #DC2626;">🔴 ACCESO DENEGADO</div>
                <div class="verdict-text" style="color: #991B1B;">{msg}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-card-amber">
                <div class="verdict-title" style="color: #D97706;">🟡 INSPECCIÓN NO CONCLUYENTE</div>
                <div class="verdict-text" style="color: #92400E;">{msg}</div>
            </div>
            """, unsafe_allow_html=True)

        # Barra y porcentaje de certeza
        st.markdown(f"**Nivel de Certeza:** `{confidence}%` (Umbral mínimo exigido: `{threshold_val}%`)")
        st.progress(float(current_result["confidence"]))

        st.markdown("##### Probabilidades por Clase:")
        for label, prob in current_result["probabilities"].items():
            st.write(f"• **{label}:** `{prob*100:.1f}%`")

    else:
        st.info("👈 Seleccione una pestaña a la izquierda: inicie el **Video en Vivo** o capture una foto para emitir el veredicto.")


# --- BITÁCORA DE HISTORIAL RECIENTE ---
st.markdown("---")
st.subheader("📋 Bitácora de Inspecciones de la Sesión")

if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(
        df_history,
        use_container_width=True,
        hide_index=True
    )
else:
    st.caption("Aún no se han registrado inspecciones en esta sesión.")
