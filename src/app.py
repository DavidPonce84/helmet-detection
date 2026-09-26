"""
Aplicación Web de Control de Cascos de Moto en Nayón.
Desarrollada con Streamlit y Teachable Machine (Keras).
"""
import time
from datetime import datetime
from PIL import Image
import pandas as pd
import streamlit as st

from model_service import load_prediction_model, load_labels, predict_helmet

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Control de Cascos - Nayón",
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
</style>
""", unsafe_allow_html=True)


# Carga de recursos cacheados para máxima velocidad y evitar fugas de memoria
@st.cache_resource(show_spinner="Iniciando modelo de Teachable Machine...")
def get_model_and_labels():
    model = load_prediction_model()
    labels = load_labels()
    return model, labels


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
        help="Nivel mínimo de certeza que debe tener el modelo para validar una decisión automática."
    )
    confidence_threshold = threshold_val / 100.0
    
    st.markdown("---")
    st.subheader("Estado del Sistema")
    if model_ready:
        st.success("✅ Modelo Keras Activo")
        st.caption(f"Clases registradas: {', '.join([l.split(' ', 1)[-1] for l in labels])}")
    else:
        st.error(f"❌ Error al cargar modelo: {model_error}")
        
    st.markdown("---")
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
    st.info("💡 **Operación:** Apunte la cámara hacia el rostro y cabeza del conductor al detenerse en la garita.")


# --- ENCABEZADO Y MÉTRICAS SUPERIORES ---
st.markdown('<div class="main-title">🛡️ Control de Cascos de Motocicleta</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Garita de Seguridad y Acceso — Nayón | Prototipo de Inspección en Tiempo Real</div>', unsafe_allow_html=True)

# Cálculo de tasa de cumplimiento
total = st.session_state.total_inspections
compliance_rate = (st.session_state.approved_count / total * 100) if total > 0 else 0.0

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("📊 Total Inspecciones", total)
col_m2.metric("🟢 Accesos Permitidos", st.session_state.approved_count, f"{compliance_rate:.1f}% tasa de uso")
col_m3.metric("🔴 Accesos Denegados", st.session_state.denied_count, "Infracciones")
col_m4.metric("🟡 No Concluyentes", st.session_state.inconclusive_count, "Reintentos")

st.markdown("---")


# --- ÁREA PRINCIPAL: CAPTURA Y EVALUACIÓN ---
col_left, col_right = st.columns([1.1, 1.0], gap="large")

image_to_evaluate = None
image_id = None
input_mode = None

with col_left:
    st.subheader("📸 Entrada de Inspección")
    
    input_tab1, input_tab2 = st.tabs(["📷 Cámara Web en Vivo", "📁 Cargar Fotografía"])
    
    with input_tab1:
        camera_img = st.camera_input("Capturar foto del motociclista en garita")
        if camera_img is not None:
            image_to_evaluate = Image.open(camera_img)
            image_id = f"cam_{camera_img.file_id if hasattr(camera_img, 'file_id') else camera_img.name}_{camera_img.size}"
            input_mode = "Cámara Web"
            
    with input_tab2:
        uploaded_file = st.file_uploader(
            "Subir imagen para auditoría o prueba", 
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image_to_evaluate = Image.open(uploaded_file)
            image_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"
            input_mode = "Archivo Subido"

    if image_to_evaluate is not None:
        st.image(image_to_evaluate, caption=f"Imagen capturada ({input_mode})", use_container_width=True)


# Procesamiento e inferencia
if image_to_evaluate is not None and model_ready:
    # Verificamos si es una imagen nueva o un cambio de umbral
    is_new_image = (st.session_state.last_processed_id != image_id)
    
    result = predict_helmet(
        model=model,
        labels=labels,
        image=image_to_evaluate,
        confidence_threshold=confidence_threshold
    )
    
    # Si es una nueva captura, actualizamos contadores históricos
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
        st.session_state.last_processed_id = image_id
        st.session_state.last_result = result
        st.session_state.last_image = image_to_evaluate
    else:
        st.session_state.last_result = result
        st.session_state.last_image = image_to_evaluate


with col_right:
    st.subheader("🚦 Dictamen de Acceso")
    
    current_result = st.session_state.last_result
    
    if current_result is not None:
        verdict = current_result["verdict"]
        color = current_result["color"]
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

        # Desglose de métricas de confianza
        st.markdown(f"**Nivel de Certeza del Modelo:** `{confidence}%` (Umbral exigido: `{threshold_val}%`)")
        st.progress(float(current_result["confidence"]))
        
        st.markdown("##### Probabilidades por Clase:")
        for label, prob in current_result["probabilities"].items():
            st.write(f"• **{label}:** `{prob*100:.1f}%`")
            
    else:
        st.info("👈 Esperando captura. Tome una foto con la cámara web o suba un archivo para realizar el control.")


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
