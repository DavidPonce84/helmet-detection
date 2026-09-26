# 🛡️ Sistema de Control de Cascos de Moto - Nayón

Prototipo interactivo desarrollado para la verificación automática del uso de casco en motociclistas que ingresan por el puesto de control de Nayón.

---

## 🚀 Inicio Rápido (1 solo clic)

Para iniciar la aplicación en macOS, abre una terminal en la carpeta del proyecto y ejecuta:

```bash
./run.sh
```

O directamente mediante Streamlit:

```bash
streamlit run src/app.py
```

La aplicación se abrirá automáticamente en tu navegador web en `http://localhost:8501`.

---

## 📋 Funcionalidades Principales

1. **Reconocimiento en Vivo (Tiempo Real):**
   - Transmisión continua por video streaming (WebRTC).
   - Superposición semafórica dinámica (Verde/Rojo/Ámbar) y marco perimetral directamente sobre el video.
   - Botón para congelar y archivar la detección en vivo en el registro histórico.

2. **Captura Estática en Garita:**
   - Opción **Foto Instantánea**: El operador toma la captura del motociclista directamente desde la interfaz.
   - Opción **Cargar Fotografía**: Permite subir imágenes (`.jpg`, `.png`) para pruebas o auditorías.

2. **Semáforo de Acceso:**
   - 🟢 **Acceso Permitido:** Conductor con casco verificado (certeza $\ge$ umbral).
   - 🔴 **Acceso Denegado:** Conductor sin casco detectado (certeza $\ge$ umbral).
   - 🟡 **Inspección No Concluyente:** Certeza inferior al umbral, invitando a repetir la toma.

3. **Control de Umbral Dinámico:**
   - Ajusta en tiempo real el umbral de confianza (50% a 95%) en la barra lateral izquierda según la visibilidad o condiciones de iluminación.

4. **Métricas y Auditoría de Sesión:**
   - Tarjetas superiores con total de inspecciones, permitidos, denegados y tasa porcentual de cumplimiento.
   - Bitácora de historial con registro temporal de cada decisión.
   - Botón para reiniciar métricas en cualquier momento.

---

## 📁 Estructura del Repositorio

* `AGENTS.md`: Definición de la célula de trabajo multi-agente y asignación de LLMs.
* `SPEC.md`: Especificación funcional y técnica detallada.
* `model/`: Modelo exportado de Teachable Machine (`keras_model.h5`, `labels.txt`).
* `src/model_service.py`: Lógica pura de inferencia, preprocesamiento y reglas de decisión.
* `src/app.py`: Interfaz de usuario Streamlit.
* `requirements.txt`: Dependencias del entorno Python.
* `run.sh`: Script ejecutor.
