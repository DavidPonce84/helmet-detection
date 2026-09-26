# 🤖 Célula de Trabajo Multi-Agente: Control de Cascos Nayón

Este documento define la estructura operativa, roles, responsabilidades y asignación de Modelos de Lenguaje (LLMs) para el equipo de desarrollo del prototipo de control de cascos en Nayón.

---

## 👥 Resumen de la Célula y Asignación de LLMs

Para maximizar la eficiencia y optimizar costos de cómputo, se dividen los roles asignando **modelos de alta capacidad de razonamiento (Tier 1)** a la lógica y calidad, y **modelos ultra-rápidos de baja latencia (Tier 2/3)** a las tareas de versionamiento y documentación.

| Rol | Agente | Nivel de LLM Recomendado | Ejemplos de LLM | Justificación Técnica |
| :--- | :--- | :--- | :--- | :--- |
| **Arquitectura y Lógica** | `@Product-Architect` | **Tier 1 (Razonamiento Profundo)** | Claude 3.7 Sonnet / Gemini 1.5 Pro / GPT-4o | Requiere abstracción de negocio, pipeline de Computer Vision, preprocesamiento Keras (224x224) y arquitectura modular. |
| **Interfaz y UX** | `@Frontend-Dev` | **Tier 1.5 (Prototipado Rápido & Código)** | Claude 3.5 Sonnet / Gemini 2.0 Flash / GPT-4o | Especialista en Streamlit, diseño visual de alto impacto (semáforo de acceso), estado reactivo (`st.session_state`) y widgets. |
| **Control de Calidad** | `@QA-Tester` | **Tier 1 (Verificación Rigurosa & Análisis)** | Claude 3.7 Sonnet / Gemini 1.5 Pro | Detección de fallos silenciosos en ML, matrices de prueba de imágenes, validación de umbrales y consistencia tras cada cambio. |
| **Versionamiento & Ops** | `@GitOps-Agent` | **Tier 2 (Ultra-Rápido & Eficiente)** | Gemini 1.5 Flash / Claude 3.5 Haiku / GPT-4o-mini | Tareas deterministas de git, commits convencionales, control de ramas, `.gitignore` y bitácoras de cambios sin latencia. |

---

## 🛠️ Detalle de Roles y Responsabilidades

### 1. 📐 `@Product-Architect` (Arquitectura & Lógica de Negocio)
* **Objetivo:** Garantizar que la lógica de clasificación, el preprocesamiento de imágenes de Teachable Machine y el flujo de decisiones respondan a la necesidad operativa en Nayón.
* **Responsabilidades:**
  - Diseñar el pipeline de inferencia: carga de `keras_model.h5`, mapeo de `labels.txt` (`0 Sin casco`, `1 Con casco`).
  - Definir la normalización de tensores según Teachable Machine: tamaño `(1, 224, 224, 3)` con escala `(image / 127.5) - 1.0`.
  - Establecer las reglas de negocio del umbral de confianza (Confidence Threshold) y la política de acceso:
    - Certeza $\ge$ Umbral y Clase = "Con casco" $\rightarrow$ **Acceso Permitido**.
    - Certeza $\ge$ Umbral y Clase = "Sin casco" $\rightarrow$ **Acceso Denegado**.
    - Certeza $<$ Umbral $\rightarrow$ **Captura No Concluyente / Reintentar**.
* **Entregables:** Diagramas de flujo, especificaciones de funciones centrales de inferencia (`model_loader.py`, `inference.py`).

---

### 2. 🎨 `@Frontend-Dev` (Especialista en Prototipado Streamlit)
* **Objetivo:** Construir una interfaz visual limpia, moderna y de alta usabilidad para operadores y directores de proyecto.
* **Responsabilidades:**
  - Implementar la aplicación web utilizando **Streamlit**.
  - Crear el selector de entrada dual: `st.camera_input` (cámara web en vivo) y `st.file_uploader` (subida de fotos).
  - Diseñar el feedback visual inmediato:
    - 🟢 **Verde brillante**: "ACCESO PERMITIDO - Conduce con casco".
    - 🔴 **Rojo de alerta**: "ACCESO DENEGADO - Infracción: Sin casco".
    - 🟡 **Ámbar de precaución**: "INSPECCIÓN NO CONCLUYENTE - Reintente captura".
  - Integrar la barra lateral (*Sidebar*) con el **slider interactivo de umbral de confianza** (50% a 95%, por defecto 75%).
  - Construir el panel de métricas de sesión (`st.session_state`): Contador Total, Aprobados, Rechazados, y tabla con historial reciente.
* **Entregables:** `app.py`, componentes visuales, estilos CSS limpios y minimalistas.

---

### 3. 🧪 `@QA-Tester` (Pruebas y Validación Continua)
* **Objetivo:** Asegurar la fiabilidad y robustez del sistema antes de cada despliegue o presentación a los directores.
* **Responsabilidades:**
  - Validar que el modelo cargue correctamente en memoria sin bloqueos de concurrencia.
  - Testear las transformaciones de imágenes con diferentes formatos (`.jpg`, `.png`, fotos oscuras o de baja resolución).
  - Comprobar que los contadores de sesión no se reinicien involuntariamente al interactuar con widgets de Streamlit.
  - Ejecutar pruebas límite en el umbral de confianza (ej. qué pasa con predicciones al 50.1%).
  - Elaborar un protocolo de verificación manual para pruebas de campo en Nayón.
* **Entregables:** Scripts de prueba (`tests/test_inference.py`), checklist de validación para directores.

---

### 4. 🚀 `@GitOps-Agent` (Documentación, Versionamiento y Entorno)
* **Objetivo:** Mantener el repositorio ordenado, reproducible y listo para ejecutar en cualquier máquina con un solo comando.
* **Responsabilidades:**
  - Mantener un `.gitignore` estricto que proteja archivos temporales de Python, pesos no deseados o cachés (`__pycache__`, `.venv`).
  - Documentar cada hito mediante commits semánticos (`feat:`, `fix:`, `docs:`, `chore:`).
  - Mantener actualizados `README.md`, `requirements.txt` y scripts de arranque rápido (`run.sh`).
  - Organizar la estructura de carpetas (`model/`, `src/`, `tests/`).
* **Entregables:** Repositorio estructurado, historial de commits limpio, manual de ejecución en un clic.

---

## 🔄 Flujo de Trabajo y Colaboración (Protocolo de Hand-off)

```mermaid
flowchart LR
    A["@Product-Architect\n(Diseña Lógica & Pipeline)"] --> B["@Frontend-Dev\n(Construye UI Streamlit)"]
    B --> C["@QA-Tester\n(Verifica Estabilidad & Model)"]
    C -->|Aprobado| D["@GitOps-Agent\n(Versiona & Documenta)"]
    C -->|Errores Detectados| B
```

1. **Definición:** `@Product-Architect` valida los requerimientos y el modelo de Teachable Machine.
2. **Construcción:** `@Frontend-Dev` crea la experiencia de usuario y enlaza el modelo.
3. **Control de Calidad:** `@QA-Tester` somete la aplicación a pruebas con y sin casco.
4. **Cierre:** `@GitOps-Agent` consolida la versión, documenta la guía de uso y prepara la entrega para los directores.
