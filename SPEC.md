# 📋 SPEC.md: Especificación Funcional y Técnica
## Prototipo de Control de Acceso y Uso de Casco de Moto (Nayón)

---

## 1. 🎯 Objetivo del Proyecto
Desarrollar un prototipo de software funcional, visual y de alta precisión para verificar el uso de casco de seguridad en conductores de motocicletas en los puntos de control de Nayón.

El sistema utiliza un modelo de Visión por Computadora pre-entrenado en **Google Teachable Machine** e integrado en una interfaz web reactiva desarrollada en **Streamlit**.

---

## 2. 🧠 Modelo y Clases de Detección
El modelo exportado se encuentra en formato Keras (`keras_model.h5`) con su archivo de etiquetas (`labels.txt`):

* **Clase 0:** `Sin casco`
* **Clase 1:** `Con casco`

### Pipeline de Preprocesamiento de Entrada:
1. **Redimensión:** `224 x 224` píxeles con interpolación bicúbica / bilinear.
2. **Canales:** RGB (3 canales).
3. **Normalización Teachable Machine:**
   $$\text{tensor\_normalizado} = \left(\frac{\text{imagen}}{127.5}\right) - 1.0$$
4. **Forma del tensor de inferencia:** `(1, 224, 224, 3)` de tipo `float32`.
5. **Salida:** Vector de probabilidades Softmax $\in [0.0, 1.0]$.

---

## 3. 🖥️ Requerimientos Funcionales

### RF-01: Interfaz de Usuario Limpia e Intuitiva
* Encabezado institucional claro con identificación del punto de control ("Puesto de Control Nayón").
* Disposición en dos columnas principales:
  * **Columna Izquierda:** Captura e inspección visual (cámara o archivo).
  * **Columna Derecha:** Resultado inmediato de la inspección y semáforo de acceso.
* **Barra Lateral (Sidebar):**
  * Control del umbral de confianza (slider interactivo).
  * Panel de estado del modelo (indicador de carga exitosa).
  * Opciones de sesión (botón para resetear métricas).

### RF-02: Modos Duales de Entrada de Imagen
1. **Modo Cámara Web (`st.camera_input`):**
   * Permite al operador capturar una foto instantánea del motociclista en el punto de control con un solo clic.
2. **Modo Subida de Archivo (`st.file_uploader`):**
   * Permite cargar imágenes existentes (`.jpg`, `.jpeg`, `.png`) para pruebas directivas o auditorías posteriores.

### RF-03: Semáforo y Alertas Visuales Inmediatas
El sistema evalúa la predicción del modelo frente al **Umbral de Confianza** configurado:

| Condición | Veredicto | Color / Alerta | Mensaje en Pantalla |
| :--- | :--- | :--- | :--- |
| Predicción = `Con casco` & Confianza $\ge$ Umbral | **ACCESO PERMITIDO** | 🟢 **Verde** (Éxito) | ✅ Motociclista con casco verificado. Acceso autorizado. |
| Predicción = `Sin casco` & Confianza $\ge$ Umbral | **ACCESO DENEGADO** | 🔴 **Rojo** (Peligro) | ⛔ Infracción detectada: Sin casco. Detener vehículo. |
| Confianza $<$ Umbral | **NO CONCLUYENTE** | 🟡 **Ámbar** (Advertencia) | ⚠️ Confianza insuficiente. Repetir captura o enfocar mejor. |

* Cada resultado muestra claramente el porcentaje de certeza obtenido (ej. `94.8%`).

### RF-04: Umbral de Confianza Configurable (Slider)
* Control tipo slider en la barra lateral con rango de **50% a 95%** (valor por defecto: **75%**).
* Permite a los directores ajustar en tiempo real el nivel de exigencia según la iluminación o condiciones climáticas de Nayón.

### RF-05: Tablero y Métricas de Sesión
El sistema mantiene el estado de la sesión activa (`st.session_state`) sin requerir bases de datos externas:
1. **Tarjetas de métricas superiores:**
   * 📊 **Total de Inspecciones:** Número acumulado en la sesión.
   * 🟢 **Accesos Permitidos:** Cantidad de motociclistas con casco.
   * 🔴 **Accesos Denegados:** Cantidad de infracciones registradas.
   * 📈 **Tasa de Cumplimiento:** Porcentaje de cumplimiento $(\frac{\text{Permitidos}}{\text{Total}} \times 100\%)$.
2. **Bitácora Histórica Reciente:**
   * Tabla con los últimos registros evaluados que incluye:
     - `Hora` (hh:mm:ss)
     - `Veredicto` (PERMITIDO / DENEGADO / NO CONCLUYENTE)
     - `Confianza` (ej. 92.4%)
     - `Modo` (Cámara / Archivo)

---

## 4. ⚡ Requerimientos No Funcionales
* **Rendimiento y Latencia:** Inferencia local en menos de **400 ms** por imagen en procesadores Apple Silicon / Intel modernos.
* **Cero Fugas de Memoria:** Carga del modelo Keras en caché usando `@st.cache_resource` para evitar recargas en cada interacción.
* **Compatibilidad:** Ejecución sobre macOS (Python 3.10 / 3.11).
* **Usabilidad:** Sin necesidad de conocimientos técnicos por parte del operador; botones grandes y estados de color inequívocos.

---

## 5. 📁 Arquitectura de Archivos Sugerida

```text
deteccion-casco/
├── AGENTS.md               # Definición de la célula multi-agente
├── SPEC.md                 # Especificaciones técnicas y funcionales
├── README.md               # Guía ejecutiva de arranque rápido para directores
├── requirements.txt        # Dependencias de Python (Streamlit, TensorFlow/Keras, Pillow, NumPy)
├── run.sh                  # Script de arranque en 1 clic
├── model/                  # Archivos extraídos del modelo Teachable Machine
│   ├── keras_model.h5      # Pesos y arquitectura de la red neuronal
│   └── labels.txt          # Etiquetas de clasificación (0 Sin casco, 1 Con casco)
└── src/
    ├── __init__.py
    ├── model_service.py    # Lógica desacoplada de inferencia y carga de modelo
    └── app.py              # Interfaz de usuario Streamlit y gestión de sesión
```
