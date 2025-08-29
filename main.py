
import joblib
import streamlit as st
import pandas as pd

# --- Configuración de la Página ---
# Esto debe ser lo primero que se ejecute en el script.
st.set_page_config(
    page_title="Predictor de Concentrado de Sílice (%) en un proceso de Flotación",
    page_icon="📊🧪🫧",
    layout="wide"
)

# --- Carga del Modelo ---
# Usamos @st.cache_resource para que el modelo se cargue solo una vez y se mantenga en memoria,
# lo que hace que la aplicación sea mucho más rápida.
@st.cache_resource
def load_model(model_path):
    """Carga el modelo entrenado desde un archivo .joblib."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en {model_path}. Asegúrate de que el archivo del modelo esté en el directorio correcto.")
        return None

# Cargamos nuestro modelo campeón. Streamlit buscará en la ruta 'modelo.joblib'.
model = load_model('modelo.joblib')

# --- Barra Lateral para las Entradas del Usuario ---
with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")
    st.markdown("""
    Ajusta los deslizadores para que coincidan con los parámetros operativos del proceso de flotación.
    """)

    # Slider para el Concentrado de hierro
    iron = st.slider(
        label=' ⛏️🪨Concentrado de hierro (%)',
        min_value=62.05,
        max_value=68.01,
        value=65, # Valor inicial
        step=0.1
    )
    st.caption("Representa la fracción del mineral que ha sido recuperada en la espuma (froth) después de la separación")


    # Slider para el Flujo de aire
    air = st.slider(
        label='🌬️🫧🌀Flujo de aire - Columna de flotación 01',
        min_value=175.84734,
        max_value=372.44264,
        value=200,
        step=0.1
    )
    st.caption("Cantidad de aire que se inyecta a través del sistema de dispersión en la columna")

    # Slider para el Flujo de Amina
    amina = st.slider(
        label='🧪💧⚙️Flujo de Amina',
        min_value=241.70237,
        max_value=739.304,
        value=350,
        step=0.1
    )
    st.caption("Representa la dosificación de reactivo colector del tipo amina que se alimenta a la columna")

# --- Contenido de la Página Principal ---
st.title("📊🧪🫧 Predictor de Concentrado de Sílice (%) en un proceso de Flotación")
st.markdown("""
¡Bienvenido! Esta aplicación usa un modelo de machine learning para predecir el Concentrado de Sílice (%) en flotación a partir de tres variables operativas clave: Concentrado de hierro (%), Flujo de aire – Columna de flotación 01 y Flujo de Amina.

**Esta herramienta te ayuda a:**
- **Optimizar** los setpoints de aire y amina manteniendo la calidad del concentrado de hierro.
- **Predecir** cómo cambiará el %SiO₂ ante ajustes en flujo de aire y dosificación de amina.
- **Solucionar** variaciones del proceso con escenarios “what-if” y análisis de sensibilidad.

**Cómo usarla:**
- **Establece** Concentrado de hierro (%) actual.
- **Define** Flujo de aire – Columna 01.
- **Indica** Flujo de Amina (como solución o equivalente en masa).
- **Obtén** la predicción de %SiO₂, con intervalo de confianza y sugerencias operativas.

""")

# --- Lógica de Predicción ---
# Solo intentamos predecir si el modelo se ha cargado correctamente.
if model is not None:
    # El botón principal que el usuario presionará para obtener un resultado.
    if st.button('🚀 Predecir Concentrado de Sílice (%)', type="primary"):
        # Creamos un DataFrame de pandas con las entradas del usuario.
        # ¡Es crucial que los nombres de las columnas coincidan exactamente con los que el modelo espera!
        df_input = pd.DataFrame(
            [[iron, air, amina]],
            columns=["% Iron Concentrate", "Flotation Column 01 Air Flow", "Amina Flow"]
        )

        # Hacemos la predicción
        try:
            prediction_value = model.predict(df_input)
            st.subheader("📈 Resultado de la Predicción")
            # Mostramos el resultado en un cuadro de éxito, formateado a dos decimales.
            st.success(f"**Concentrado Predicho:** `{prediction_value[0]:.2f}%`")
            st.info("Este valor representa el porcentaje del concentrado de sílice estimado.")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
else:
    st.warning("El modelo no pudo ser cargado. Por favor, verifica la ruta del archivo del modelo.")

st.divider()

# --- Sección de Explicación ---
with st.expander("ℹ️ Sobre la Aplicación"):
    st.markdown("""
    **¿Cómo funciona?**

    1.  **Datos de Entrada:** Proporcionas los parámetros operativos clave usando los deslizadores en la barra lateral.
    2.  **Predicción:** El modelo de machine learning pre-entrenado recibe estas entradas y las analiza basándose en los patrones que aprendió de datos históricos.
    3.  **Resultado:** La aplicación muestra el rendimiento final predicho como un porcentaje.

    **Detalles del Modelo:**

    * **Tipo de Modelo:** `Regression Model` (XGBoost Optimizado)
    * **Propósito:** Predecir el valor continuo del porcentaje del concentrado de sílice.
    * **Características Usadas:** Concentrado de hierro, Flujo de aire – Columna 01 y Flujo de Amina.
    """)
