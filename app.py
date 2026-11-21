import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO

st.set_page_config(page_title="Detector YOLO en Vivo", page_icon="📷", layout="wide")

st.title("📷 Detección de Objetos en Tiempo Real con YOLO")
st.caption("Usa la cámara de tu navegador para detectar clases entrenadas con tu modelo YOLO.")

# Sidebar de configuración
st.sidebar.header("Configuración del Modelo")
model_path = st.sidebar.text_input("Ruta o nombre del modelo (.pt)", value="best.pt")
conf_threshold = st.sidebar.slider("Umbral de confianza (conf)", 0.10, 0.95, 0.40, 0.05)
img_size = st.sidebar.selectbox("Tamaño de inferencia (imgsz)", [320, 480, 640], index=1)


@st.cache_resource
def load_model(path: str):
    return YOLO(path)


try:
    model = load_model(model_path)
    st.sidebar.success(f"Modelo cargado: {model_path}")
except Exception as e:
    st.error(f"No se pudo cargar el modelo '{model_path}'. Verifica que el archivo exista en el repositorio.")
    st.exception(e)
    st.stop()

# Filtro opcional por clases entrenadas
class_names = model.names
selected_classes = st.sidebar.multiselect(
    "Clases a detectar (vacío = todas)",
    options=list(class_names.values()),
    default=[],
)
selected_class_ids = [
    cls_id for cls_id, name in class_names.items() if name in selected_classes
] if selected_classes else None


class YOLOProcessor(VideoProcessorBase):
    def __init__(self, yolo_model, conf: float, imgsz: int, classes):
        self.model = yolo_model
        self.conf = conf
        self.imgsz = imgsz
        self.classes = classes

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        results = self.model.predict(
            source=img,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.classes,
            verbose=False,
        )

        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="yolo-webcam",
    rtc_configuration=rtc_configuration,
    video_processor_factory=lambda: YOLOProcessor(
        yolo_model=model,
        conf=conf_threshold,
        imgsz=img_size,
        classes=selected_class_ids,
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
