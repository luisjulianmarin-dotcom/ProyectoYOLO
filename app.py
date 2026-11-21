import threading
import time
import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO

st.set_page_config(page_title="Proyecto Aprendizaje Automático", page_icon="📷", layout="wide")
st.caption("Integrantes: Mayra Castañeda - Andres Torres - Julián Marin")

st.title("Identificación de Objetos: Persona, Celular y Gorras")
st.caption("El proyecto genera una alerta 🚨 cuando el bounding box de un celular se intersecta o se ubica dentro del bounding box de una persona.")

# Sidebar de configuración
st.sidebar.header("Configuración del Modelo")
model_path = st.sidebar.text_input("Ruta o nombre del modelo (.pt)", value="best.pt")
conf_threshold = st.sidebar.slider("Umbral de confianza (conf)", 0.10, 0.95, 0.40, 0.05)
img_size = st.sidebar.selectbox("Tamaño de inferencia (imgsz)", [320, 480, 640], index=1)

min_overlap_ratio = st.sidebar.slider(
    "Solapamiento mínimo del celular (%)",
    min_value=0,
    max_value=80,
    value=5,
    step=5,
    help="Porcentaje del área del celular que debe intersectar con la persona (0% = cualquier intersección).",
)


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

class_names = model.names  # dict {id: nombre}

PERSON_KEYWORDS = ("person", "persona", "humano")
PHONE_KEYWORDS = ("cell phone", "phone", "celular", "mobile", "telefono", "teléfono")

person_ids = [
    cid for cid, name in class_names.items()
    if any(k in str(name).strip().lower() for k in PERSON_KEYWORDS)
]
phone_ids = [
    cid for cid, name in class_names.items()
    if any(k in str(name).strip().lower() for k in PHONE_KEYWORDS)
]
target_ids = list(set(person_ids + phone_ids))


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_box(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def check_intersection(person_box, phone_box, min_phone_overlap_frac: float):
    inter = intersection_box(person_box, phone_box)
    if inter is None:
        return False, None

    phone_area = box_area(phone_box)
    if phone_area <= 0:
        return False, None

    inter_area = box_area(inter)
    phone_overlap_frac = inter_area / phone_area
    matched = phone_overlap_frac >= min_phone_overlap_frac
    return matched, inter


class YOLOAlertProcessor(VideoProcessorBase):
    def __init__(self, yolo_model, conf: float, imgsz: int, p_ids, ph_ids, f_ids, min_frac: float):
        self.model = yolo_model
        self.conf = conf
        self.imgsz = imgsz
        self.person_ids = set(p_ids)
        self.phone_ids = set(ph_ids)
        self.filter_ids = f_ids if f_ids else None
        self.min_frac = min_frac

        self._lock = threading.Lock()
        self.alert_active = False
        self.alert_count = 0
        self.last_alert_ts = 0.0

    def get_alert_state(self):
        with self._lock:
            return self.alert_active, self.alert_count, self.last_alert_ts

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        results = self.model.predict(
            source=img,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.filter_ids,
            verbose=False,
        )

        res = results[0]
        annotated = res.plot()

        person_boxes = []
        phone_boxes = []

        if res.boxes is not None and len(res.boxes) > 0:
            xyxy_list = res.boxes.xyxy.cpu().numpy()
            cls_list = res.boxes.cls.cpu().numpy().astype(int)

            for box, cid in zip(xyxy_list, cls_list):
                if cid in self.person_ids:
                    person_boxes.append(box.tolist())
                if cid in self.phone_ids:
                    phone_boxes.append(box.tolist())

        pairs_matched = 0
        for p_box in person_boxes:
            for ph_box in phone_boxes:
                matched, inter = check_intersection(p_box, ph_box, self.min_frac)
                if matched:
                    pairs_matched += 1
                    if inter is not None:
                        x1, y1, x2, y2 = map(int, inter)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    px1, py1, px2, py2 = map(int, ph_box)
                    cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 2)

        is_alert = pairs_matched > 0
        now = time.time()

        with self._lock:
            self.alert_active = is_alert
            self.alert_count = pairs_matched
            if is_alert:
                self.last_alert_ts = now

        if is_alert:
            banner_text = f"ALERTA: CELULAR EN PERSONA ({pairs_matched} coincidencia/s)"
            cv2.rectangle(annotated, (10, 10), (620, 52), (0, 0, 220), -1)
            cv2.putText(
                annotated,
                banner_text,
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


alert_placeholder = st.empty()

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="yolo-person-phone-alert",
    rtc_configuration=rtc_configuration,
    video_processor_factory=lambda: YOLOAlertProcessor(
        yolo_model=model,
        conf=conf_threshold,
        imgsz=img_size,
        p_ids=person_ids,
        ph_ids=phone_ids,
        f_ids=target_ids if target_ids else None,
        min_frac=min_overlap_ratio / 100.0,
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.state.playing and ctx.video_processor:
    while ctx.state.playing:
        active, count, last_ts = ctx.video_processor.get_alert_state()
        if active:
            alert_placeholder.error(
                f"🚨 **ALERTA ACTIVA:** Se detectó un celular interceptando el bounding box de una persona ({count} coincidencia(s))."
            )
        else:
            alert_placeholder.success("✅ Sin alertas: no hay intersección activa entre persona y celular.")
        time.sleep(0.3)
else:
    alert_placeholder.info("Haz clic en **START** para iniciar la cámara y monitorear alertas.")
