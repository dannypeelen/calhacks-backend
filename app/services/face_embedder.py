# app/services/face_embedder.py
import torch
import numpy as np
import cv2
import chromadb
import uuid
from datetime import datetime
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from app.services.face_utils import crop_and_zoom

# -------------------- Setup Models -------------------- #

device = "cuda" if torch.cuda.is_available() else "cpu"

# YOLOv8 face detection (replace with your trained weights)
yolo_face = YOLO("yolov8n-face.pt")

# Face embedding model (FaceNet)
embed_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Connect or create ChromaDB collection
chroma_client = chromadb.Client()
collection_name = "incident_faces"

try:
    collection = chroma_client.get_collection(collection_name)
except:
    collection = chroma_client.create_collection(name=collection_name)


# -------------------- Core Functions -------------------- #

def get_embedding(face_bgr: np.ndarray) -> np.ndarray:
    """Generate a 512-dim embedding vector from a cropped face."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (160, 160))
    tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float().unsqueeze(0)
    tensor = fixed_image_standardization(tensor).to(device)
    with torch.no_grad():
        emb = embed_model(tensor).cpu().numpy()[0].astype(float)
    return emb


def store_face_embedding(embedding: np.ndarray, meta: dict) -> str:
    """Store embedding vector and metadata in ChromaDB."""
    face_id = str(uuid.uuid4())
    collection.add(
        ids=[face_id],
        embeddings=[embedding.tolist()],
        metadatas=[meta]
    )
    return face_id


def process_faces_from_frame(frame: np.ndarray, event_type: str, theft_conf=None, weapon_conf=None):
    """
    Run YOLOv8 face detection, generate embeddings, and store in ChromaDB.
    Returns list of stored faces and metadata.
    """
    results = yolo_face.predict(frame, conf=0.5, device=device)
    stored_faces = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            if conf < 0.6:
                continue

            face_crop = crop_and_zoom(frame, box)
            if face_crop.size == 0:
                continue

            embedding = get_embedding(face_crop)
            meta = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "theft_confidence": theft_conf,
                "weapon_confidence": weapon_conf,
                "face_confidence": float(conf)
            }
            face_id = store_face_embedding(embedding, meta)
            stored_faces.append({"face_id": face_id, "metadata": meta})

    return stored_faces
