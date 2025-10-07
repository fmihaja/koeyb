from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from contextlib import contextmanager
from typing import List, Optional, Generic, TypeVar
from pydantic.generics import GenericModel
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
from PIL import Image, ImageOps
import io
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration PostgreSQL (Koyeb)
DB_CONFIG = {
    'host': 'ep-super-darkness-a2u72vus.eu-central-1.pg.koyeb.app',
    'user': 'root-adm',
    'password': 'npg_hUoN63HfFyRe',
    'database': 'koyebdb',
    'port': 5432,
     'sslmode': 'require', 
    'cursor_factory': RealDictCursor
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles Pydantic
class Device(BaseModel):
    id: Optional[int]
    status: bool  # True pour allumé, False pour éteint
    name: str

class DeviceResponse(BaseModel):
    id: int
    status: bool
    name: str

class FaceMatchResponse(BaseModel):
    status: bool             # True = correspondance, False = pas de correspondance
    message: str             # Message descriptif
T = TypeVar("T")
class ApiResponse(GenericModel, Generic[T]):
    data: T
    message: str

# Gestionnaire de connexion PostgreSQL
@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    except psycopg2.Error as e:
        print(f"Erreur PostgreSQL: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Erreur de base de données: {str(e)}")
    except Exception as e:
        print(f"Erreur générale: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail="Erreur serveur interne")
    finally:
        if conn:
            conn.close()

# Fonctions améliorées pour la reconnaissance faciale
def load_and_preprocess_image(file: UploadFile):
    """Charge et prétraite une image pour améliorer la détection faciale"""
    try:
        # Lire l'image
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Corriger l'orientation EXIF
        image = ImageOps.exif_transpose(image)
        
        # Convertir en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Redimensionner si l'image est trop grande (optimisation performance)
        max_size = 1000
        if image.size[0] > max_size or image.size[1] > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        return np.array(image)
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de l'image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erreur lors du chargement de l'image: {str(e)}")

def get_face_encoding_improved(image_array):
    """Extrait l'encodage facial avec plusieurs tentatives"""
    try:
        # Essayer d'abord avec le modèle par défaut
        face_locations = face_recognition.face_locations(image_array)
        
        # Si aucun visage détecté, essayer avec un modèle plus sensible
        if len(face_locations) == 0:
            logger.info("Aucun visage détecté avec le modèle par défaut, tentative avec CNN...")
            face_locations = face_recognition.face_locations(image_array, model="cnn")
        
        # Si toujours aucun visage, essayer avec différents paramètres
        if len(face_locations) == 0:
            logger.info("Tentative avec nombre d'échantillons augmenté...")
            face_locations = face_recognition.face_locations(
                image_array, 
                number_of_times_to_upsample=2
            )
        
        if len(face_locations) == 0:
            # Sauvegarder l'image pour debug (optionnel)
            debug_img = Image.fromarray(image_array)
            debug_img.save("debug_image.jpg")
            logger.warning("Image sauvegardée pour debug: debug_image.jpg")
            
            raise HTTPException(
                status_code=400, 
                detail="Aucun visage détecté dans l'image. Assurez-vous que le visage est bien visible et éclairé."
            )
        
        # Si plusieurs visages, utiliser le plus grand
        if len(face_locations) > 1:
            face_sizes = [(bottom - top) * (right - left) for top, right, bottom, left in face_locations]
            largest_face_index = face_sizes.index(max(face_sizes))
            face_locations = [face_locations[largest_face_index]]
            logger.info(f"Multiple visages détectés, utilisation du plus grand (index {largest_face_index})")
        
        # Extraire les encodages
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        if len(face_encodings) == 0:
            raise HTTPException(
                status_code=400, 
                detail="Impossible d'extraire les caractéristiques du visage"
            )
        
        logger.info(f"Visage détecté avec succès, localisation: {face_locations[0]}")
        return face_encodings[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction faciale: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de traitement facial: {str(e)}")

# Routes (inchangées pour les devices)
@app.get("/")
def read_root():
    return {"message": "API ESP32 device Controller"}

@app.get("/health")
def health_check():
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 as test")
                result = cursor.fetchone()
                return {
                    "status": "healthy",
                    "database": "connected",
                    "test_query": result
                }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/device/", response_model=ApiResponse[DeviceResponse])
def create_device_status(device: Device):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO device (status) VALUES (%s) RETURNING id",
                (device.status,)
            )
            device_id = cursor.fetchone()['id']
            connection.commit()

            cursor.execute("SELECT id, status, name FROM device WHERE id = %s", (device_id,))
            device_data = cursor.fetchone()
            response = DeviceResponse(
                id=device_data['id'],
                status=bool(device_data['status']),
                name=device_data['name']
            )
            return ApiResponse(data=response, message="Creation fait")

@app.get("/device/", response_model=ApiResponse[List[DeviceResponse]])
def get_all_device_status(skip: int = 0, limit: int = 100):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, status, name FROM device ORDER BY id ASC LIMIT %s OFFSET %s",
                (limit, skip)
            )
            device_data = cursor.fetchall()
            if not device_data:
                raise HTTPException(status_code=404, detail="device non trouvé")
            response = [
                
                DeviceResponse(id=device['id'], status=bool(device['status']), name=device['name']) for device in device_data
            ]
            return ApiResponse(data=response, message="Liste des lampes")

@app.get("/device/{device_id}", response_model=ApiResponse[DeviceResponse])
def get_device_status(device_id: int):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, status, name FROM device WHERE id = %s", (device_id,))
            device_data = cursor.fetchone()
            if not device_data:
                raise HTTPException(status_code=404, detail="device non trouvée")
            response = DeviceResponse(id=device_data['id'], status=bool(device_data['status']), name=device_data['name'])
            return ApiResponse(data=response, message="Lampe numero " + str(response.id))

@app.put("/device/{device_id}", response_model=ApiResponse[DeviceResponse])
def update_device_status(device: Device):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id FROM device WHERE id = %s", (device.id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="device non trouvée")

            cursor.execute("UPDATE device SET status = %s WHERE id = %s RETURNING id, status, name",
                           (device.status, device.id))
            device_data = cursor.fetchone()
            connection.commit()

            response = DeviceResponse(id=device_data['id'], status=bool(device_data['status']), name=device_data['name'])
            return ApiResponse(data=response, message="Lampe numero " + str(response.id) + " Modifier")

@app.delete("/device/{device_id}")
def delete_device_status(device_id: int):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM device WHERE id = %s", (device_id,))
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="device non trouvée")
            connection.commit()
            return {"message": f"Statut device {device_id} supprimé avec succès"}

@app.post("/device/toggle/", response_model=DeviceResponse)
def toggle_device():
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT status FROM device ORDER BY id DESC LIMIT 1")
            last_status = cursor.fetchone()
            new_status = not bool(last_status['status']) if last_status else True

            cursor.execute("INSERT INTO device (status) VALUES (%s) RETURNING id, name", (new_status,))
            device_id = cursor.fetchone()['id']
            connection.commit()

            cursor.execute("SELECT id, status, name FROM device WHERE id = %s", (device_id,))
            device_data = cursor.fetchone()

            return DeviceResponse(id=device_data['id'], status=bool(device_data['status']), name=device_data['name'])

# NOUVELLE VERSION AMÉLIORÉE DE COMPARE-FACES
@app.post("/compare-faces", response_model=ApiResponse[FaceMatchResponse])
async def compare_faces_files(
    camera_image: UploadFile = File(...),
    stored_image: UploadFile = File(...)
):
    """
    Compare deux images pour la reconnaissance faciale
    """
    try:
        logger.info("📥 Réception des fichiers pour comparaison faciale...")
        logger.info(f"Camera image: {camera_image.filename}, Type: {camera_image.content_type}")
        logger.info(f"Stored image: {stored_image.filename}, Type: {stored_image.content_type}")

        # Vérifications des types de fichiers
        if not camera_image.content_type or not camera_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier camera_image doit être une image")
        
        if not stored_image.content_type or not stored_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier stored_image doit être une image")

        # Charger et prétraiter les images
        img1 = load_and_preprocess_image(camera_image)
        img2 = load_and_preprocess_image(stored_image)
        
        logger.info(f"✅ Images chargées - Camera shape: {img1.shape}, Stored shape: {img2.shape}")

        # Extraire les encodages faciaux
        face1 = get_face_encoding_improved(img1)
        face2 = get_face_encoding_improved(img2)

        # Comparer les visages
        distance = face_recognition.face_distance([face1], face2)[0]
        match = distance < 0.5  # Seuil de similarité

        logger.info(f"🔍 Résultat comparaison - Distance: {distance:.4f}, Match: {match}")

        # Construire la réponse
        if match:
            response_data = FaceMatchResponse(
                status=True,
                message="Visage reconnu avec succès"
            )
        else:
            response_data = FaceMatchResponse(
                status=False,
                message="Aucune correspondance trouvée"
            )

        return ApiResponse(
            data=response_data,
            message="Comparaison terminée"
        )

    except HTTPException as e:
        logger.error(f"❌ Erreur HTTP: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"❌ Erreur générale: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur lors de la comparaison: {str(e)}")

# Nouvelle route pour tester la détection faciale seule
@app.post("/detect-face")
async def detect_face_only(image: UploadFile = File(...)):
    """
    Route pour tester la détection faciale sur une seule image
    """
    try:
        logger.info("🔍 Test de détection faciale...")
        
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")

        img = load_and_preprocess_image(image)
        logger.info(f"Image chargée - Shape: {img.shape}")
        
        # Tenter la détection
        face_locations = face_recognition.face_locations(img)
        
        if len(face_locations) == 0:
            # Essayer d'autres méthodes
            face_locations = face_recognition.face_locations(img, model="cnn")
            
        if len(face_locations) == 0:
            face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=2)
            
        return {
            "faces_detected": len(face_locations),
            "face_locations": face_locations,
            "image_size": img.shape,
            "message": f"{len(face_locations)} visage(s) détecté(s)" if face_locations else "Aucun visage détecté"
        }
        
    except Exception as e:
        logger.error(f"Erreur détection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de détection: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)