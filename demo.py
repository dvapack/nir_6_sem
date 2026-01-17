import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import time
import os
import json
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self, database_path="face_database.json"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.mtcnn = MTCNN(
            margin=14,
            factor=0.6,
            keep_all=True,
            device=self.device
        )
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.database_path = database_path
        self.face_database = self.load_database()
        
        self.similarity_threshold = 0.7
        self.min_face_size = 20
        
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def load_database(self):
        """Загрузка базы данных лиц из JSON файла"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'r') as f:
                    return json.load(f)
            except:
                print("Ошибка при загрузке базы данных, создаю новую")
        return {}

    def save_database(self):
        """Сохранение базы данных в JSON файл"""
        with open(self.database_path, 'w') as f:
            json.dump(self.face_database, f, indent=2)
        print(f"База данных сохранена в {self.database_path}")

    def detect_faces(self, frame):
        """Обнаружение лиц на кадре и получение эмбеддингов"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            boxes, _ = self.mtcnn.detect(rgb_frame)
            
            if boxes is None:
                return [], []
            
            valid_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1
                if width >= self.min_face_size and height >= self.min_face_size:
                    valid_boxes.append(box)
            
            if not valid_boxes:
                return [], []
            
            faces = self.mtcnn.extract(rgb_frame, valid_boxes, save_path=None)
            
            if faces is None:
                return [], []
            
            embeddings = self.resnet(faces.to(self.device)).cpu().detach().numpy()
            
            return valid_boxes, embeddings
            
        except Exception as e:
            print(f"Ошибка обнаружения лиц: {e}")
            return [], []

    def identify_face(self, embedding):
        """Идентификация лица по эмбеддингу"""
        if not self.face_database:
            return "unknown", 0.0
        
        best_match = "unknown"
        best_similarity = 0.0
        
        for name, data in self.face_database.items():
            for stored_embedding in data["embeddings"]:
                similarity = np.dot(embedding, stored_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
        
        if best_similarity < self.similarity_threshold:
            return "unknown", best_similarity
        
        return best_match, best_similarity

    def add_face_to_database(self, name, embedding):
        """Добавление нового лица в базу данных"""
        if name not in self.face_database:
            self.face_database[name] = {
                "embeddings": [],
                "added_date": datetime.now().isoformat(),
                "count": 0
            }
        
        if len(self.face_database[name]["embeddings"]) < 5:
            self.face_database[name]["embeddings"].append(embedding.tolist())
            self.face_database[name]["count"] += 1
            print(f"Лицо '{name}' добавлено в базу данных")
            self.save_database()
        else:
            print(f"У '{name}' уже достаточно образцов в базе данных")

    def draw_results(self, frame, boxes, identities, similarities):
        """Отрисовка результатов на кадре"""
        for i, (box, identity, similarity) in enumerate(zip(boxes, identities, similarities)):
            x1, y1, x2, y2 = map(int, box)
            
            color = (0, 255, 0) if identity != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{identity} ({similarity:.2f})"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        db_status = f"DB: {len(self.face_database)} persons"
        cv2.putText(frame, db_status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        return frame

    def run(self, camera_id=0):
        """Основной цикл работы системы"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Ошибка: не удалось открыть веб-камеру!")
            print("Попробуйте:")
            print("  1. Проверить подключение камеры")
            print("  2. Указать другой индекс камеры (1, 2)")
            print("  3. Запустить с правами sudo (Linux)")
            return
        
        print("Система распознавания лиц запущена!")
        print("Управление:")
        print("  'a' - добавить текущее лицо в базу данных")
        print("  's' - показать текущую базу данных")
        print("  'd' - удалить последнее добавленное лицо")
        print("  'q' - выход")
        print(f"Загружено лиц из базы: {len(self.face_database)}")
        for name, data in self.face_database.items():
            print(f"  👤 {name}: {data['count']} образцов")
        
        current_name = ""
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка захвата кадра")
                break
            
            boxes, embeddings = self.detect_faces(frame)
            
            identities = []
            similarities = []
            
            for embedding in embeddings:
                identity, similarity = self.identify_face(embedding)
                identities.append(identity)
                similarities.append(similarity)
            
            frame = self.draw_results(frame, boxes, identities, similarities)
            
            if current_name:
                cv2.putText(frame, f"Adding: {current_name}", (10, frame.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "Press ENTER to confirm", (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            cv2.imshow('Face Recognition System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('a') and len(boxes) > 0:
                identity, similarity = self.identify_face(embeddings[0])
                if identity != "unknown":
                    print(f"Лицо уже известно: {identity}")
                else:
                    print("\nВведите имя для добавления (английские буквы):")
                    current_name = input("Имя: ").strip()
                    if current_name:
                        self.add_face_to_database(current_name, embeddings[0])
                    else:
                        print("Имя не введено")
                    current_name = ""
            elif key == ord('s'):
                print("Текущая база данных")
                if not self.face_database:
                    print("База данных пуста")
                else:
                    for name, data in self.face_database.items():
                        print(f"👤 {name}:")
                        print(f"Добавлено: {data['added_date']}")
                        print(f"Образцов: {data['count']}")
            elif key == ord('d'):
                if self.face_database:
                    last_name = list(self.face_database.keys())[-1]
                    del self.face_database[last_name]
                    self.save_database()
                    print(f"Лицо '{last_name}' удалено из базы данных")
                else:
                    print("База данных пуста")
            elif key == ord('c'):
                confirm = input("Очистить всю базу данных? (y/n): ").strip().lower()
                if confirm == 'y':
                    self.face_database = {}
                    self.save_database()
                    print("База данных очищена")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Работа системы завершена")

def main():
    """Основная функция"""
    system = FaceRecognitionSystem(database_path="face_database.json")
    system.run(camera_id=0)

if __name__ == "__main__":
    main()
