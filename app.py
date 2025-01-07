import cv2
import numpy as np
from scipy.spatial.distance import cosine

# Haarcascade yuzni aniqlash uchun modelni yuklash
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Yuzni kesib olish va o'lchamini o'zgartirish funksiyasi
def preprocess_face(image, target_size=(150, 150)):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Yuzni oq-qora formatga o'tkazish
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Faqat birinchi aniqlangan yuzni qaytaradi
        face = gray_image[y:y+h, x:x+w]  # Yuzni kesib olish
        resized_face = cv2.resize(face, target_size)  # Belgilangan o'lchamga o'zgartirish
        return resized_face.flatten()  # Bitta o'lchamli massivga aylantirish
    return None

# Ma'lumotlar bazasidagi rasmni yuklash va yuzni oldindan qayta ishlash
reference_image_path = "reference.jpg"
reference_image = cv2.imread(reference_image_path)
if reference_image is None:
    raise FileNotFoundError(f"Rasm topilmadi: {reference_image_path}")

reference_face = preprocess_face(reference_image)
if reference_face is None:
    raise ValueError("Ma'lumotlar bazasidagi yuz aniqlanmadi!")

# Kamerani ishga tushirish
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Kamera ochilmadi.")
else:
    print("Kamera ochildi.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Har bir kadrda yuzni aniqlash va oldindan qayta ishlash
    current_face = preprocess_face(frame)
    if current_face is not None:
        # O'xshashlikni baholash
        similarity = 1 - cosine(reference_face, current_face)
        match_text = "Mos keladi" if similarity > 0.6 else "Mos kelmaydi"

        # Natijani oynaga chiqarish
        cv2.putText(frame, f"Similarity: {similarity:.2f} ({match_text})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if similarity > 0.6 else (0, 0, 255), 2)

    # Natijani ko'rsatish
    cv2.imshow("Face ID Check", frame)

    # 'q' tugmasi bosilganda chiqish
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tizimni tozalash
video_capture.release()
cv2.destroyAllWindows()




#  import cv2
# import mediapipe as mp
# import numpy as np
# from scipy.spatial.distance import cosine
# import os
# import tensorflow as tf
# import logging

# # Ogohlantirishlarni o'chirish
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow loglarini o'chirish
# logging.getLogger('tensorflow').setLevel(logging.FATAL)  # TensorFlow ogohlantirishlari

# # Mediapipe loglarini o'chirish
# import absl.logging
# absl.logging.set_verbosity(absl.logging.ERROR)

# # Mediapipe modullarini sozlash
# mp_face_detection = mp.solutions.face_detection
# mp_face_mesh = mp.solutions.face_mesh

# face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# reference_landmarks = None  # Ma'lumotlar bazasidagi yuz xususiyatlarini saqlash

# # Yuz xususiyatlarini chiqaruvchi funksiya
# def extract_face_landmarks(image, face_mesh_model):
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh_model.process(rgb_image)
#     if results.multi_face_landmarks:
#         return np.array([[p.x, p.y, p.z] for p in results.multi_face_landmarks[0].landmark])
#     return None

# # '/start' komandasi
# def start(update, context):
#     update.message.reply_text("Assalomu alaykum! Menga asosiy rasmni yuboring (reference.jpg).")

# # Rasmni qabul qilish va saqlash
# def receive_reference_image(update, context):
#     global reference_landmarks
#     photo = update.message.photo[-1].get_file()
#     photo.download("reference.jpg")
    
#     # Rasmni yuklash va yuzni olish
#     reference_image = cv2.imread("reference.jpg")
#     if reference_image is None:
#         update.message.reply_text("Yuklangan rasmni ochib bo'lmadi!")
#         return

#     reference_landmarks = extract_face_landmarks(reference_image, face_mesh)
#     if reference_landmarks is None:
#         update.message.reply_text("Rasmda yuz aniqlanmadi. Boshqa rasm yuboring.")
#     else:
#         update.message.reply_text("Rasm saqlandi! Endi kamerani ishga tushirish uchun /check komandasini bosing.")

# # Kamerani ishga tushirish va yuzni tekshirish
# def check_face(update, context):
#     global reference_landmarks
#     if reference_landmarks is None:
#         update.message.reply_text("Avval asosiy rasmni yuboring!")
#         return

#     video_capture = cv2.VideoCapture(0)
#     update.message.reply_text("Kamera ishga tushdi. Yuzingizni aniqlash uchun bir oz kuting.")

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break

#         landmarks = extract_face_landmarks(frame, face_mesh)

#         if landmarks is not None:
#             similarity = 1 - cosine(reference_landmarks.flatten(), landmarks.flatten())
#             match_text = "Mos keladi" if similarity > 0.6 else "Mos kelmaydi"

#             h, w, _ = frame.shape
#             cv2.putText(frame, f"Similarity: {similarity:.2f} ({match_text})", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if similarity > 0.6 else (0, 0, 255), 2)

#         cv2.imshow("Face ID Check", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video_capture.release()
#     cv2.destroyAllWindows()
#     update.message.reply_text("Kamera o'chirildi. 'q' tugmasi bosildi.")

# # Telegram botni sozlash
# def main():
#     # Telegram bot tokenini kiriting
#     TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

#     from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
#     updater = Updater(TOKEN)
#     dispatcher = updater.dispatcher

#     # Komandalarni ro'yxatdan o'tkazish
#     dispatcher.add_handler(CommandHandler("start", start))
#     dispatcher.add_handler(MessageHandler(Filters.photo, receive_reference_image))
#     dispatcher.add_handler(CommandHandler("check", check_face))

#     # Botni ishga tushirish
#     updater.start_polling()
#     updater.idle()

# if __name__ == "__main__":
#     main()



# import cv2
# import mediapipe as mp
# import numpy as np
# from scipy.spatial.distance import cosine
# import os
# import absl.logging

# # Ogohlantirishlarni o'chirish
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# absl.logging.set_verbosity(absl.logging.ERROR)

# # Mediapipe modullarini sozlash
# mp_face_detection = mp.solutions.face_detection
# mp_face_mesh = mp.solutions.face_mesh

# # Yuzni aniqlash va xususiyatlarini olish uchun Mediapipe modelini ishga tushirish
# face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# # Yuz xususiyatlarini chiqish funksiyasi
# def extract_face_landmarks(image, face_mesh_model):
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh_model.process(rgb_image)
#     if results.multi_face_landmarks:
#         return np.array([[p.x, p.y, p.z] for p in results.multi_face_landmarks[0].landmark])
#     return None

# # Ma'lumotlar bazasidagi yuzni yuklash
# reference_image_path = "reference.jpg"
# reference_image = cv2.imread(reference_image_path)
# if reference_image is None:
#     raise FileNotFoundError(f"Rasm topilmadi: {reference_image_path}")

# reference_landmarks = extract_face_landmarks(reference_image, face_mesh)
# if reference_landmarks is None:
#     raise ValueError("Ma'lumotlar bazasidagi yuz aniqlanmadi!")

# # Kamerani yoqish
# video_capture = cv2.VideoCapture(0)

# while True:
#     ret, frame = video_capture.read()
#     if not ret:
#         break

#     # Kadrdan yuzni aniqlash
#     landmarks = extract_face_landmarks(frame, face_mesh)

#     if landmarks is not None:
#         # Yuzlarni solishtirish
#         similarity = 1 - cosine(reference_landmarks.flatten(), landmarks.flatten())
#         match_text = "Mos keladi" if similarity > 0.6 else "Mos kelmaydi"

#         # Natijani kadrda ko'rsatish
#         h, w, _ = frame.shape
#         cv2.putText(frame, f"Similarity: {similarity:.2f} ({match_text})", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if similarity > 0.6 else (0, 0, 255), 2)

#     # Natijani oynada ko'rsatish
#     cv2.imshow("Face ID Check", frame)

#     # 'q' tugmasi bosilganda chiqish
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Tizimni tozalash
# video_capture.release()
# cv2.destroyAllWindows()

# # import os
# # import cv2
# # import mediapipe as mp
# # import absl.logging

# # # Ogohlantirish loglarini o'chirish
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow log darajasi
# # absl.logging.set_verbosity(absl.logging.ERROR)  # Mediapipe log darajasi

# # # Mediapipe yuz aniqlash moduli
# # mp_face_detection = mp.solutions.face_detection
# # mp_drawing = mp.solutions.drawing_utils

# # # Mediapipe modelini ishga tushirish
# # face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# # # Kamerani ishga tushirish
# # video_capture = cv2.VideoCapture(0)

# # while True:
# #     # Kameradan kadr olish
# #     ret, frame = video_capture.read()
# #     if not ret:
# #         break

# #     # Mediapipe RGB formatda ishlaydi
# #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #     # Yuzni aniqlash
# #     results = face_detection.process(rgb_frame)

# #     # Aniqlangan yuzlarni belgilash
# #     if results.detections:
# #         for detection in results.detections:
# #             mp_drawing.draw_detection(frame, detection)

# #     # Natijani ko'rsatish
# #     cv2.imshow('Video', frame)

# #     # 'q' tugmasini bosib chiqish
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # # Kamerani yopish
# # video_capture.release()
# # cv2.destroyAllWindows()
