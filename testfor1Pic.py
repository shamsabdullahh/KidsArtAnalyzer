from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# قائمة الفئات
classes = ["Sad","Fear" , "Happy", "Angry"]

# تحميل النموذج
model_path = r"C:\Users\shams\Documents\drawing\model_drawing_analysis.h5"
model = load_model(model_path)

# تحميل صورة اختبار
test_image_path = r"C:\Users\shams\Documents\drawing\happyPic.jpg"# تأكد من مسار الصورة
image = load_img(test_image_path, target_size=(224, 224))  # تأكد من استخدام نفس الحجم الذي تدرب عليه النموذج
image_array = img_to_array(image) / 255.0  # تطبيع الصورة
image_array = np.expand_dims(image_array, axis=0)

# التنبؤ بالفئة
prediction = model.predict(image_array)
class_index = np.argmax(prediction)
confidence = np.max(prediction)

# طباعة اسم الفئة والثقة
predicted_class = classes[class_index]
print(f"Your Kid is: {predicted_class} with {confidence*100:.2f}% confidence")

# عرض الصورة
plt.imshow(image)
plt.title(f"Your child's feeling is: {predicted_class}")
plt.show()
