import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import matplotlib.pyplot as plt

# إعداد مسارات المجلدات
train_dir = r"C:\Users\shams\Documents\drawing\dataset\train"
val_dir = r"C:\Users\shams\Documents\drawing\dataset\val"
test_dir = r"C:\Users\shams\Documents\drawing\dataset\test"

# إعداد الحجم والدفعة
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# تحميل بيانات التدريب والتحقق
train_dataset = image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'  # إذا كان لدينا مشاعر متعددة
)

# استخراج أسماء الفئات
class_names = train_dataset.class_names
print(f"Class Names: {class_names}")

val_dataset = image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# تحميل بيانات الاختبار
test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# تحسين الأداء (Prefetching)
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# تحميل النموذج الأساسي MobileNetV2
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# تجميد الطبقات الأساسية
base_model.trainable = False

# بناء النموذج
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # تحويل المخرجات إلى شكل مسطح
    layers.Dropout(0.5),              # تقليل الإفراط في التعميم
    layers.Dense(256, activation='relu'),  # طبقة مخفية
    layers.Dense(len(class_names), activation='softmax')  # طبقة الإخراج
])

# تلخيص النموذج
model.summary()

# إعداد النموذج للتدريب
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# تدريب النموذج
EPOCHS = 100  # حاول زيادة عدد الفترات (epochs) لتحسين الأداء
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# تقييم النموذج على بيانات الاختبار
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.2f}")

# حفظ النموذج
model.save(r"C:\Users\shams\Documents\drawing\model_drawing_analysis.h5")
print("Model saved successfully!")

# اختبار النموذج على مجموعة الاختبار
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.2f}")

# حفظ النموذج
model.save(r"C:\Users\shams\Documents\drawing\model_drawing_analysis.keras")
