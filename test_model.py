
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

model = tf.keras.models.load_model(
    "models/cats_dogs_classifier.keras"
)

image_path = input("Enter image path: ").strip().strip('"')

if not Path(image_path).is_file():
    print("Error: Image file nahi mili. Path check karo.")
else:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    dog_probability = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    if dog_probability >= 0.5:
        prediction = "Dog"
        confidence = dog_probability * 100
    else:
        prediction = "Cat"
        confidence = (1 - dog_probability) * 100

    print("Prediction:", prediction)
    print(f"Confidence: {confidence:.2f}%")
