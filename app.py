from model import DogBreedClassifier
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)

dog_breed_classifier = DogBreedClassifier()
dog_breed_classifier.load_model()

# taken from https://gist.github.com/cdiener/10491632
def image_to_text(img):
    chars = np.asarray(list(" .,:;irsXA253hMHGS#9B&@"))
    SC, GCF, WCF = 0.1, 1, 7 / 4
    S = (round(img.size[0] * SC * WCF), round(img.size[1] * SC))
    img = np.sum(np.asarray(img.resize(S)), axis=2)
    img -= img.min()
    img = (1.0 - img / img.max()) ** GCF * (chars.size - 1)

    return "\n".join(("".join(r) for r in chars[img.astype(int)]))


@app.route("/", methods=["POST"])
def index():
    max_allowed_size = 5242880  # 5MB

    if request.content_length > max_allowed_size:
        return f"Request sent size {request.content_length} is bigger than allowed size {max_allowed_size}", 413

    file = request.files["image"]

    with Image.open(file) as imagefile:
        if dog_breed_classifier.dog_detector(imagefile):
            breed = dog_breed_classifier.predict_breed(imagefile)
            return "A {} recognized in the image.\n\n".format(breed) + "\n" + image_to_text(imagefile)
        elif dog_breed_classifier.face_detector(imagefile):
            breed = dog_breed_classifier.predict_breed(imagefile)
            return "Human face detected in the image. It looks like a: " + breed + "\n\n" + image_to_text(imagefile)
        else:
            return "The input doesn't seem to be an image of a human or a dog."
