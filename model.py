import torch
import torchvision.models
import torch.nn as nn
import numpy as np
import cv2

from PIL import Image
from torchvision import datasets, transforms
from class_names import class_names


class DogBreedClassifier:
    """A class for loading a pretrained model"""

    def __init__(self, model_file="hound_model_vgg.pt"):
        """Initialize the classifier

        Args:
            model_file (str, optional): Provide a trained model file. Defaults to "hound_model_vgg.pt".
        """
        self.model_file = model_file
        self.model = torchvision.models.vgg16(pretrained=True)
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, len(class_names))
        self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor(), self.normalize_transform]
        )

        self.face_cascade = face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

        self.vgg16 = torchvision.models.vgg16(pretrained=True)

    def load_model(self):
        """Load the model from a file"""
        print("Loading model...")
        self.model.load_state_dict(torch.load(self.model_file, map_location=torch.device("cpu")))
        self.model.eval()

    def predict_breed(self, img):
        """

        Args:
            img (PIL Image): Image data

        Returns:
            str: dog breed
        """
        imt = self.transform(img).unsqueeze(0)
        out = self.model(imt)
        out_idx = np.argmax(out.detach().numpy())
        return class_names[out_idx]

    def face_detector(self, img):
        """Detect whether there is a human face in the image

        Args:
            img (PIL Image): Image data

        Returns:
            bool: True for face in image
        """
        img = img.convert("RGB")
        cv_image = np.array(img)  # Convert RGB to BGR
        cv_image = cv_image[:, :, ::-1].copy()
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def vgg16_predict(self, img):
        """Vgg16 classifier

        Args:
            img (PIL Image): Image data

        Returns:
            int: vgg16 class
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        resize = transforms.Resize((250, 250))
        transformation = transforms.Compose([resize, transforms.ToTensor(), normalize])
        imgt = transformation(img).unsqueeze(0)
        self.vgg16.eval()
        result = self.vgg16(imgt)
        return np.argmax(result.detach().cpu().numpy())

    def dog_detector(self, img):
        """Detect whether there is a dog in the image

        Args:
            img (PIL Image): Image data

        Returns:
            bool: True for dog in image
        """
        object_id = self.vgg16_predict(img)
        if object_id >= 151 and object_id <= 268:
            return True

        return False
