import pickle
import numpy as np

from torchvision import models
from torch import nn

from models.utils import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

models_init={
"VGG16": models.vgg16(pretrained=True),
"mobNetv3small": models.mobilenet_v3_small(pretrained=True),
"mobNetLarge": models.mobilenet_v3_large(pretrained=True)
}

models_numfeatures={
"VGG16": 25088,
"mobNetv3small": 576,
"mobNetLarge": 960
}

model_weights={
    "VGG16": 'models/vgg16/pytorch_model.pth',
    "mobNetv3small": 'models/mobNetv3small/pytorch_model.pth',
    "mobNetLarge": 'models/mobNetLarge/pytorch_model.pth'
}

class MobNetSimpsons():
    def __init__(self, model_name):
        print("Loading model...")
        self.model = models_init[model_name]

        num_features = models_numfeatures[model_name]

        n_classes = 42

        if model_name == 'VGG16':
            self.model.classifier = nn.Sequential(
                        nn.Linear(in_features=num_features, out_features=4096, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5, inplace=False),
                        nn.Linear(in_features=4096, out_features=4096, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5, inplace=False),
                        nn.Linear(in_features=4096, out_features=n_classes))


        if model_name == 'mobNetLarge':
            self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, n_classes, bias=True) 
        )

        
        if model_name == 'mobNetv3small':
            self.model.classifier = nn.Sequential(
                                  nn.Linear(num_features, 1024, bias=True),
                                  nn.Hardswish(),
                                  nn.Dropout(p=0.2, inplace=True),
                                  nn.Linear(1024, n_classes, bias=True) 
        )

        print("Setting parameters...")
        self.model.load_state_dict(torch.load(model_weights[model_name], map_location=DEVICE))
        print("Setting on evaluation mode...")
        self.model.eval()

        self.label_encoder = pickle.loads(open('models/label_encoder.pkl', 'rb').read())
        print("Model is ready!")

    def predict(self, image_path):
        img = prepare_img(image_path)

        proba = predict_one_sample(self.model, img, device=DEVICE)
        predicted_proba = np.max(proba)*100
        y_pred = np.argmax(proba)

        label = self.label_encoder.inverse_transform([y_pred])[0].split('_')
        label = label_to_string(label)

        return [label, predicted_proba]


