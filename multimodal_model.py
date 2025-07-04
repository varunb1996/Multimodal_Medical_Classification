import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision import models

class MultimodalClassifier(nn.Module):
    def __init__(self, text_model='bert-base-uncased', num_classes=5):
        super(MultimodalClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(text_model)
        self.text_fc = nn.Linear(self.bert.config.hidden_size, 256)
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.img_fc = nn.Linear(512, 256)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.text_fc(text_output.pooler_output)
        img_feat = self.cnn(image).squeeze()
        img_feat = self.img_fc(img_feat)
        combined = torch.cat((text_feat, img_feat), dim=1)
        logits = self.classifier(combined)
        return logits

if __name__ == "__main__":
    model = MultimodalClassifier(num_classes=3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = ["Patient shows signs of pulmonary infection."]
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    dummy_img = torch.randn(1, 3, 224, 224)
    output = model(tokens['input_ids'], tokens['attention_mask'], dummy_img)
    print("Logits:", output)
