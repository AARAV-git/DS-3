@echo off
echo ============================================
echo  DS-MTFNet Project Setup
echo ============================================

:: Create root project folder
mkdir DS-MTFNet
cd DS-MTFNet

:: Dataset folders
mkdir dataset\train\ai\human
mkdir dataset\train\ai\face
mkdir dataset\train\ai\animal\cat
mkdir dataset\train\ai\animal\dog
mkdir dataset\train\ai\animal\elephant
mkdir dataset\train\ai\animal\horse
mkdir dataset\train\ai\animal\lion
mkdir dataset\train\real\human
mkdir dataset\train\real\face
mkdir dataset\train\real\animal\cat
mkdir dataset\train\real\animal\dog
mkdir dataset\train\real\animal\elephant
mkdir dataset\train\real\animal\horse
mkdir dataset\train\real\animal\lion
mkdir dataset\train\ai_edited\human
mkdir dataset\train\ai_edited\face
mkdir dataset\train\ai_edited\animal\cat
mkdir dataset\train\ai_edited\animal\dog
mkdir dataset\train\ai_edited\animal\elephant
mkdir dataset\train\ai_edited\animal\horse
mkdir dataset\train\ai_edited\animal\lion

mkdir dataset\val\ai\human
mkdir dataset\val\ai\face
mkdir dataset\val\ai\animal\cat
mkdir dataset\val\ai\animal\dog
mkdir dataset\val\ai\animal\elephant
mkdir dataset\val\ai\animal\horse
mkdir dataset\val\ai\animal\lion
mkdir dataset\val\real\human
mkdir dataset\val\real\face
mkdir dataset\val\real\animal\cat
mkdir dataset\val\real\animal\dog
mkdir dataset\val\real\animal\elephant
mkdir dataset\val\real\animal\horse
mkdir dataset\val\real\animal\lion
mkdir dataset\val\ai_edited\human
mkdir dataset\val\ai_edited\face
mkdir dataset\val\ai_edited\animal\cat
mkdir dataset\val\ai_edited\animal\dog
mkdir dataset\val\ai_edited\animal\elephant
mkdir dataset\val\ai_edited\animal\horse
mkdir dataset\val\ai_edited\animal\lion

mkdir dataset\test\ai\human
mkdir dataset\test\ai\face
mkdir dataset\test\ai\animal\cat
mkdir dataset\test\ai\animal\dog
mkdir dataset\test\ai\animal\elephant
mkdir dataset\test\ai\animal\horse
mkdir dataset\test\ai\animal\lion
mkdir dataset\test\real\human
mkdir dataset\test\real\face
mkdir dataset\test\real\animal\cat
mkdir dataset\test\real\animal\dog
mkdir dataset\test\real\animal\elephant
mkdir dataset\test\real\animal\horse
mkdir dataset\test\real\animal\lion
mkdir dataset\test\ai_edited\human
mkdir dataset\test\ai_edited\face
mkdir dataset\test\ai_edited\animal\cat
mkdir dataset\test\ai_edited\animal\dog
mkdir dataset\test\ai_edited\animal\elephant
mkdir dataset\test\ai_edited\animal\horse
mkdir dataset\test\ai_edited\animal\lion

:: Source code folders
mkdir src
mkdir checkpoints
mkdir logs
mkdir outputs

:: Create all Python source files
echo. > src\__init__.py

:: ── dataset.py ──────────────────────────────────────────────────────────────
(
echo import os
echo import torch
echo import numpy as np
echo from PIL import Image
echo from torch.utils.data import Dataset
echo import torchvision.transforms as T
echo import cv2
echo.
echo ORIGIN_CLASSES = {"ai": 0, "real": 1, "ai_edited": 2}
echo CONTENT_CLASSES = {"human": 0, "face": 1, "animal": 2}
echo.
echo def get_transforms(train=True^):
echo     if train:
echo         return T.Compose([
echo             T.Resize((224, 224^)^),
echo             T.RandomHorizontalFlip(^),
echo             T.RandomRotation(10^),
echo             T.ColorJitter(brightness=0.2, contrast=0.2^),
echo             T.ToTensor(^),
echo             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]^),
echo         ]^)
echo     return T.Compose([
echo         T.Resize((224, 224^)^),
echo         T.ToTensor(^),
echo         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]^),
echo     ]^)
echo.
echo def get_fft(img_pil^):
echo     img_np = np.array(img_pil.convert("L"^).resize((224, 224^)^)^).astype(np.float32^)
echo     fft = np.fft.fft2(img_np^)
echo     fft_shift = np.fft.fftshift(fft^)
echo     magnitude = np.log1p(np.abs(fft_shift^)^)
echo     magnitude = (magnitude - magnitude.min(^)^) / (magnitude.max(^) - magnitude.min(^) + 1e-8^)
echo     fft_tensor = torch.tensor(magnitude^).unsqueeze(0^).repeat(3, 1, 1^)
echo     return fft_tensor
echo.
echo class DSMTFNetDataset(Dataset^):
echo     def __init__(self, root, transform=None^):
echo         self.samples = []
echo         self.transform = transform
echo         for origin_name, origin_label in ORIGIN_CLASSES.items(^):
echo             for content_name, content_label in CONTENT_CLASSES.items(^):
echo                 folder = os.path.join(root, origin_name, content_name^)
echo                 if content_name == "animal":
echo                     for animal in ["cat", "dog", "elephant", "horse", "lion"]:
echo                         sub = os.path.join(folder, animal^)
echo                         if os.path.isdir(sub^):
echo                             for f in os.listdir(sub^):
echo                                 if f.lower(^).endswith((".jpg", ".jpeg", ".png"^)^):
echo                                     self.samples.append((os.path.join(sub, f^), origin_label, content_label^)^)
echo                 else:
echo                     if os.path.isdir(folder^):
echo                         for f in os.listdir(folder^):
echo                             if f.lower(^).endswith((".jpg", ".jpeg", ".png"^)^):
echo                                 self.samples.append((os.path.join(folder, f^), origin_label, content_label^)^)
echo.
echo     def __len__(self^):
echo         return len(self.samples^)
echo.
echo     def __getitem__(self, idx^):
echo         path, origin_label, content_label = self.samples[idx]
echo         img = Image.open(path^).convert("RGB"^)
echo         fft_tensor = get_fft(img^)
echo         if self.transform:
echo             img = self.transform(img^)
echo         return img, fft_tensor, origin_label, content_label
) > src\dataset.py

:: ── model.py ────────────────────────────────────────────────────────────────
(
echo import torch
echo import torch.nn as nn
echo from torchvision import models
echo.
echo class DSMTFNet(nn.Module^):
echo     def __init__(self, num_origin=3, num_content=3^):
echo         super(^).__init__(^)
echo         # RGB stream: EfficientNet-B3
echo         eff = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT^)
echo         self.rgb_stream = nn.Sequential(*list(eff.children(^)^)[:-1]^)
echo         rgb_features = 1536
echo         # FFT stream: ResNet-18
echo         res = models.resnet18(weights=models.ResNet18_Weights.DEFAULT^)
echo         self.fft_stream = nn.Sequential(*list(res.children(^)^)[:-1]^)
echo         fft_features = 512
echo         # Fusion
echo         self.fusion = nn.Sequential(
echo             nn.Linear(rgb_features + fft_features, 512^),
echo             nn.BatchNorm1d(512^),
echo             nn.ReLU(^),
echo             nn.Dropout(0.4^),
echo             nn.Linear(512, 256^),
echo             nn.BatchNorm1d(256^),
echo             nn.ReLU(^),
echo             nn.Dropout(0.3^),
echo         ^)
echo         # Classification heads
echo         self.origin_head = nn.Linear(256, num_origin^)
echo         self.content_head = nn.Linear(256, num_content^)
echo.
echo     def forward(self, rgb, fft^):
echo         r = self.rgb_stream(rgb^).flatten(1^)
echo         f = self.fft_stream(fft^).flatten(1^)
echo         fused = self.fusion(torch.cat([r, f], dim=1^)^)
echo         return self.origin_head(fused^), self.content_head(fused^)
) > src\model.py

:: ── train.py ────────────────────────────────────────────────────────────────
(
echo import os, torch, json
echo import torch.nn as nn
echo from torch.utils.data import DataLoader
echo from torch.optim.lr_scheduler import CosineAnnealingLR
echo from src.dataset import DSMTFNetDataset, get_transforms
echo from src.model import DSMTFNet
echo.
echo DEVICE = "cuda" if torch.cuda.is_available(^) else "cpu"
echo EPOCHS = 30
echo BATCH_SIZE = 16
echo LR = 1e-4
echo ORIGIN_WEIGHT = 0.6
echo CONTENT_WEIGHT = 0.4
echo.
echo def train(^):
echo     print(f"Training on: {DEVICE}"^)
echo     train_ds = DSMTFNetDataset("dataset/train", get_transforms(train=True^)^)
echo     val_ds   = DSMTFNetDataset("dataset/val",   get_transforms(train=False^)^)
echo     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2^)
echo     val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2^)
echo     model = DSMTFNet(^).to(DEVICE^)
echo     criterion = nn.CrossEntropyLoss(^)
echo     optimizer = torch.optim.Adam(model.parameters(^), lr=LR^)
echo     scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS^)
echo     best_val_acc = 0.0
echo     history = []
echo     for epoch in range(1, EPOCHS + 1^):
echo         model.train(^)
echo         total_loss = correct_o = correct_c = total = 0
echo         for rgb, fft, o_lbl, c_lbl in train_loader:
echo             rgb, fft = rgb.to(DEVICE^), fft.to(DEVICE^)
echo             o_lbl, c_lbl = o_lbl.to(DEVICE^), c_lbl.to(DEVICE^)
echo             optimizer.zero_grad(^)
echo             o_out, c_out = model(rgb, fft^)
echo             loss = ORIGIN_WEIGHT * criterion(o_out, o_lbl^) + CONTENT_WEIGHT * criterion(c_out, c_lbl^)
echo             loss.backward(^)
echo             optimizer.step(^)
echo             total_loss += loss.item(^)
echo             correct_o += (o_out.argmax(1^) == o_lbl^).sum(^).item(^)
echo             correct_c += (c_out.argmax(1^) == c_lbl^).sum(^).item(^)
echo             total += o_lbl.size(0^)
echo         scheduler.step(^)
echo         # Validation
echo         model.eval(^)
echo         val_correct_o = val_total = 0
echo         with torch.no_grad(^):
echo             for rgb, fft, o_lbl, c_lbl in val_loader:
echo                 rgb, fft = rgb.to(DEVICE^), fft.to(DEVICE^)
echo                 o_lbl = o_lbl.to(DEVICE^)
echo                 o_out, _ = model(rgb, fft^)
echo                 val_correct_o += (o_out.argmax(1^) == o_lbl^).sum(^).item(^)
echo                 val_total += o_lbl.size(0^)
echo         val_acc = val_correct_o / val_total
echo         train_acc = correct_o / total
echo         avg_loss = total_loss / len(train_loader^)
echo         print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"^)
echo         history.append({"epoch": epoch, "loss": avg_loss, "train_acc": train_acc, "val_acc": val_acc}^)
echo         if val_acc ^> best_val_acc:
echo             best_val_acc = val_acc
echo             torch.save(model.state_dict(^), "checkpoints/best_model.pth"^)
echo             print(f"  Saved best model (val_acc={val_acc:.4f}^)"^)
echo     with open("logs/history.json", "w"^) as f:
echo         json.dump(history, f, indent=2^)
echo     print("Training complete. Best val acc:", best_val_acc^)
echo.
echo if __name__ == "__main__":
echo     train(^)
) > train.py

:: ── evaluate.py ─────────────────────────────────────────────────────────────
(
echo import torch
echo from torch.utils.data import DataLoader
echo from sklearn.metrics import classification_report, confusion_matrix
echo from src.dataset import DSMTFNetDataset, get_transforms
echo from src.model import DSMTFNet
echo.
echo DEVICE = "cuda" if torch.cuda.is_available(^) else "cpu"
echo ORIGIN_NAMES  = ["ai", "real", "ai_edited"]
echo CONTENT_NAMES = ["human", "face", "animal"]
echo.
echo def evaluate(^):
echo     test_ds = DSMTFNetDataset("dataset/test", get_transforms(train=False^)^)
echo     loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2^)
echo     model = DSMTFNet(^).to(DEVICE^)
echo     model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE^)^)
echo     model.eval(^)
echo     all_o_true, all_o_pred = [], []
echo     all_c_true, all_c_pred = [], []
echo     with torch.no_grad(^):
echo         for rgb, fft, o_lbl, c_lbl in loader:
echo             rgb, fft = rgb.to(DEVICE^), fft.to(DEVICE^)
echo             o_out, c_out = model(rgb, fft^)
echo             all_o_true.extend(o_lbl.tolist(^)^)
echo             all_o_pred.extend(o_out.argmax(1^).cpu(^).tolist(^)^)
echo             all_c_true.extend(c_lbl.tolist(^)^)
echo             all_c_pred.extend(c_out.argmax(1^).cpu(^).tolist(^)^)
echo     print("=== Origin Classification ==="^)
echo     print(classification_report(all_o_true, all_o_pred, target_names=ORIGIN_NAMES^)^)
echo     print("=== Content Classification ==="^)
echo     print(classification_report(all_c_true, all_c_pred, target_names=CONTENT_NAMES^)^)
echo.
echo if __name__ == "__main__":
echo     evaluate(^)
) > evaluate.py

:: ── gradcam.py ──────────────────────────────────────────────────────────────
(
echo import torch
echo import torch.nn.functional as F
echo import numpy as np
echo import cv2
echo from PIL import Image
echo from src.dataset import get_transforms, get_fft
echo from src.model import DSMTFNet
echo.
echo DEVICE = "cuda" if torch.cuda.is_available(^) else "cpu"
echo.
echo class GradCAM:
echo     def __init__(self, model, target_layer^):
echo         self.model = model
echo         self.gradients = None
echo         self.activations = None
echo         target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o^)^)
echo         target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0]^)^)
echo.
echo     def generate(self, rgb, fft, class_idx=None, head="origin"^):
echo         self.model.zero_grad(^)
echo         o_out, c_out = self.model(rgb, fft^)
echo         out = o_out if head == "origin" else c_out
echo         if class_idx is None:
echo             class_idx = out.argmax(1^).item(^)
echo         out[0, class_idx].backward(^)
echo         weights = self.gradients.mean(dim=[2, 3], keepdim=True^)
echo         cam = (weights * self.activations^).sum(dim=1, keepdim=True^)
echo         cam = F.relu(cam^)
echo         cam = F.interpolate(cam, size=(224, 224^), mode="bilinear", align_corners=False^)
echo         cam = cam.squeeze(^).detach(^).cpu(^).numpy(^)
echo         cam = (cam - cam.min(^)^) / (cam.max(^) - cam.min(^) + 1e-8^)
echo         return cam
echo.
echo def run_gradcam(image_path, output_path="outputs/heatmap.jpg"^):
echo     img = Image.open(image_path^).convert("RGB"^)
echo     transform = get_transforms(train=False^)
echo     rgb = transform(img^).unsqueeze(0^).to(DEVICE^)
echo     fft = get_fft(img^).unsqueeze(0^).to(DEVICE^)
echo     model = DSMTFNet(^).to(DEVICE^)
echo     model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE^)^)
echo     model.eval(^)
echo     # Hook onto last conv block of EfficientNet-B3
echo     target_layer = list(model.rgb_stream.children(^)^)[0][-1]
echo     gcam = GradCAM(model, target_layer^)
echo     cam = gcam.generate(rgb, fft^)
echo     img_cv = cv2.imread(image_path^)
echo     img_cv = cv2.resize(img_cv, (224, 224^)^)
echo     heatmap = cv2.applyColorMap(np.uint8(255 * cam^), cv2.COLORMAP_JET^)
echo     overlay = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0^)
echo     cv2.imwrite(output_path, overlay^)
echo     print(f"Heatmap saved to {output_path}"^)
echo.
echo if __name__ == "__main__":
echo     import sys
echo     run_gradcam(sys.argv[1]^)
) > gradcam.py

:: ── api.py ───────────────────────────────────────────────────────────────────
(
echo from fastapi import FastAPI, UploadFile, File
echo from fastapi.responses import JSONResponse
echo import torch, io
echo from PIL import Image
echo from src.dataset import get_transforms, get_fft
echo from src.model import DSMTFNet
echo.
echo DEVICE = "cuda" if torch.cuda.is_available(^) else "cpu"
echo ORIGIN_NAMES  = ["ai", "real", "ai_edited"]
echo CONTENT_NAMES = ["human", "face", "animal"]
echo.
echo app = FastAPI(title="DS-MTFNet API"^)
echo model = DSMTFNet(^).to(DEVICE^)
echo model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE^)^)
echo model.eval(^)
echo transform = get_transforms(train=False^)
echo.
echo @app.post("/predict"^)
echo async def predict(file: UploadFile = File(...^)^):
echo     contents = await file.read(^)
echo     img = Image.open(io.BytesIO(contents^)^).convert("RGB"^)
echo     rgb = transform(img^).unsqueeze(0^).to(DEVICE^)
echo     fft = get_fft(img^).unsqueeze(0^).to(DEVICE^)
echo     with torch.no_grad(^):
echo         o_out, c_out = model(rgb, fft^)
echo     origin_probs  = torch.softmax(o_out, dim=1^)[0].tolist(^)
echo     content_probs = torch.softmax(c_out, dim=1^)[0].tolist(^)
echo     return JSONResponse({
echo         "origin":  {ORIGIN_NAMES[i]:  round(p, 4^) for i, p in enumerate(origin_probs^)},
echo         "content": {CONTENT_NAMES[i]: round(p, 4^) for i, p in enumerate(content_probs^)},
echo     }^)
) > api.py

:: ── requirements.txt ────────────────────────────────────────────────────────
(
echo torch
echo torchvision
echo torchaudio
echo numpy
echo opencv-python
echo Pillow
echo matplotlib
echo scikit-learn
echo fastapi
echo uvicorn
echo python-multipart
) > requirements.txt

:: ── README.txt ───────────────────────────────────────────────────────────────
(
echo DS-MTFNet — Quick Start
echo =======================
echo.
echo 1. Activate your environment:
echo    antigravity\Scripts\activate
echo.
echo 2. Install dependencies:
echo    pip install -r requirements.txt
echo.
echo 3. Put your images in dataset\train\, dataset\val\, dataset\test\
echo    following the folder structure already created.
echo.
echo 4. Start training:
echo    python train.py
echo.
echo 5. Evaluate on test set:
echo    python evaluate.py
echo.
echo 6. Generate a Grad-CAM heatmap:
echo    python gradcam.py path\to\image.jpg
echo.
echo 7. Run the API server:
echo    uvicorn api:app --reload
echo    Then open: http://127.0.0.1:8000/docs
) > README.txt

echo.
echo ============================================
echo  Done! Project created in .\DS-MTFNet\
echo ============================================
echo.
echo  Next steps:
echo  1. cd DS-MTFNet
echo  2. antigravity\Scripts\activate
echo  3. pip install -r requirements.txt
echo  4. Add your images to dataset folders
echo  5. python train.py
echo ============================================
