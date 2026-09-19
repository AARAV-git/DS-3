import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import uniform_filter, laplace
import colorsys

images = {
    'AI/face':       'dataset/train/ai/face/AI_face_001.png',
    'AI/human_001':  'dataset/train/ai/human/AI_human_001.jpg',
    'AI/human_002':  'dataset/train/ai/human/AI_human_002.png',
    'AI/cat':        'dataset/train/ai/animal/cat/AI_cat_001.png',
    'REAL/human':    'dataset/train/real/human/real_human_001.jpg',
    'REAL/face':     'dataset/train/real/face/real_faces_001.png',
    'REAL/cat':      'dataset/train/real/animal/cat/real_cat_001.jpg',
    'EDITED/face':   'dataset/train/ai_edited/face/AI_edited_face_001.png',
    'EDITED/human':  'dataset/train/ai_edited/human/AI_edited_human_001.png',
    'EDITED/cat055': 'dataset/train/ai_edited/animal/cat/AI_edited_Cat_055.jpg',
    'EDITED/cat057': 'dataset/train/ai_edited/animal/cat/AI_edited_Cat_057.png',
}

header = "%-16s %10s %12s %12s %10s %9s %8s %12s" % (
    'Label', 'Noise_Std', 'LocalVar_Mu', 'GradMag_Mu', 'BlurScore', 'SatMean', 'SatStd', 'Size'
)
print(header)
print('-' * 100)

for label, path in images.items():
    img = Image.open(path).convert('RGB')
    W, H = img.size
    arr = np.array(img).astype(np.float32)

    # 1. Sensor noise (high-freq residual after blur)
    blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=1))).astype(np.float32)
    noise_std = float(np.std(arr - blurred))

    # 2. Local variance (texture richness)
    gray = np.array(img.convert('L')).astype(np.float32)
    local_mean = uniform_filter(gray, size=8)
    local_var  = uniform_filter(gray**2, size=8) - local_mean**2
    local_var_mu = float(np.mean(np.sqrt(np.maximum(local_var, 0))))

    # 3. Gradient magnitude (edge sharpness)
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    grad_mu = float(np.mean(np.sqrt(gx**2 + gy**2)))

    # 4. Blur score (Laplacian variance)
    blur_score = float(np.var(laplace(gray)))

    # 5. Colour saturation
    hsv = np.array([colorsys.rgb_to_hsv(r/255, g/255, b/255)
                    for r, g, b in arr.reshape(-1, 3)])
    sat_mean = float(np.mean(hsv[:, 1]))
    sat_std  = float(np.std(hsv[:, 1]))

    size_str = "%dx%d" % (W, H)
    row = "%-16s %10.3f %12.3f %12.3f %10.2f %9.4f %8.4f %12s" % (
        label, noise_std, local_var_mu, grad_mu, blur_score, sat_mean, sat_std, size_str
    )
    print(row)
