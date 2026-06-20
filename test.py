# smoke_test.py
import torch
from PIL import Image
import numpy as np

# These imports tell you immediately if Copilot's module structure has any issues
from models.embedding_model import load_model
from pipeline.preprocess import get_transform, preprocess_single

print("--- Model Load Test ---")
model = load_model(backbone="resnet50")
print(f"Model loaded | Device: {model.device if hasattr(model, 'device') else 'check manually'}")

print("\n--- Single Image Embedding Test ---")
# Create a random RGB image (no real image needed yet)
dummy_image = Image.fromarray(np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8))

transform = get_transform()
tensor = transform(dummy_image.convert("RGB")).unsqueeze(0)  # (1, 3, 224, 224)
print(f"Input tensor shape: {tensor.shape}")  # Should be: torch.Size([1, 3, 224, 224])

with torch.no_grad():
    embedding = model(tensor)

print(f"Output embedding shape: {embedding.shape}")   # Should be: torch.Size([1, 2048])
print(f"Embedding norm: {embedding.norm().item():.6f}")  # Should be ~1.0 (L2-normalized)

# Verify normalization
assert abs(embedding.norm().item() - 1.0) < 1e-5, "FAIL: Embeddings are not L2-normalized!"
print("\n✅ Smoke test passed. Model is working correctly.")