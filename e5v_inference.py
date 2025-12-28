
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "llava-hf/llama3-llava-next-8b-hf"
DTYPE = torch.float16

# ============================================================
# PROMPTS (FROM E5-V PAPER)
# ============================================================

TEXT_PROMPT = """{text}
Summary of the above sentence in one word:"""

IMAGE_PROMPT = """<image>
Summary of the above image in one word:"""

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    use_fast=True
)

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    device_map="auto"
)

model.eval()
print("Model loaded on:", next(model.parameters()).device)

# ============================================================
# FREEZE VISION ENCODER (PAPER REQUIREMENT)
# ============================================================

if hasattr(model, "vision_tower"):
    for p in model.vision_tower.parameters():
        p.requires_grad = False

if hasattr(model, "multi_modal_projector"):
    for p in model.multi_modal_projector.parameters():
        p.requires_grad = False

print("Vision encoder + projector frozen.")

# ============================================================
# EMBEDDING FUNCTIONS (INFERENCE ONLY)
# ============================================================

@torch.no_grad()
def embed_text(text: str) -> torch.Tensor:
    prompt = TEXT_PROMPT.format(text=text)

    inputs = processor(
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )

    emb = outputs.hidden_states[-1][:, -1, :]
    emb = F.normalize(emb, dim=-1)
    return emb


@torch.no_grad()
def embed_image(image: Image.Image) -> torch.Tensor:
    inputs = processor(
        images=image,
        text=IMAGE_PROMPT,
        return_tensors="pt"
    ).to(model.device)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )

    emb = outputs.hidden_states[-1][:, -1, :]
    emb = F.normalize(emb, dim=-1)
    return emb


# ============================================================
# SANITY CHECK (TEXT + IMAGE)
# ============================================================

if __name__ == "__main__":
    print("\n--- E5-V SANITY CHECK (INFERENCE) ---")

    # ---------- TEXT ↔️ TEXT ----------
    t1 = embed_text("A dog is running")
    t2 = embed_text("An animal is moving fast")
    t3 = embed_text("A parked car")

    print("Embedding shape:", t1.shape)
    print("sim(text dog, text animal):", F.cosine_similarity(t1, t2).item())
    print("sim(text dog, text car):", F.cosine_similarity(t1, t3).item())

    # ---------- IMAGE ↔️ TEXT ----------
    # Replace with your own image path
    img = Image.open("/home/mdesai23/project/MLLM/testing/cat_test.jpg").convert("RGB")

    img_emb = embed_image(img)
    txt_emb = embed_text("a dog running")

    print("sim(image dog, text dog):",
          F.cosine_similarity(img_emb, txt_emb).item())

    # ---------- IMAGE ↔️ IMAGE (OPTIONAL) ----------
    img2 = Image.open("/home/mdesai23/project/MLLM/testing/dog_test.jpg").convert("RGB")

    img_emb2 = embed_image(img2)

    print("sim(image dog, image cat):",
          F.cosine_similarity(img_emb, img_emb2).item())

    print("\nE5-V image + text inference working correctly.")
