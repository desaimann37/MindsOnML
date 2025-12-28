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
    return F.normalize(emb, dim=-1)


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
    return F.normalize(emb, dim=-1)

# ============================================================
# CONTRASTIVE RETRIEVAL QA (FIXED OPTION A)
# ============================================================

def answer_question_contrastive(
    positive_query: str,
    negative_query: str,
    image_embeddings: torch.Tensor,
    image_labels: list
):
    """
    Contrastive retrieval:
    score = sim(pos_query, image) - sim(neg_query, image)
    """

    pos_emb = embed_text(positive_query)
    neg_emb = embed_text(negative_query)

    pos_scores = F.cosine_similarity(pos_emb, image_embeddings)
    neg_scores = F.cosine_similarity(neg_emb, image_embeddings)

    scores = pos_scores - neg_scores
    best_idx = scores.argmax().item()

    return image_labels[best_idx], scores[best_idx].item()

# ============================================================
# DEMO / SANITY CHECK
# ============================================================

if __name__ == "__main__":
    print("\n--- E5-V CONTRASTIVE RETRIEVAL QA DEMO ---")

    # Load images
    img_dog = Image.open(
        "/home/mdesai23/project/MLLM/testing/dog_test.jpg"
    ).convert("RGB")

    img_cat = Image.open(
        "/home/mdesai23/project/MLLM/testing/cat_test.jpg"
    ).convert("RGB")

    # Embed images
    emb_dog = embed_image(img_dog)
    emb_cat = embed_image(img_cat)

    image_embeddings = torch.cat([emb_dog, emb_cat], dim=0)
    image_labels = ["dog.jpg", "cat.jpg"]

    # Contrastive questions
    queries = [
        ("an image of a dog", "an image of a cat"),
        ("an image of a cat", "an image of a dog"),
        ("an image of an animal", "an image of a vehicle"),
        # Semantic logic
        ("an image of a pet", "an image of an object"),
        ("a living thing", "a non-living object"),
        ("a domestic animal", "a wild animal"),
    ]

    for pos_q, neg_q in queries:
        label, score = answer_question_contrastive(
            pos_q,
            neg_q,
            image_embeddings,
            image_labels
        )

        print(f"Positive query: {pos_q}")
        print(f"Negative query: {neg_q}")
        print(f"→ Answer: {label} (score={score:.3f})\n")
