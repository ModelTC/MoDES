from datasets import load_dataset

COCO_RAW_IMAGE_DATASET = None
COCO_ID2IMAGE = None


def coco_doc_to_visual(doc):
    """Extract image from COCO document by imageId.

    Args:
        doc: Dict with 'imageId' key.

    Returns:
        List containing the PIL image in RGB format.
    """
    global COCO_RAW_IMAGE_DATASET
    global COCO_ID2IMAGE
    if COCO_RAW_IMAGE_DATASET is None:
        COCO_RAW_IMAGE_DATASET = load_dataset(
            "storage/datasets/COCO-Caption2017/data", token=True
        )["validation"]
        COCO_ID2IMAGE = {}
        for row in COCO_RAW_IMAGE_DATASET:
            COCO_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    image = COCO_ID2IMAGE[doc["imageId"]]
    return [image]


def coco_doc_to_text(doc):
    """Format caption prompt for COCO.

    Args:
        doc: Unused; kept for API consistency.

    Returns:
        Caption instruction string.
    """
    return f"Provide a one-sentence caption for the provided image."


def coco_doc_to_org_text(doc):
    """Return the caption prompt as org text for COCO.

    Args:
        doc: Unused; kept for API consistency.

    Returns:
        Caption instruction string.
    """
    return f"Provide a one-sentence caption for the provided image."


def coco_doc_to_answer(doc):
    """Extract caption answer from doc.

    Args:
        doc: Dict with 'answer' key (list, first element used).

    Returns:
        First caption string from answer list.
    """
    return doc["answer"][0]


def coco_doc_to_full_answer(doc):
    return doc["answer"][0]


def coco_transform(batch):
    """Transform a COCO batch into model input format.

    Args:
        batch: Dict with keys id, answer.

    Returns:
        Dict with model_input_text, model_input_visual, model_input_answer,
        model_input_full_answer, model_input_org_text.
    """
    # e.g., batch['imageId'] = ['id1', 'id2', ...]
    processed_texts = []
    processed_visuals = []
    processed_answers = []
    processed_full_answers = []
    processed_org_texts = []
    for img_id, a_text in zip(batch["id"], batch["answer"]):
        doc = {
            "imageId": img_id,
            "answer": a_text,
        }
        visuals = coco_doc_to_visual(doc)
        text = coco_doc_to_text(doc)
        org_text = coco_doc_to_org_text(doc)
        answer = coco_doc_to_answer(doc)
        full_answer = coco_doc_to_full_answer(doc)
        processed_visuals.append(visuals[0])
        processed_texts.append(text)
        processed_answers.append(answer)
        processed_full_answers.append(full_answer)
        processed_org_texts.append(org_text)
    return {
        "model_input_text": processed_texts,
        "model_input_visual": processed_visuals,
        "model_input_answer": processed_answers,
        "model_input_full_answer": processed_full_answers,
        "model_input_org_text": processed_org_texts,
    }
