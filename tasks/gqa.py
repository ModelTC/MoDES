from datasets import load_dataset

GQA_RAW_IMAGE_DATASET = None
GQA_ID2IMAGE = None


def gqa_doc_to_visual(doc):
    """Extract image from GQA document by imageId.

    Args:
        doc: Dict with 'imageId' key.

    Returns:
        List containing the PIL image in RGB format.
    """
    global GQA_RAW_IMAGE_DATASET
    global GQA_ID2IMAGE
    if GQA_RAW_IMAGE_DATASET is None:
        GQA_RAW_IMAGE_DATASET = load_dataset(
            "storage/datasets/GQA/testdev_balanced_images", token=True
        )["train"]
        GQA_ID2IMAGE = {}
        for row in GQA_RAW_IMAGE_DATASET:
            GQA_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    image = GQA_ID2IMAGE[doc["imageId"]]
    return [image]


def gqa_doc_to_text(doc):
    """Format question as model input prompt for GQA.

    Args:
        doc: Dict with 'question' key.

    Returns:
        Formatted question string with answer instruction.
    """
    question = doc["question"]
    return f"{question}\nAnswer the question using a single word or phrase."


def gqa_doc_to_org_text(doc):
    """Return the raw question from doc.

    Args:
        doc: Dict with 'question' key.

    Returns:
        Original question string.
    """
    return doc["question"]


def gqa_doc_to_answer(doc):
    """Extract short answer from doc.

    Args:
        doc: Dict with 'answer' key.

    Returns:
        Short answer string.
    """
    return doc["answer"]


def gqa_doc_to_full_answer(doc):
    """Extract full answer from doc.

    Args:
        doc: Dict with 'fullAnswer' key.

    Returns:
        Full answer string.
    """
    return doc["fullAnswer"]


def gqa_transform(batch):
    """Transform a GQA batch into model input format.

    Args:
        batch: Dict with keys imageId, question, answer, fullAnswer.

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
    for img_id, q_text, a_text, f_a_text in zip(
        batch["imageId"], batch["question"], batch["answer"], batch["fullAnswer"]
    ):
        doc = {
            "imageId": img_id,
            "question": q_text,
            "answer": a_text,
            "fullAnswer": f_a_text,
        }
        visuals = gqa_doc_to_visual(doc)
        text = gqa_doc_to_text(doc)
        org_text = gqa_doc_to_org_text(doc)
        answer = gqa_doc_to_answer(doc)
        full_answer = gqa_doc_to_full_answer(doc)
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
