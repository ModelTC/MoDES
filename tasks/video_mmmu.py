from typing import List, Optional, Tuple, Union, Dict
from PIL import Image
import decord
import os
import sys
from pathlib import Path
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HOME = "storage/datasets/VideoMMMU"


def get_cache_dir(subject):
    """Map subject to cache subdirectory (Art, Science, Humanities, etc.).

    Args:
        subject: VideoMMMU subject name (e.g., Biology, History).

    Returns:
        Cache directory name string.

    Raises:
        ValueError: If subject is not recognized.
    """
    if subject in ["Art", "Art_Theory", "Design", "Music"]:
        return "Art"
    elif subject in ["Biology", "Chemistry", "Geography", "Math", "Physics"]:
        return "Science"
    elif subject in ["History", "Literature", "Sociology", "Psychology"]:
        return "Humanities"
    elif subject in [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ]:
        return "Engineering"
    elif subject in [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ]:
        return "Medicine"
    elif subject in ["Accounting", "Economics", "Finance", "Manage", "Marketing"]:
        return "Business"
    else:
        raise ValueError(f"Subject {subject} not recognized.")


def _resize_frame(frame: Image.Image, max_long_side: int) -> Image.Image:
    """Resize frame so the longer side equals max_long_side, preserving aspect ratio.

    Args:
        frame: PIL Image to resize.
        max_long_side: Maximum length of the longer side in pixels.

    Returns:
        Resized PIL Image.
    """
    w, h = frame.size
    if max(w, h) <= max_long_side:
        return frame
    if h > w:
        new_h = max_long_side
        aspect_ratio = w / h
        new_w = int(round(new_h * aspect_ratio))
    else:
        new_w = max_long_side
        aspect_ratio = h / w
        new_h = int(round(new_w * aspect_ratio))
    resized_frame = frame.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized_frame


def process_media(
    file_path: Union[str, Path], max_frames: int = 32, max_long_side: int = 480
) -> tuple:
    """Load and process video or image file into list of resized PIL frames.

    Args:
        file_path: Path to video (.mp4, .avi, etc.) or image (.jpg, .png, etc.).
        max_frames: Maximum number of frames to sample from video.
        max_long_side: Maximum length of the longer side when resizing.

    Returns:
        Tuple of (list of PIL Images, number of frames).

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If file type is not supported.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"No file found: {file_path}")
    file_ext = file_path.suffix.lower()
    processed_frames = []
    if file_ext in VIDEO_EXTENSIONS:
        vr = decord.VideoReader(str(file_path), ctx=decord.cpu(0))
        num_frames = len(vr)
        if num_frames <= max_frames:
            indices = list(range(num_frames))
        else:
            indices = np.linspace(0, num_frames - 1, max_frames, dtype=int).tolist()
            if (num_frames - 1) not in indices:
                indices.pop()
                indices.append(num_frames - 1)
        frames_np = vr.get_batch(indices).asnumpy()
        for frame_np in frames_np:
            pil_frame = Image.fromarray(frame_np)
            resized_frame = _resize_frame(pil_frame, max_long_side)
            processed_frames.append(resized_frame)
    elif file_ext in IMAGE_EXTENSIONS:
        with Image.open(file_path) as img:
            img_rgb = img.convert("RGB")
            resized_image = _resize_frame(img_rgb, max_long_side)
            processed_frames.append(resized_image)
    else:
        raise ValueError(
            f"Not supported file type: '{file_ext}'. "
            f"Supported video formats: {VIDEO_EXTENSIONS}, "
            f"Supported image formats: {IMAGE_EXTENSIONS}"
        )
    return processed_frames, len(processed_frames)


def videommmu_doc_to_visual(doc):
    """Load video from VideoMMMU doc by id and return processed frames.

    Args:
        doc: Dict with 'id' key (video id without extension).

    Returns:
        Tuple of (list of PIL frames, frame count).
    """
    global HOME
    # Extract the subject between the first and last underscores
    subject = "_".join(doc["id"].split("_")[1:-1])

    # Get the appropriate cache directory based on the subject
    videommmu_cache_dir = os.path.join(HOME, get_cache_dir(subject))

    video_path = doc["id"] + ".mp4"
    video_path = os.path.join(videommmu_cache_dir, video_path)

    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return process_media(video_path)


def videommmu_doc_to_text_adaptation(doc):
    """Format Adaptation-style VideoMMMU prompt.

    Args:
        doc: Dict with 'question' and 'question_type'.

    Returns:
        Formatted prompt string.
    """
    pre_prompt = (
        "You should watch and learn the video content. Then apply what you learned to "
    )
    question = doc["question"]

    if doc["question_type"] == "multiple-choice":
        pre_prompt += "answer the following question. The image for this question is at the end of the video."
        # parsed_options = parse_options(doc["options"])
        # question += "\n" + parsed_options
    else:
        pre_prompt += "answer the following open-ended question. The image for this question is at the end of the video."

    return f"{pre_prompt}{question}\nThe answer is: "


def videommmu_doc_to_text_perception_comprehension(doc):
    """Format Perception/Comprehension-style VideoMMMU prompt.

    Args:
        doc: Dict with 'question' key.

    Returns:
        Formatted prompt string.
    """
    post_prompt = "\nPlease ignore the Quiz question in last frame of the video."
    question = doc["question"]
    # parsed_options = parse_options(doc["options"])
    # question += "\n" + parsed_options

    return f"{question}{post_prompt}\nThe answer is: "


def videommmu_doc_to_org_text(doc):
    """Return raw question from doc.

    Args:
        doc: Dict with 'question' key.

    Returns:
        Original question string.
    """
    return doc["question"]


def videommmu_doc_to_answer(doc):
    """Extract answer from doc.

    Args:
        doc: Dict with 'answer' key.

    Returns:
        Answer string (option letter or free-form).
    """
    return doc["answer"]


def videommmu_doc_to_full_answer(doc):
    """Extract full answer text from doc (option text for MC, else raw answer).

    Args:
        doc: Dict with 'answer' and 'options' keys.

    Returns:
        Full answer string.
    """
    options = doc["options"]
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    if doc["answer"] in option_letters:
        return f"{options[option_letters.index(doc['answer'])]}"
    else:
        assert (
            len(options) == 0
        ), f"doc {doc['id']} has options but answer is not in them"
        return doc["answer"]


def videommmu_transform(batch):
    """Transform a VideoMMMU batch into model input format.

    Args:
        batch: Dict with keys id, question, question_type, options, answer.

    Returns:
        Dict with model_input_text, model_input_visual, model_input_answer,
        model_input_full_answer, model_input_org_text, model_input_frames.
    """
    # e.g., batch['imageId'] = ['id1', 'id2', ...]
    processed_texts = []
    processed_visuals = []
    processed_answers = []
    processed_full_answers = []
    processed_org_texts = []
    frames = []
    for id, q_text, q_type, options, a_text in zip(
        batch["id"],
        batch["question"],
        batch["question_type"],
        batch["options"],
        batch["answer"],
    ):
        doc = {
            "id": id,
            "question": q_text,
            "question_type": q_type,
            "options": options,
            "answer": a_text,
        }
        visuals = videommmu_doc_to_visual(doc)
        if doc["question_type"].endswith("Adaptation") or doc["question_type"].endswith(
            "Analysis"
        ):
            text = videommmu_doc_to_text_adaptation(doc)
        else:
            text = videommmu_doc_to_text_perception_comprehension(doc)
        org_text = videommmu_doc_to_org_text(doc)
        answer = videommmu_doc_to_answer(doc)
        full_answer = videommmu_doc_to_full_answer(doc)
        processed_visuals.append(visuals[0])
        frames.append(visuals[1])
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
        "model_input_frames": frames,
    }


if __name__ == "__main__":
    from datasets import load_dataset, concatenate_datasets

    x = load_dataset("storage/datasets/VideoMMMU/Adaptation", token=True)["test"]
    y = load_dataset("storage/datasets/VideoMMMU/Comprehension", token=True)["test"]
    z = load_dataset("storage/datasets/VideoMMMU/Perception", token=True)["test"]
    # only preserve columns "id", "question", "question_type", "options", "answer", "fullAnswer"
    # x = x.map(lambda x: {"id": x["id"], "question": x["question"], "question_type": x["question_type"], "options": x["options"], "answer": x["answer"]})
    # y = y.map(lambda x: {"id": x["id"], "question": x["question"], "question_type": x["question_type"], "options": x["options"], "answer": x["answer"]})
    # z = z.map(lambda x: {"id": x["id"], "question": x["question"], "question_type": x["question_type"], "options": x["options"], "answer": x["answer"]})
    # condense to one dataset, x: Dataset, y: Dataset, z: Dataset
    x = concatenate_datasets([x, y, z])
    print(x)
