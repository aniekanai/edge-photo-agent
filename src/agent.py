import requests
import base64
import cv2
import os
import threading

# NVIDIA Nemotron API Configuration
NEMOTRON_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NEMOTRON_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
API_KEY_ENV = "NVIDIA_API_KEY"
API_TIMEOUT_SECONDS = 5.0


def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


# ---------------------------
# LOCAL FAST CV BRAIN
# ---------------------------
def local_guidance(metrics):
    """
    Instant guidance. This runs every frame.
    NO internet.
    """

    if not metrics["face_detected"]:
        return "Center subject"

    if not metrics["face_centered"]:
        return "Move to center"

    if metrics["brightness"] < 80:
        return "Increase lighting"

    if metrics["sharpness"] < 400:
        return "Hold camera steady"

    return "READY"


# ---------------------------
# NEMOTRON REFINEMENT
# ---------------------------
def nemotron_refinement(frame, metrics):
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        return ""

    image_b64 = encode_frame_to_base64(frame)

    system_prompt = (
        "You are an AI photography co-pilot assisting a photographer. "
        "Return ONE short instruction (max 5 words). "
        "If no problem exists, return OK."
    )

    user_prompt = """
Look for high-level composition issues:
- subject looking away
- head turned sideways
- obstruction in front
- poor camera angle
- distracting background

Return ONE short instruction.
Examples:
Ask subject face forward
Raise camera slightly
Remove foreground object
OK
"""

    payload = {
        "model": NEMOTRON_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            },
        ],
        "max_tokens": 25,
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            NEMOTRON_API_URL,
            headers=headers,
            json=payload,
            timeout=API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result = response.json()
        text = result["choices"][0]["message"]["content"].strip()

        if text.upper() == "OK":
            return ""

        return text

    except Exception:
        return ""


# ---------------------------
# BACKGROUND THREAD SYSTEM
# ---------------------------

latest_refinement = ""
refinement_lock = threading.Lock()


def start_nemotron_background(frame, metrics):
    def worker():
        global latest_refinement
        result = nemotron_refinement(frame, metrics)
        with refinement_lock:
            latest_refinement = result

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def get_latest_refinement():
    with refinement_lock:
        return latest_refinement