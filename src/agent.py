

import requests
import base64
import cv2
import time
import numpy as np

# ==============================
# CONFIG
# ==============================

NEMOTRON_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NEMOTRON_MODEL = "nvidia/nemotron-nano-12b-v2-vl"

API_TIMEOUT_SECONDS = 2.0  # HARD timeout (very important)
API_KEY_ENV = "NVIDIA_API_KEY"

# ==============================
# UTILS
# ==============================

def encode_frame_to_base64(frame):
    """
    Encode an OpenCV frame to base64 JPEG.
    """
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


# ==============================
# FALLBACK GUIDANCE (RULE-BASED)
# ==============================

def fallback_guidance(metrics):
    """
    Safe rule-based guidance if Nemotron fails.
    """
    if not metrics["face_detected"]:
        return "Ensure your face is visible in the frame."

    if not metrics["face_centered"]:
        return "Move slightly toward the center of the frame."

    if metrics["brightness"] < 80:
        return "Increase lighting or move to a brighter area."

    if metrics["sharpness"] < 150:
        return "Hold still to reduce motion blur."

    return "Adjust position slightly for optimal framing."


# ==============================
# NEMOTRON GUIDANCE
# ==============================

def nemotron_guidance(frame, metrics):
    """
    Call Nemotron to generate actionable guidance.
    Falls back safely if API fails.
    """

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        return fallback_guidance(metrics)

    image_b64 = encode_frame_to_base64(frame)

    system_prompt = (
        "You are an AI photo assistant. "
        "Your task is to provide short, actionable guidance "
        "to help a user adjust their position or environment "
        "to take a better photo."
    )

    user_prompt = f"""
System observations:
- Brightness: {metrics['brightness']:.1f}
- Sharpness: {metrics['sharpness']:.1f}
- Face detected: {metrics['face_detected']}
- Face centered: {metrics['face_centered']}
- Quality score: {metrics['quality_score']}/100

Analyze the image and provide ONE short instruction
(e.g., 'Move slightly left', 'Remove object blocking face', 'Hold still').
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
        "max_tokens": 50,
        "temperature": 0.3,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
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

        return result["choices"][0]["message"]["content"].strip()

    except Exception:
        return fallback_guidance(metrics)
