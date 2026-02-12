import requests
import base64
import cv2
import os

NEMOTRON_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NEMOTRON_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
API_KEY_ENV = "NVIDIA_API_KEY"
API_TIMEOUT_SECONDS = 2.0


def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def fallback_guidance(metrics):
    if not metrics["face_detected"]:
        return "Center yourself"

    if not metrics["face_centered"]:
        return "Move to center"

    if metrics["brightness"] < 80:
        return "Increase lighting"

    if metrics["sharpness"] < 150:
        return "Hold still"

    return "Adjust slightly"


def nemotron_guidance(frame, metrics):
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        return fallback_guidance(metrics)

    image_b64 = encode_frame_to_base64(frame)

    system_prompt = (
        "You are a smart camera assistant. "
        "Return only ONE short instruction (max 4 words). "
        "No explanations."
    )

    user_prompt = f"""
Observations:
Brightness: {metrics['brightness']:.1f}
Sharpness: {metrics['sharpness']:.1f}
Face detected: {metrics['face_detected']}
Face centered: {metrics['face_centered']}
Score: {metrics['quality_score']}/100

Return ONE short instruction like:
Move left
Move right
Hold still
Remove object
Look at camera
Smile
Perfect - capturing
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
        "max_tokens": 20,
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
        return result["choices"][0]["message"]["content"].strip()

    except Exception:
        return fallback_guidance(metrics)
        
