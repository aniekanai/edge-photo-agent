import requests
import base64
import cv2
import os

# NVIDIA Nemotron API Configuration
NEMOTRON_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NEMOTRON_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
API_KEY_ENV = "NVIDIA_API_KEY"
API_TIMEOUT_SECONDS = 5.0  # Increased for stability


def encode_frame_to_base64(frame):
    """
    Encode OpenCV frame to base64 for API transmission.
    """
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def nemotron_refinement(frame, metrics):
    """
    Professional photography refinement layer.

    This function does NOT handle basic brightness/centering/sharpness.
    Local CV handles those instantly.

    Nemotron focuses on higher-level semantic composition issues:
    - Pose orientation
    - Occlusion
    - Camera height
    - Background distractions
    - Subject gaze direction
    """

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        return ""  # Safe fallback if key missing

    image_b64 = encode_frame_to_base64(frame)

    system_prompt = (
        "You are an AI photography co-pilot assisting a professional photographer. "
        "Return only ONE short instruction (max 5 words). "
        "No explanations. If no issue detected, return 'OK'."
    )

    user_prompt = """
You are assisting a professional photographer.

Analyze the image and detect higher-level composition issues that CV metrics may miss:

- Subject turned too far sideways
- Face partially occluded (hands, objects, mask)
- Distracting object in foreground
- Camera angle too low or too high
- Poor head positioning
- Background distraction
- Subject looking away from camera
- Framing imbalance

Return ONE short instruction (max 5 words) directed to the photographer.

Examples:
Ask subject to turn slightly
Remove foreground object
Raise camera angle
Lower camera slightly
Reduce background distraction
Ask subject to face forward
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
        "max_tokens": 30,
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

        # Only return meaningful refinement
        if text.upper() == "OK":
            return ""
        return text

    except Exception:
        # Silent fallback for smooth UX
        return ""