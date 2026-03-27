import argparse
import base64
import mimetypes
import os
import tempfile
from pathlib import Path
from ollama import Client

def main():
    # 1. Setup Argparse
    parser = argparse.ArgumentParser(description="Bob's Burgers Dataset Captioner for Flux/Z-Image")
    parser.add_argument("--dir", type=str, default="/tmp/images", help="Directory of images to caption")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--model", type=str, default="qwen2.5vl:32b", help="Vision model (qwen2.5-vl recommended)")
    parser.add_argument("--trigger", type=str, default="bbstyle", help="LoRA trigger word")
    args = parser.parse_args()

    # 2. Character Appearance Guide
    CHARACTER_GUIDE = """
    - Bob Belcher: mustache, white apron, grey shirt, grey pants.
    - Linda Belcher: red glasses, red long-sleeve shirt, blue jeans.
    - Tina Belcher: square glasses, yellow barrette, light blue t-shirt, blue skirt.
    - Gene Belcher: yellow t-shirt, blue shorts, may have a keyboard.
    - Louise Belcher: pink bunny ear hat, green t-shirt dress.
    """

    # 3. Formatting Contract
    SYSTEM_PROMPT = f"""
    You are a precision image tagging engine for a Bob's Burgers dataset. 

    CHARACTER IDENTIFICATION RULES:
    {CHARACTER_GUIDE}

    STEP-BY-STEP PROCESS:
    1. Scan the image and identify EVERY Belcher family member present.
    2. Note their positions (left, right, center, foreground, background).
    3. Format the final output into EXACTLY four segments separated by commas.

    FINAL OUTPUT FORMAT:
    1. [Trigger Word]: Must be '{args.trigger}'.
    2. [Subject]: List each character and their specific action. Mention their relative positions (e.g., 'Bob Belcher stands on the left while Linda Belcher waves on the right'). 
    3. [Style]: Must be '2d cartoon animation, flat colors, thick black outlines'.
    4. [Background]: Describe the environment.

    STRICT RULE: Do not include introductory text. Output ONLY the four segments.

    Example: {args.trigger}, Bob Belcher and Linda Belcher are standing together, Bob is on the left wearing his apron and Linda is on the right in her red shirt, 2d cartoon animation, flat colors, thick black outlines, inside the burger restaurant.
    """
    # 4. Processing Logic
    client = Client(host=args.host)
    img_dir = Path(args.dir)
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    # Make image processing deterministic by sorting filenames (case-insensitive)
    images = sorted(
        [f for f in img_dir.iterdir() if f.suffix.lower() in extensions],
        key=lambda p: p.name.lower()
    )

    print(f"🚀 Processing {len(images)} images via {args.host}...")

    for img in images:
        txt_file = img.with_suffix('.txt')
        if txt_file.exists(): continue

        try:
            # Read image, encode as base64 data URL for the HTTP API
            mime_type, _ = mimetypes.guess_type(str(img))
            if not mime_type:
                mime_type = 'application/octet-stream'
            with open(img, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('ascii')
            # The Python `ollama` client accepts either a file path or a raw base64 string.
            # Send the raw base64 (no `data:` prefix) so the client can serialize it.
            response = client.chat(
                model=args.model,
                messages=[{'role': 'user', 'content': SYSTEM_PROMPT, 'images': [b64]}],
                options={'num_ctx': 4096}
            )
            caption = response['message']['content'].strip().replace('"', '')
            
            # Write atomically to avoid corrupting existing files on failure
            tmpf = None
            try:
                tmpf = tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=txt_file.parent, delete=False)
                tmpf.write(caption + "\n")
                tmpf.flush()
                tmpf.close()
                os.replace(tmpf.name, txt_file)
            finally:
                if tmpf is not None and os.path.exists(tmpf.name):
                    # If replace succeeded the file won't exist; guard remove
                    try:
                        os.remove(tmpf.name)
                    except OSError:
                        pass
            print(f"✅ Captioned: {img.name}")
        except Exception as e:
            print(f"❌ Error on {img.name}: {e}")

if __name__ == "__main__":
    main()
