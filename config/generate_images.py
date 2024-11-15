import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import replicate
import requests
from loguru import logger

from config.image_uploader import UTFSUploader
from config.persona_utils import persona_manager
from config.prompts import (
    BACKGROUND_INSTRUCTIONS,
    BACKGROUND_LOCATIONS,
    DETAIL_LEVEL_INSTRUCTIONS,
    IMAGE_NEGATIVE_PROMPT,
    IMAGE_PROMPT_TEMPLATE,
    IMAGE_STYLE_ELEMENTS,
    PERSONA_ANIMALS,
    PERSONA_IMAGE_INSTRUCTIONS,
)


def create_prompt_for_persona(persona: Dict) -> str:
    """Create an appropriate prompt for Stable Diffusion based on persona details."""
    # Get a random animal from the list
    animal = random.choice(PERSONA_ANIMALS)

    # Create base prompt with animal personification
    prompt = IMAGE_PROMPT_TEMPLATE.format(
        animal=animal, name=persona["name"], personality=persona["prompt"]
    )

    prompt += ",\n ".join(IMAGE_STYLE_ELEMENTS) + "\n\n"
    prompt += ",\n ".join(PERSONA_IMAGE_INSTRUCTIONS) + "\n\n"
    prompt += ",\n ".join(BACKGROUND_INSTRUCTIONS) + "\n\n"
    prompt += ",\n ".join(random.choices(BACKGROUND_LOCATIONS, k=1)) + "\n\n"
    prompt += ",\n ".join(DETAIL_LEVEL_INSTRUCTIONS)
    prompt += ",\n ".join(PERSONA_IMAGE_INSTRUCTIONS) + "\n\n"

    logger.debug(f"Generated prompt: {prompt}")
    return prompt


def generate_image_worker(
    prompt: str, api_key: str, output_path: Path, persona_name: str
):
    """Worker function for generating a single image"""
    try:
        logger.info(f"[{persona_name}] Starting image generation")

        # Remove 'sk_live_' prefix if present
        clean_api_key = api_key.replace("sk_live_", "")

        # Set the API token for this process
        os.environ["REPLICATE_API_TOKEN"] = clean_api_key

        logger.debug(f"[{persona_name}] Using API key: {clean_api_key[:8]}...")

        # Run SDXL with the given prompt using the latest version
        output = replicate.run(
            "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
            input={
                "prompt": prompt,
                "width": 1920,
                "height": 1080,
                "refine": "expert_ensemble_refiner",
                "apply_watermark": False,
                "num_inference_steps": 25,
                "negative_prompt": IMAGE_NEGATIVE_PROMPT,
            },
        )

        # Save the generated image
        if output and len(output) > 0:
            # Get the image URL
            image_url = output[0]

            # Download the image
            response = requests.get(image_url)
            response.raise_for_status()

            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.success(f"[{persona_name}] Image saved to {output_path}")
            return True
        else:
            raise Exception("No output received from model")

    except Exception as e:
        logger.error(f"[{persona_name}] Error generating image: {str(e)}")
        logger.exception(f"[{persona_name}] Full error details:")
        return False


def get_available_models(api_key: str) -> list:
    """Fetch list of available models from ModelsLab API"""
    try:
        url = "https://modelslab.com/api/v4/dreambooth/model_list"
        payload = {"key": api_key}

        response = requests.post(url, json=payload)
        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text}")

        models = response.json()
        return models
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        return []


def main():
    if len(sys.argv) != 7:
        logger.error("Missing arguments")
        print(
            "Usage: python generate_images.py --replicate-key <replicate_key> --utfs-key <utfs_key> --app-id <app_id>"
        )
        sys.exit(1)

    if (
        sys.argv[1] != "--replicate-key"
        or sys.argv[3] != "--utfs-key"
        or sys.argv[5] != "--app-id"
    ):
        logger.error(
            "Arguments must be --replicate-key <replicate_key> --utfs-key <utfs_key> --app-id <app_id>"
        )
        print(
            "Usage: python generate_images.py --replicate-key <replicate_key> --utfs-key <utfs_key> --app-id <app_id>"
        )
        sys.exit(1)

    replicate_api_key = sys.argv[2]  # This is the r8_* key for replicate
    utfs_api_key = sys.argv[4]  # The UTFS key
    app_id = sys.argv[6]

    # Remove 'sk_live_' prefix if present from replicate key
    replicate_api_key = replicate_api_key.replace("sk_live_", "")

    logger.info("Starting image generation process")

    # First, let's get available models
    logger.info("Fetching available models...")
    models = get_available_models(replicate_api_key)
    if models:
        logger.info("Available models:")
        for model in models:
            # Handle both string and dictionary formats
            if isinstance(model, dict):
                logger.info(
                    f"- {model.get('model_name', 'Unknown')} (ID: {model.get('model_id', 'Unknown')})"
                )
            else:
                logger.info(f"- {model}")
    else:
        logger.warning("No models found or error fetching models")

    # Initialize PersonaManager
    persona_manager.load_personas()

    # Create images directory (updated path)
    images_dir = Path(__file__).parent / "local_images"
    images_dir.mkdir(exist_ok=True)

    # Prepare tasks for personas that need images
    tasks = []
    for key, persona in persona_manager.personas.items():
        if not persona.get("image"):
            prompt = create_prompt_for_persona(persona)
            image_path = images_dir / f"{key}.png"
            tasks.append((prompt, replicate_api_key, image_path, persona["name"]))

    # Process tasks with limited concurrency
    max_concurrent = 3
    with mp.Pool(processes=max_concurrent) as pool:
        results = []
        for task in tasks:
            time.sleep(2)  # Small delay between starting processes
            result = pool.apply_async(generate_image_worker, task)
            results.append((task[3], result))

        # Wait for all processes to complete
        for persona_name, result in results:
            try:
                success = result.get()
                if success:
                    # Update persona image path in the JSON (updated path)
                    key = next(
                        k
                        for k, v in persona_manager.personas.items()
                        if v["name"] == persona_name
                    )
                    persona_manager.personas[key]["image"] = f"local_images/{key}.png"
                    logger.info(f"✓ Successfully generated image for {persona_name}")
            except Exception as e:
                logger.error(f"✗ Failed to generate image for {persona_name}: {str(e)}")

        # Save updated JSON
        persona_manager.save_personas()

    # After successful image generation, upload to UTFS
    uploader = UTFSUploader(api_key=utfs_api_key, app_id=app_id)

    # Verify UTFS credentials before proceeding
    if not uploader.verify_credentials():
        logger.error("Invalid UTFS credentials")
        return 1

    for persona_name, result in results:
        try:
            success = result.get()
            if success:
                key = next(
                    k
                    for k, v in persona_manager.personas.items()
                    if v["name"] == persona_name
                )
                # Use absolute path instead of relative
                local_image_path = images_dir / f"{key}.png"

                # Update local image path
                persona_manager.update_persona_image(key, str(local_image_path))

                # Upload to UTFS using the absolute path
                file_url = uploader.upload_file(local_image_path)
                if file_url:
                    persona_manager.update_persona_image(key, file_url)
                    logger.success(
                        f"✓ Successfully generated and uploaded image for {persona_name}"
                    )
        except Exception as e:
            logger.error(f"✗ Failed to process image for {persona_name}: {str(e)}")

    logger.success("Image generation and upload complete!")


if __name__ == "__main__":
    main()
