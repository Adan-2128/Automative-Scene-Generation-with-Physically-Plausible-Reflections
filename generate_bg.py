import os
import json
import glob
import requests
import random

def generate_ai_backgrounds():
    print("Reading JSON files and generating diverse environmental backgrounds with integrated reflections...")
    
    scenes_folder = "outputs/scenes/"
    json_files = glob.glob(os.path.join(scenes_folder, "*.json"))
    
    if not json_files:
        print(f"Error: No JSON files found in {scenes_folder}!")
        return

    os.makedirs("generated_backgrounds", exist_ok=True)

    # Pre-defined list of diverse environments to ensure variety across generations
    environment_presets = [
        {"env_type": "lush green forest highway", "location": "surrounded by tall pine trees and natural sunlight"},
        {"env_type": "futuristic industrial dock", "location": "metal structures, container yards, and neon accent lights"},
        {"env_type": "mountain pass winding road", "location": "dramatic cliffs, rocky terrain, and open scenic sky"},
        {"env_type": "coastal beachside highway", "location": "ocean view, sandy shores, and bright sunny horizon"},
        {"env_type": "modern downtown city street", "location": "skyscrapers, glass buildings, and urban crosswalks"}
    ]

    for json_path in json_files:
        base_name = os.path.basename(json_path).replace(".json", "")
        
        # Read base scene details and lighting profile from JSON
        with open(json_path, "r") as file:
            scene_data = json.load(file)
            
        env_data = scene_data.get("environment_data", {})
        time_of_day = env_data.get("time_of_day", "daytime")
        
        # Extract reflection properties from the JSON file to habituate the environment
        reflection_intensity = env_data.get("reflection_intensity", 0.5)
        reflection_roughness = env_data.get("reflection_roughness", 0.5)
        specular_sharpness = env_data.get("specular_highlight_sharpness", 0.5)
        
        # Translate numeric values into descriptive text tokens for the image prompt
        refl_desc = "high gloss wet reflective asphalt surfaces" if reflection_intensity > 0.6 else "matte dry asphalt road"
        rough_desc = "sharp clear reflections" if reflection_roughness < 0.4 else "diffuse soft ambient reflections"
        spec_desc = "intense specular light highlights" if specular_sharpness > 0.6 else "soft balanced highlights"
        
        # Generate multiple distinct environment variations for this scene
        for i, preset in enumerate(environment_presets, start=1):
            env_type = preset["env_type"]
            location = preset["location"]
            
            # Build a distinct prompt incorporating the physical reflection parameters
            prompt = (
                f"Ground-level eye-level medium shot of an empty asphalt road lane in a {env_type}, "
                f"{location} during {time_of_day}, featuring {refl_desc}, {rough_desc}, and {spec_desc}, "
                f"centered single lane perspective, no other vehicles, photorealistic, 8k"
            )
            print(f"\nGenerating background variation {i} ({env_type}):\n-> Prompt: '{prompt}'")

            # Request the image from the AI API with a randomized seed for variety
            random_seed = random.randint(1, 999999)
            url_friendly_prompt = prompt.replace(" ", "%20")
            api_url = f"https://image.pollinations.ai/prompt/{url_friendly_prompt}?width=1920&height=1080&nologo=true&seed={random_seed}"
            
            response = requests.get(api_url)
            
            if response.status_code == 200:
                output_bg_path = os.path.join("generated_backgrounds", f"bg_{base_name}_var{i}.jpg")
                with open(output_bg_path, "wb") as file:
                    file.write(response.content)
                print(f"SUCCESS! Saved variation {i} to '{output_bg_path}'.")
            else:
                print(f"Error: Could not generate background variation {i}. Status code: {response.status_code}")

if __name__ == "__main__":
    generate_ai_backgrounds()