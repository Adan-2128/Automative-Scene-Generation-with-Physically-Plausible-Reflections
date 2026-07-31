# Automative Scene Generation with Physically Plausible Reflections (PS1)

An automated, high-fidelity Python pipeline engineered to composite a single 3D car model into diverse, AI-generated background environments with mathematically precise grounding, dynamic sunlit exposure matching, and realistic contact shadows.

---

## 🚀 Key Features

* **Geometric Consistency:** Keeps the vehicle's model properties, textures, paint finish, and decals completely invariant across all environment variations.
* **Pixel-Level Ground Anchoring:** Automatically scans alpha masks to detect exact tire-to-road touchpoints, completely eliminating the "floating car" rendering artifact.
* **Dynamic Lighting & Exposure Matching:** Adjusts color grading, brightness thresholds, and contrast dynamically to match outdoor ambient sunlight and environmental color tones.
* **Multi-Environment Batch Generation:** Automatically parses JSON scene metadata and bulk-generates distinct composite variations (e.g., forest highways, industrial docks, mountain passes, and urban cityscapes).
* **High-Clarity Rendering Engine:** Utilizes high-quality **LANCZOS resampling**, sharpness enhancement filters, and **Unsharp Masking** coupled with maximum JPEG export quality (`100`) to guarantee production-ready visual fidelity.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Image Processing:** Pillow (`PIL`)
* **Background Removal:** `rembg` (Deep-learning-based matting)
* **API Integration:** Pollinations AI (High-resolution generative image endpoints)

---

## 📂 Project Structure

```text
PS C:\Users\DELL\cars> 
├── models/                     # ML components (not car 3D assets)
│   ├── feature_extractor.py     # ResNet CNN feature extraction
│   ├── pca_reducer.py           # PCA fit/transform/save/load
│   ├── lighting_engine.py       # Groq API call + JSON schema validation
│   └── artifacts/
│       └── pca_state.joblib     # ← fitted PCA model is SAVED here automatically
│
├── environments/               # everything about the scene/environment
│   ├── loader.py                # discovers images in environments/images/
│   ├── car_constraints.py       # car invariance lock (geometry/paint/wheels)
│   └── images/                  # ← PUT YOUR INPUT ENVIRONMENT PHOTOS/RENDERS HERE
│                                  (.jpg / .jpeg / .png / .webp / .bmp)
│
└── outputs/                    # everything the pipeline PRODUCES
    ├── scenes/                  # ← final scene descriptor JSON lands here
    │                                (one <image_name>_scene.json per input image)
    ├── feature_cache/           # reserved for cached raw ResNet feature dumps          
    └── logs/
        └── pipeline.log         # ← run logs are written here automatically
