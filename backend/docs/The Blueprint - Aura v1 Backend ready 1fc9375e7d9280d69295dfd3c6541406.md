# The Blueprint - Aura v1  Backend ready

Owner: Daniel Franic
Tags: Infrastructure

**Definitive Build Order for "Aura Master V1" (Replit First, then Firebase) - UNIFIED MASTER PLAN**

---

**Phase 0: Local Foundation, Restructure & Core Analysis/Processing Schemas (Replit Environment)**

*Goal: Establish a clean, well-tested, and robust local codebase in Replit with a complete schema for all analysis and processing.*

1. **0.1: Audit, Configuration & Environment Setup**
    1. **0.1.1: `requirements.txt` Overhaul:**
        - Remove all duplicate entries (e.g., `librosa`).
       - Pin all key dependencies to their latest stable, compatible versions: `flask`, `gunicorn`, `numpy`, `soundfile`, `pyloudnorm`, `scipy`, `torch>=2.0.0` (ensure CPU-only if not using Replit GPU, or explicitly configure for GPU if planned), `openai`, `pydantic`, `pydantic-settings`, `librosa>=0.10.0`, `scikit-learn==1.4.2`, `pytest`, `flake8`, `mypy`, `psutil`, `pyyaml` (if still needed, verify usage).
        - Verify compatibility between all pinned versions.
    2. **0.1.2: `aura/config.py` Implementation:**
        - Implement class `AppConfig(BaseSettings)` using `pydantic_settings`.
        - Define fields: `MAX_UPLOAD_MB: int = 150`, `MAX_DURATION_MIN: int = 15`, `OPENAI_API_KEY: SecretStr` (from `pydantic.types`), `LOG_LEVEL: str = "INFO"`, `DEFAULT_SR: int = 44100`.
        - Additional fields allow LLM selection: `LLM_PROVIDER` ("openai" or "gemini") and optional `GEMINI_API_KEY`.
    3. **0.1.3: Replit Environment Configuration:**
        - Utilize Replit "Secrets" for `OPENAI_API_KEY` and any other sensitive credentials for local development.
        - Create a `.env` file (and add to `.gitignore`) for non-sensitive local overrides or less critical development keys.
        - Update `.gitignore` comprehensively (e.g., `.env`, `__pycache__/`, `.pyc`, `.DS_Store`, `build/`, `dist/`, `.egg-info/`, `local_firebase_creds.json` if applicable, virtual environment folders).
2. **0.2: Project Restructuring into Packages**
    1. **0.2.1: Implement Directory Structure:**
        
        ```mermaid
        aura/
          ├── __init__.py # Exports key elements from sub-packages
          ├── analysis/
          │   ├── __init__.py
          │   ├── orchestrator.py      # analyze_audio_file(), sanitize_audio_input()
          │   ├── metrics.py           # All global & segment-level raw metric functions
          │   ├── segmentation.py      # sliding_window_segment_orchestrator(), analyze_single_sliding_window_segment_metrics()
          │   ├── musical_sections.py  # detect_musical_sections()
          │   └── ai_summary.py        # generate_ai_analysis_summary()
          ├── processing/
          │   ├── __init__.py
          │   ├── chain.py             # run_full_chain()
          │   ├── eq.py
          │   ├── compressor.py        # Single-band and Multiband
          │   ├── dynamic_eq.py
          │   ├── saturation.py
          │   ├── width.py
          │   ├── deesser.py
          │   ├── transient_shaper.py
          │   ├── clipper.py
          │   ├── dither.py
          │   └── normalize.py
          ├── agent/
          │   ├── __init__.py
          │   └── aura_agent.py
          ├── worker/
          │   ├── __init__.py
          │   └── local_worker.py      # Local job orchestration logic
          ├── api/                     # For Flask API endpoints later
          │   ├── __init__.py
          │   └── routes.py            # Initial placeholder for Flask routes
          ├── config.py                # AppConfig
          └── schemas.py               # All Pydantic models
        tests/
          ├── __init__.py
          ├── fixtures/
          │   └── test_song.wav
          │   └── test_tone_track.wav  # For musical section testing
          ├── conftest.py
          ├── test_analysis_orchestrator.py
          ├── test_analysis_metrics.py
          ├── test_analysis_segmentation.py
          ├── test_analysis_musical_sections.py
          ├── test_analysis_ai_summary.py
          ├── test_processing_eq.py
          # ... one test_processing_X.py file for EACH processing module ...
          ├── test_processing_chain.py
          ├── test_agent_aura.py
          └── test_schemas.py
        scripts/
          └── pipeline_sanity.py
        # Root project files:
        .env
        .gitignore
        poetry.lock / requirements.txt
        pyproject.toml # Recommended
        README.md
        
        ```
        
    2. **0.2.2: `__init__.py` Files:** Populate all `__init__.py` files to correctly expose intended public interfaces of each package/sub-package and manage imports.
3. **0.3: Comprehensive Pydantic Schema Definition (`aura/schemas.py`)**
    1. **0.3.1: Analysis Schemas:**
        - `FileInfoModel(BaseModel)`: Typed fields for `sf.info()` output + `exists`, `error_message`.
        - `GlobalMetricsModel(BaseModel)`: Typed fields for *all* global metrics to be implemented in Phase 1.1.
        - `SegmentRawMetricsModel(BaseModel)`: Typed fields for *all* metrics calculated per segment (mirroring `GlobalMetricsModel` where applicable).
        - `SegmentMetricsModel(BaseModel)`: Contains `raw_metrics: SegmentRawMetricsModel` and `deviations: dict` (or a typed `DeviationsModel`).
        - `SlidingWindowAnalysisModel(BaseModel)`: (Was `SegmentAnalysisModel`) Contains `segments: List[SegmentMetricsModel]`, `segment_length_sec: float`, `overlap_ratio: float`.
        - `SectionModel(BaseModel)`: `name: str`, `start_time: float`, `end_time: float`.
        - `MusicalSectionAnalysisModel(BaseModel)`: (Was `SectionAnalysisModel`) `sections: List[SectionModel]`.
        - `AISummaryModel(BaseModel)`: Defines the structured output of `generate_ai_analysis_summary()`, including fields for global highlights, problematic sliding-window segments (with musical section cross-reference), and musical section characteristics.
        - `AnalysisResult(BaseModel)`: The master output of `analyze_audio_file()`.
            
            ```python
            class AnalysisResult(BaseModel):
                file_info: FileInfoModel
                global_metrics: GlobalMetricsModel
                sliding_window_analysis: SlidingWindowAnalysisModel
                musical_section_analysis: MusicalSectionAnalysisModel
                ai_summary: AISummaryModel
            
            ```
            
    2. **0.3.2: `MasteringParams(BaseModel)` Overhaul (Critical):**
        - Root model for the AI's entire plan.
        - Boolean flags for each processor: `apply_parametric_eq: bool = True`, `apply_dynamic_eq: bool = False`, etc.
        - Nested Pydantic models for detailed settings of *each* processor. Example:
            
            ```python
            class EQBandModel(BaseModel):
                freq: float = Field(..., gt=20, lt=20000)
                gain_db: float = Field(..., ge=-24, le=24)
                q: float = Field(..., gt=0.1, le=18)
                type: str = Field(..., pattern="^(lowpass|highpass|peak|lowshelf|highshelf|notch)$") # Example enum/pattern
                slope_db_oct: Optional[int] = Field(None, pattern="^(12|24|36|48)$") # For LPF/HPF
            
            class ParametricEQSettingsModel(BaseModel):
                bands: List[EQBandModel] = []
                linear_phase: bool = False
                global_gain_db: float = 0.0
            
            # ... similar detailed models for DynamicEQ, MultibandCompressor (with BandSettings), SingleBandCompressor, Saturation, AdvancedStereo, DeEsser, TransientShaper, Limiter, Dither, LUFSNormalization ...
            
            class MasteringParams(BaseModel):
                # Flags
                apply_parametric_eq: bool = True
                apply_dynamic_eq: bool = False
                # ... all other apply_X flags ...
            
                # Settings Models
                parametric_eq_settings: Optional[ParametricEQSettingsModel] = None
                dynamic_eq_settings: Optional[DynamicEQSettingsModel] = None
                # ... all other settings_X models ...
            
                # Global/Output Params
                target_lufs: float = -14.0
                final_true_peak_db: float = -1.0
                output_bit_depth: int = Field(32, pattern="^(32|24|16)$") # 32 means float
                # ... any other global parameters ...
                model_config = ConfigDict(extra="forbid", validate_assignment=True) # Strict validation
            
            ```
            
    3. **0.3.3: Testing (`tests/test_schemas.py`):** Thoroughly test all Pydantic models: default instantiation, validation of correct and incorrect data types/ranges, successful updates, JSON serialization/deserialization.

---

**Phase 1: Elite Audio Analysis Engine ("Aura's Hyper-Acute Senses")**

*Goal: Implement all analysis functions, populating the rich Pydantic schemas defined in Phase 0.*

1. **1.1: Implement `aura/analysis/orchestrator.py`**
    1. **1.1.1: `sanitize_audio_input(audio: np.ndarray, sr: int) -> np.ndarray`:**
        - Ensure float32, handle NaN/Inf.
        - Force stereo (dual-mono if single channel, take first two if multi-channel > 2 with a warning).
        - Implement mandatory DC offset removal (e.g., 2nd order Butterworth high-pass at ~5-10Hz using `scipy.signal.butter` and `scipy.signal.sosfilt`).
    2. **1.1.2: `analyze_audio_file(input_audio_path: str, app_config: AppConfig) -> AnalysisResult`:**
        - Load audio using `soundfile`. Validate against `app_config.MAX_UPLOAD_MB` & `MAX_DURATION_MIN` (raise `ValueError` if exceeded). Populate `FileInfoModel`.
        - Call `sanitize_audio_input()`.
        - Call all global metric functions from `aura.analysis.metrics` and populate `GlobalMetricsModel`.
        - Call `aura.analysis.segmentation.sliding_window_segment_orchestrator()` (passing `GlobalMetricsModel`) and populate `SlidingWindowAnalysisModel`.
        - Call `aura.analysis.musical_sections.detect_musical_sections()` and populate `MusicalSectionAnalysisModel`.
        - Call `aura.analysis.ai_summary.generate_ai_analysis_summary()` (passing all three prior models) and populate `AISummaryModel`.
        - Construct and return the full `AnalysisResult` instance.
    3. **1.1.3: Testing (`tests/test_analysis_orchestrator.py`):** Test file loading (valid/invalid paths, oversized files), sanitization, correct invocation of sub-modules (using mocks where necessary for focused testing), and the structure/completeness of the returned `AnalysisResult`.
2. **1.2: Implement `aura/analysis/metrics.py`**
    1. **1.2.1: Loudness & Peak Metric Functions:** `measure_integrated_lufs`, `measure_lra`, `measure_true_peak` (4x oversample via `torch.nn.functional.interpolate`), `measure_sample_peak`.
    2. **1.2.2: Dynamic Range Metric Functions:** `measure_crest_factor` (overall, short-term), `measure_plr`, `measure_pmlr`.
    3. **1.2.3: Stereo Image Metric Functions:** `measure_multiband_stereo_correlation` (e.g., low/mid/high bands), `measure_channel_balance` (RMS & Peak L/R delta).
    4. **1.2.4: Advanced Spectral Metric Functions:** `measure_spectral_centroid`, `measure_spectral_bandwidth`, `measure_spectral_contrast`, `measure_spectral_flatness`, `measure_spectral_rolloff`, `measure_zero_crossing_rate`.
    5. **1.2.5: Musical Feature Metric Functions:** `detect_key_and_confidence` (Librosa), `detect_tempo_bpm` (Librosa), `track_beats` (Librosa).
    6. **1.2.6: Transient Metric Functions:** `calculate_onset_strength_envelope` (Librosa), `detect_transient_density_strength` (Librosa).
    7. **1.2.7: Testing (`tests/test_analysis_metrics.py`):** For each metric function, use synthetic test signals (sine waves, noise, impulses, stereo files with known imbalances, tracks with known key/tempo) to verify accuracy and expected output ranges. Use `pytest.approx` for float comparisons.
3. **1.3: Implement `aura/analysis/segmentation.py` (Sliding Window)**
    1. **1.3.1: `analyze_single_sliding_window_segment_metrics(segment_audio: np.ndarray, sr: int) -> SegmentRawMetricsModel`:**
        - Takes a short audio chunk.
        - Calls *all applicable functions* from `aura.analysis.metrics.py` to calculate the raw metrics for this segment.
        - Returns a populated `SegmentRawMetricsModel`.
    2. **1.3.2: `sliding_window_segment_orchestrator(full_audio: np.ndarray, sr: int, global_metrics_results: GlobalMetricsModel) -> SlidingWindowAnalysisModel`:**
        - Divides `full_audio` into fixed-size, overlapping windows (e.g., 3-5s length, 50% overlap - configurable in `AppConfig`).
        - For each window:
            - Calls `analyze_single_sliding_window_segment_metrics()`.
            - Calculates deviations of segment's raw metrics from the `global_metrics_results`.
            - Populates a `SegmentMetricsModel` (containing raw segment metrics + deviations).
        - Returns a `SlidingWindowAnalysisModel` containing the list of `SegmentMetricsModel` instances.
    3. **1.3.3: Testing (`tests/test_analysis_segmentation.py`):** Test with known audio to verify correct number of segments, segment boundaries, accurate raw metric calculation per segment (by comparing with direct metric calls on extracted segments), and correct deviation calculations.
4. **1.4: Implement `aura/analysis/musical_sections.py`**
    1. **1.4.1: `detect_musical_sections(audio: np.ndarray, sample_rate: int) -> MusicalSectionAnalysisModel`:**
        - Implement robust logic using `librosa.onset.onset_detect`, `librosa.beat.beat_track`.
        - Employ advanced segmentation techniques (e.g., `librosa.segment.agglomerative` or `librosa.feature.stack_memory` on chroma/MFCCs followed by clustering, or simpler heuristics based on significant changes in RMS/spectral features between beat-grouped blocks).
        - Aim to identify common sections (Intro, Verse, Pre-Chorus, Chorus, Bridge, Solo, Outro) or at least structurally distinct blocks (A, B, C). The naming can be heuristic.
        - Populate `SectionModel` list with names, start/end times.
        - Return `MusicalSectionAnalysisModel`.
    2. **1.4.2: `analyze_musical_sections(...) -> List[SectionModel]`:**
        - High-level orchestration that leverages internal helpers for boundary detection, feature aggregation, clustering and heuristic labeling.
        - Returns a simple list of heuristically labeled `SectionModel` objects for downstream processing.
    3. **1.4.3: Testing (`tests/test_analysis_musical_sections.py`):** Use `test_tone_track.wav` (with clear Verse/Chorus structure) and other varied musical pieces. Assert reasonable section detection, naming, and timing. Mock Librosa calls if needed to test clustering logic in isolation.
5. **1.5: Implement `aura/analysis/ai_summary.py`**
    1. **1.5.1: `generate_ai_analysis_summary(analysis_result: AnalysisResult) -> AISummaryModel`:**
        - Takes the *full* `AnalysisResult` (containing global, sliding window, and musical section analyses).
        - Generates a rich, structured, yet concise `AISummaryModel` for the LLM.
        - **Key Content:**
            - Overall track assessment (e.g., "Dynamically active track in C# Minor, ~120 BPM, slightly dark spectral balance, good stereo width").
            - Highlights from global metrics (e.g., "Integrated LUFS very low at -20 LUFS", "True Peak at +0.5 dBFS - clipping!").
            - Identification of problematic *sliding window* segments, cross-referenced with *musical sections* they fall into (e.g., "Chorus 1 (0:45-1:15) contains segments (0:50-0:55) with excessive low-mid energy and poor stereo correlation").
            - Summary of musical structure (e.g., "Typical Verse-Chorus structure with a Bridge").
            - This summary should be versioned if its internal structure changes over time.
    2. **1.5.2: Testing (`tests/test_analysis_ai_summary.py`):** Provide diverse `AnalysisResult` fixtures and assert that the `AISummaryModel` correctly identifies and summarizes key characteristics, problems, and structural elements.

---

**Phase 2: State-of-the-Art Processing Chain ("Aura's Mastering Studio")**

*Goal: Implement all advanced DSP modules, fully controllable via the comprehensive `MasteringParams` schema.*

1. **2.1: Implement All Processing Modules in `aura/processing/` (one file per processor type where logical):**
    - Each function signature: `apply_X(audio: np.ndarray, sr: int, settings: XSettingsModel) -> np.ndarray`.
    1. **2.1.1: `eq.py` - `apply_parametric_eq`:** Full filter suite (LPF, HPF with variable slopes; Bell, Shelf, Notch). **Mandatory Linear Phase option.**
    2. **2.1.2: `dynamic_eq.py` - `apply_dynamic_eq`:** Per-band Freq, Q, Threshold, Ratio, Attack, Release, Static Gain, Filter Type.
    3. **2.1.3: `compressor.py`:**
        - `apply_single_band_compressor`: With Knee, optional sidechain HPF.
        - `apply_multiband_compressor`: Linkwitz-Riley crossovers, AI defines bands (3-6) & crossovers. Per-band Threshold, Ratio, Attack, Release, Makeup, Knee.
    4. **2.1.4: `saturation.py` - `apply_saturation`:** Multiple algorithms (Tape, Tube, Transformer). Controls: Drive, Type, Mix, Output.
    5. **2.1.5: `width.py` - `apply_advanced_stereo_width`:** M/S processing. Independent Mid/Side gain & EQ. Frequency-dependent width.
    6. **2.1.6: `deesser.py` - `apply_deesser`:** Narrow-band dynamic attenuation for sibilance.
    7. **2.1.7: `transient_shaper.py` - `apply_transient_shaper`:** Independent Attack/Sustain modification.
    8. **2.1.8: `clipper.py` - `apply_lookahead_clipper` (True Peak Limiter):** Robust lookahead, multiple waveshaping algorithms, character/softness, adaptive or AI-controlled release. Must accurately hit `final_true_peak_db`.
    9. **2.1.9: `dither.py` - `apply_dithering`:** For output to 24-bit or 16-bit (e.g., TPDF noise shaping). `MasteringParams` gets `output_bit_depth`. (Default output may still be 32-bit float for V1, making dither optional based on this param).
    10. **2.1.10: `normalize.py` - `normalize_lufs`:** Accurate LUFS targeting post-limiting if needed.
2. **2.2: Implement `aura/processing/chain.py`**
    1. **2.2.1: `run_full_chain(audio: np.ndarray, sr: int, params: MasteringParams) -> np.ndarray`:**
        - Takes initial audio and the AI-generated `MasteringParams`.
        - Calls each processing function from 2.1 sequentially, strictly in the defined mastering order (e.g., Corrective EQ -> De-Esser -> Dynamic EQ -> Compressor -> Additive EQ -> Saturation -> etc. -> Limiter -> Dither -> LUFS Norm).
        - Only calls a processor if its `apply_X` flag in `params` is true and its `X_settings` model is provided.
3. **2.3: Testing (`tests/test_processing_X.py` for each module, `tests/test_processing_chain.py`):**
    - Rigorous unit tests for each DSP module using test signals (sine, noise, impulse, sweeps) to verify parameter effects and output integrity.
    - `test_processing_chain.py`: Loads a fixture WAV, creates a comprehensive `MasteringParams` object, runs `run_full_chain`, and asserts expected changes in output audio characteristics (LUFS, peak, spectral balance, dynamic range reduction, etc.).

---

**Phase 3: The AI Mastering Engine ("Aura's Creative & Technical Mind")**

*Goal: Implement a sophisticated AI agent in `aura_agent.py` using the rich analysis to generate a bespoke `MasteringParams` plan and a clear explanation.*

1. **3.1: Implement `aura/agent/aura_agent.py`**
    1. **3.1.1: `get_ai_mastering_plan(ai_summary: AISummaryModel, user_intent: str, app_config: AppConfig) -> tuple[MasteringParams, str]`:**
        - **Prompt Engineering:**
            - System Prompt: "You are Aura, an expert AI mastering engineer. Your goal is to enhance audio for clarity, balance, appropriate loudness, and to match the user's artistic intent. You output a precise JSON mastering plan and a human-readable explanation."
            - User Prompt:
                - Provide the full `ai_summary` (from `AISummaryModel`).
                - Include `user_intent`.
                - Provide the complete JSON schema of `MasteringParams` (generated from the Pydantic model using `MasteringParams.model_json_schema()`). This is crucial for the LLM to produce valid output.
                - Instruct the LLM to:
                    1. First, generate the JSON object conforming *exactly* to the provided `MasteringParams` schema.
                    2. Second, after the JSON, provide a "human_explanation" field/section containing a step-by-step reasoning for its key decisions, referencing specific parts of the analysis (e.g., "Due to the harshness noted in the upper mids of Chorus 1 (1:15-1:30), I've applied a dynamic EQ cut...").
        - **LLM Interaction:** Use `openai` SDK, target latest model (GPT-4o or successor). Utilize JSON mode for the `MasteringParams` part if supported reliably by the model, otherwise parse carefully. Alternatively set `LLM_PROVIDER` to `gemini` and use `google-generativeai` with a `GEMINI_API_KEY`.
        - **Response Parsing & Validation:**
            - Separate the JSON plan from the explanation.
            - Parse the JSON plan string.
            - **Critical:** Validate the parsed dictionary against the `MasteringParams` Pydantic model.
            - **Retry Logic:** If validation fails or LLM output is malformed:
                - Log the error and the malformed LLM output.
                - Attempt 1-2 retries, potentially simplifying the prompt or explicitly telling the LLM about the validation errors from its previous attempt.
                - If retries fail, return a "safe default" `MasteringParams` (e.g., only Limiter at -1.0dBTP and LUFS Normalization to -14 LUFS) and an explanation like: "I encountered difficulty generating a detailed plan for this track. Here's a safe master. Please try rephrasing your intent or a different track."
        - Return the validated `MasteringParams` instance and the `ai_explanation` string.
    2. **3.1.2: Hybrid Intelligence Rules (Deterministic Pre-computation):**
        - Before calling the LLM, implement a rules-based function: `pre_process_ai_constraints(ai_summary: AISummaryModel) -> dict`.
        - This function could, for example:
            - If `ai_summary` indicates True Peak is already above -0.5 dBFS, suggest `limiter_settings.threshold_db` should be lower.
            - If `ai_summary` indicates very low LUFS, suggest `lufs_normalization_settings.max_gain_db` could be higher.
            - These deterministic suggestions/constraints are then fed into the LLM prompt to guide its output and improve consistency.
    3. **3.1.3: Prompt Versioning:** Store prompt templates in a manageable way (e.g., dedicated `.txt` files or a Python module) and include versioning if they evolve significantly.
2. **3.2: Testing (`tests/test_agent_aura.py`):**
    - Mock OpenAI or Gemini API calls depending on `LLM_PROVIDER`.
    - Provide diverse `AISummaryModel` inputs and user intents.
    - Assert that:
        - Correct prompts are constructed (including the `MasteringParams` JSON schema).
        - Mocked LLM responses (valid JSON, invalid JSON, malformed structures) are handled correctly.
        - Pydantic validation of the AI's plan is invoked and works as expected.
        - Retry and fallback logic for `MasteringParams` generation function correctly.
        - The explanation string is extracted.

---

**Phase 4: Local Worker Orchestration & End-to-End Sanity Check (Replit)**

*Goal: Implement `aura/worker/local_worker.py` to run the entire pipeline locally, driven by `scripts/pipeline_sanity.py`.*

1. **4.1: Implement `aura/worker/local_worker.py`**
    1. **4.1.1: `process_audio_locally(input_audio_path: str, user_intent_data: dict, app_config: AppConfig) -> dict`:**
        - This function orchestrates the local pipeline as per the Mermaid chart:
            1. Call `aura.analysis.orchestrator.analyze_audio_file()` -> `analysis_result: AnalysisResult`.
            2. Call `aura.agent.aura_agent.get_ai_mastering_plan(analysis_result.ai_summary, user_intent_data['text_intent'], app_config)` -> `mastering_plan: MasteringParams`, `ai_explanation: str`.
            3. Load audio data for processing using `soundfile` and `sanitize_audio_input`.
            4. Call `aura.processing.chain.run_full_chain(audio_data, sr, mastering_plan)` -> `processed_audio: np.ndarray`.
            5. Save `processed_audio` to a temporary output path.
            6. Run post-analysis: Call `analyze_audio_file()` on the temporary processed audio path -> `post_analysis_result: AnalysisResult`.
            7. Clean up temporary output path.
            8. Return a dictionary containing: `original_analysis_summary`, `ai_mastering_plan_params_dict`, `ai_explanation`, `post_analysis_summary`, and `processed_audio_data_array` (or path to it if too large to return directly).
2. **4.2: Implement/Finalize `scripts/pipeline_sanity.py`**
    1. **4.2.1:** Update the script (as drafted previously) to:
        - Initialize `AppConfig`.
        - Define a test audio path and user intent.
        - Call `aura.worker.local_worker.process_audio_locally()`.
        - Print key outputs: snippets of initial analysis, AI explanation, key parameters from the AI plan, and key global metrics from the post-analysis.
        - Assert basic success criteria (e.g., output LUFS is near target, peak is controlled).
    2. **4.2.2:** Ensure it runs cleanly from the project root via `python3 -m scripts.pipeline_sanity`.
3. **4.3: Final Local Testing & CI Validation**
    1. **4.3.1:** Run all tests: `pytest`, `flake8`, `mypy`. All must pass.
    2. **4.3.2:** Successfully execute `scripts/pipeline_sanity.py`. This signifies local readiness.

---

**Trigger for Firebase Migration:**

- **Condition:** `scripts/pipeline_sanity.py` runs successfully, all automated tests (`pytest`, `flake8`, `mypy`) pass locally in the Replit environment. The core engine is proven.

---

**Phase 5: Firebase Backend - Cloud Infrastructure & Worker Migration**

*Goal: Transition the proven local worker logic to a scalable Firebase backend, using Firestore for data and Cloud Storage for files.*

1. **5.1: Firebase Project Setup (Finalize):**
    1. **5.1.1:** Confirm Firebase Authentication, Firestore, Cloud Functions, and Firebase Hosting are enabled and configured in the Firebase console.
    2. **5.1.2:** Set up Firebase Admin SDK in the Python environment for Cloud Functions. Securely manage service account credentials (e.g., store service account JSON in Replit Secrets if deploying functions from Replit, or use environment variables in the Cloud Function environment).
    3. **5.1.3: Firestore Data Structures:**
        - `users/{uid}`: User profile information.
        - `jobs/{job_id}`: Document for each mastering job. Fields: `uid` (owner), `status` (queued, analyzing, planning, processing_eq, ..., complete, error), `timestamp_created`, `timestamp_updated`, `original_filename`, `input_gcs_path`, `user_intent_text`, `analysis_result_gcs_path` (path to JSON in GCS if too big for Firestore), `ai_mastering_plan` (the `MasteringParams` dict), `ai_explanation`, `processed_audio_gcs_path`, `post_analysis_result_gcs_path`.
    4. **5.1.4: Cloud Storage Buckets:**
        - `aura-uploads-{project_id}`: For user-uploaded raw audio.
        - `aura-results-{project_id}`: For storing large JSON analysis/plan files.
        - `aura-mastered-outputs-{project_id}`: For processed audio files.
        - Configure appropriate lifecycle rules and security rules for these buckets.
2. **5.2: Cloud Function Worker (`aura/worker/cloud_worker.py` - to be deployed):**
    1. **5.2.1:** Create an HTTP-triggered Firebase Cloud Function (Python runtime).
    2. **5.2.2: Function Logic (adapts `local_worker.py`):**
        - **Input:** Request body contains `uid`, `job_id`, `original_file_gcs_path`, `user_intent_text`. (The function is invoked by the API layer after job creation in Firestore).
        - Immediately update Firestore `jobs/{job_id}` status to "processing_started".
        - **File Handling:** Download audio from `original_file_gcs_path` to the Cloud Function's `/tmp` directory.
        - **Pipeline Execution:**
            1. `analysis_result = analyze_audio_file(...)`. Update Firestore job status. Store `AnalysisResult` (or GCS path to it) in Firestore.
            2. `mastering_plan, ai_explanation = get_ai_mastering_plan(...)`. Update Firestore. Store plan and explanation.
            3. Load audio again. `processed_audio = run_full_chain(...)`. Update Firestore through each major processing step.
            4. Save processed audio locally to `/tmp`. Upload to `aura-mastered-outputs-{project_id}` GCS bucket. Store GCS path in Firestore.
            5. `post_analysis_result = analyze_audio_file(...)` on the local processed file. Update Firestore. Store result (or GCS path).
            6. Clean up all files in `/tmp`.
            7. Final Firestore job status update to "complete".
        - **Error Handling:** Wrap entire process in try/except. On error, log to Cloud Logging, update Firestore job status to "error" with details.
        - **Resource Configuration:** Set adequate memory (e.g., 1GB or 2GB), CPU, and timeout (e.g., 540 seconds for 9 min, or use longer-running Cloud Run if needed for >10 min tracks) for the Cloud Function.

---

**Phase 6: API Layer & Frontend Implementation (Firebase)**

*Goal: Develop the Flask API (can be run on Cloud Run or as separate Cloud Functions) and the user-facing web application hosted on Firebase Hosting.*

1. **6.1: Flask API Layer (`aura/api/routes.py`, deployed to Cloud Run or as multiple HTTP Cloud Functions):**
    1. **6.1.1: Authentication:** Endpoint (e.g., `/auth/session_login`) that receives Firebase ID token from frontend, verifies it with Firebase Admin SDK, and creates a secure session cookie or returns a custom API token for subsequent authenticated requests.
    2. **6.1.2: Job Management Endpoints (require authentication):**
        - `POST /api/jobs`:
            - Receives `original_filename`, `original_file_gcs_path` (provided by frontend after direct GCS upload), `user_intent_text`.
            - Creates a new document in `jobs` collection in Firestore with status "queued", `uid`, timestamps, etc.
            - Asynchronously invokes the mastering Cloud Function worker (e.g., via HTTP request to its trigger URL, or by publishing to a Pub/Sub topic that triggers the worker).
            - Returns `job_id` to frontend.
        - `GET /api/jobs/<job_id>/status`: Retrieves and returns current job status from Firestore.
        - `GET /api/jobs/<job_id>/results`: Retrieves AI explanation, key analysis summaries, and signed GCS URLs for downloading/streaming the original (if kept) and mastered audio.
        - `GET /api/jobs`: Lists jobs for the authenticated user.
2. **6.2: Frontend Application (HTML, CSS, JS - Deployed to Firebase Hosting):**
    1. **6.2.1: User Authentication Flow (Firebase JS SDK):**
        - Login, Sign-up, Logout pages/components.
        - On successful Firebase JS SDK login, get ID token and send to Flask API for session creation.
    2. **6.2.2: Upload Page:**
        - File input, validates against `AppConfig.MAX_UPLOAD_MB` client-side.
        - Uses Firebase JS SDK to upload selected file directly to `aura-uploads-{project_id}` GCS bucket (client-side upload).
        - On successful GCS upload, receives the `gs://` path.
        - Collects user intent text, genre, target platform via form inputs.
        - Sends GCS path and form data to Flask API `POST /api/jobs`.
        - Redirects to Job Status page with the returned `job_id`.
    3. **6.2.3: Job Status & Results Page (Dynamic JS):**
        - Polls `/api/jobs/<job_id>/status` or uses Firestore real-time listener (if frontend directly interacts with Firestore for reads, respecting security rules).
        - Displays progress stages, messages, percentage.
        - Once status is "complete", fetches full results from `/api/jobs/<job_id>/results`.
        - Displays AI's natural language explanation.
        - Embeds HTML5 audio player with "Before/After" toggle (using signed GCS URLs).
        - Provides download link using signed GCS URL.
        - Implements the **Basic CUI Avatar Animation** (SVG animation triggered when explanation text is displayed).
    4. **6.2.4: User Dashboard Page:** Fetches and displays user's job history from `/api/jobs`.
    5. **6.2.5: UI/UX:** Clean, professional, intuitive, and trustworthy design. Responsive for desktop and mobile.

---

**Phase 7: Final Testing, CI/CD, Deployment, and Beta Launch**

*Goal: Ensure the entire cloud-integrated system is robust, secure, performs well, and is ready for initial user feedback.*

1. **7.1: Comprehensive Automated Testing:**
    1. **7.1.1: Unit & Integration Tests:** Ensure all local tests pass. Add integration tests for Flask API endpoints and their interactions with Cloud Functions and Firestore. Use Firebase Local Emulator Suite for testing Firebase services locally.
    2. **7.1.2: End-to-End (E2E) Tests:** Use Playwright or Selenium to automate full user journeys: sign-up, login, upload file to GCS, submit job, poll status, view results, play audio, download. Test against a staging Firebase project.
2. **7.2: Performance, Load & Cost Analysis:**
    1. **7.2.1:** Test concurrent job submissions.
    2. **7.2.2:** Monitor Cloud Function execution times, memory usage, and associated costs.
    3. **7.2.3:** Monitor Firestore read/write operations and GCS storage/bandwidth costs. Optimize if necessary.
3. **7.3: Security Hardening:**
    1. **7.3.1:** Rigorously review and test Firebase Security Rules for Firestore and Cloud Storage.
    2. **7.3.2:** Ensure all API endpoints have proper authentication and authorization.
    3. **7.3.3:** Perform vulnerability scanning on dependencies.
    4. **7.3.4:** Implement rate limiting on sensitive API endpoints if necessary.
4. **7.4: CI/CD Pipeline (e.g., GitHub Actions, or Replit's built-in deployment if suitable):**
    1. **7.4.1:** Automate running `pytest`, `flake8`, `mypy` on every push/PR.
    2. **7.4.2:** Automate deployment of frontend to Firebase Hosting.
    3. **7.4.3:** Automate deployment of backend (Flask API to Cloud Run, Cloud Functions to Firebase).
5. **7.5: Beta Program Launch:**
    1. **7.5.1:** Prepare clear onboarding documentation and feedback channels for beta testers.
    2. **7.5.2:** Onboard a select group of target users.
    3. **7.5.3:** Actively monitor system performance, logs, and costs during beta.
    4. **7.5.4:** Collect and categorize user feedback (bugs, usability, audio quality, explanation clarity, feature requests).
    5. **7.5.5:** Iterate rapidly based on beta feedback, deploying fixes and improvements.

This unified and highly detailed plan provides a clear, sequential path to "Aura Master V1". Each step builds upon the last, emphasizing quality, thorough testing, and robust foundations for future expansion (like the advanced visualizer, which will leverage the rich analysis data already being prepared for the frontend).