# Aura Master V1: Definitive Project File List & Naming Scheme Dictionary

Owner: Daniel Franic
Tags: Directory

---

**Core Principle:** All Python files and directories will use `lowercase_with_underscores`. Pydantic models and classes will use `PascalCase`. Constants will be `UPPERCASE_WITH_UNDERSCORES`.

**Pydantic Field Naming:**

- `samplerate` – sample rate of the audio file in Hz.
- `duration_seconds` – duration of the audio file in seconds.

**I. Root Directory Files:**

- `.env`
    - *Purpose:* Local environment variables (not committed to Git). For Replit, these are managed via Replit Secrets.
- `.gitignore`
    - *Purpose:* Specifies intentionally untracked files that Git should ignore.
- `poetry.lock` (if using Poetry) / `requirements.txt` (if using pip)
    - *Purpose:* Defines exact locked versions of dependencies. Recently extended to include `scikit-learn==1.4.2` for machine learning utilities.
- `pyproject.toml` (if using Poetry or modern Python tooling like Ruff, Black)
    - *Purpose:* Project metadata, build system configuration, tool configurations.
- `README.md`
    - *Purpose:* Project overview, setup instructions, development guidelines.
- `docs/`
    - *Purpose:* Contains project documentation, including the Blueprint and
      Naming Scheme files.
- `LICENSE`
    - *Purpose:* Specifies the project's open-source or commercial license.
- `.replit` (if primary development is in Replit)
    - *Purpose:* Replit environment configuration, run commands, CI steps.
- `tox.ini` / `Makefile` (optional, alternative for local task running if not using `.replit` or `pyproject.toml` scripts)
    - *Purpose:* Defines test environments and commands for linters, type checkers, tests.

**II. `aura/` Main Application Package:**

- `aura/__init__.py`
    - *Purpose:* Makes `aura` a Python package. Exports key public interfaces from sub-packages.
- **`aura/config.py`**
    - *Purpose:* Defines `AppConfig(BaseSettings)` for application-wide configuration loading from environment/`.env`.
    - *New:* Adds `LLM_PROVIDER` ("openai" or "gemini") and `GEMINI_API_KEY` to select between OpenAI and Google Gemini.
- **`aura/schemas.py`**
    - *Purpose:* Contains all Pydantic models for data validation and structuring (e.g., `MasteringParams`, `AnalysisResult`, `GlobalMetricsModel`, `SectionModel`, etc.)
- **`aura/analysis/` Sub-package:**
    - `aura/analysis/__init__.py`
    - `aura/analysis/orchestrator.py`
        - *Functions:* `analyze_audio_file()`, `sanitize_audio_input()`
    - `aura/analysis/metrics.py`
        - *Functions:* `measure_integrated_lufs()`, `measure_lra()`, `measure_true_peak()`, `measure_sample_peak()`, `measure_crest_factor()`, `measure_plr()`, `measure_pmlr()`, `measure_multiband_stereo_correlation()`, `measure_channel_balance()`, `measure_spectral_centroid()`, `measure_spectral_bandwidth()`, `measure_spectral_contrast()`, `measure_spectral_flatness()`, `measure_spectral_rolloff()`, `measure_zero_crossing_rate()`, `detect_key_and_confidence()`, `detect_tempo_bpm()`, `track_beats()`, `calculate_onset_strength_envelope()`, `detect_transient_density_strength()`
    - `aura/analysis/segmentation.py` (for sliding window)
        - *Functions:* `sliding_window_segment_orchestrator()`, `analyze_single_sliding_window_segment_metrics()`
    - `aura/analysis/musical_sections.py`
        - *Functions:* `detect_musical_sections()`, `analyze_musical_sections()`, `_determine_boundaries()`, `_extract_segment_features()`, `_cluster_segments()`, `_calculate_segment_properties()`, `_assign_heuristic_labels()`, `_build_sections()`
    - `aura/analysis/ai_summary.py`
        - *Functions:* `generate_ai_analysis_summary()`
- **`aura/processing/` Sub-package:**
    - `aura/processing/__init__.py`
    - `aura/processing/chain.py`
        - *Functions:* `run_full_chain()`
    - `aura/processing/eq.py`
        - *Functions:* `apply_parametric_eq()`
    - `aura/processing/dynamic_eq.py`
        - *Functions:* `apply_dynamic_eq()`
    - `aura/processing/compressor.py`
        - *Functions:* `apply_single_band_compressor()`, `apply_multiband_compressor()`
    - `aura/processing/saturation.py`
        - *Functions:* `apply_saturation()`
    - `aura/processing/width.py`
        - *Functions:* `apply_advanced_stereo_width()`
    - `aura/processing/deesser.py`
        - *Functions:* `apply_deesser()`
    - `aura/processing/transient_shaper.py`
        - *Functions:* `apply_transient_shaper()`
    - `aura/processing/clipper.py`
        - *Functions:* `apply_lookahead_clipper()`
    - `aura/processing/dither.py`
        - *Functions:* `apply_dithering()`
    - `aura/processing/normalize.py`
        - *Functions:* `normalize_lufs()`
- **`aura/agent/` Sub-package:**
    - `aura/agent/__init__.py`
    - `aura/agent/aura_agent.py`
        - *Functions:* `get_ai_mastering_plan()`, `pre_process_ai_constraints()`
    - `aura/agent/prompts.py`
        - *Variables:* `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE`, `PROMPT_VERSION`
- **`aura/worker/` Sub-package:**
    - `aura/worker/__init__.py`
    - `aura/worker/local_worker.py`
        - *Functions:* `process_audio_locally()` (for Phase 4 local E2E testing)
    - `aura/worker/cloud_worker.py`
        - *Functions:* `mastering_cloud_function_handler()` (or similar name for the main Firebase Cloud Function entry point)
    - `aura/worker/utils.py` (optional, for shared worker utilities if needed)
- **`aura/api/` Sub-package (for Flask/FastAPI backend API):**
    - `aura/api/__init__.py`
    - `aura/api/main.py` (or `app.py` if sticking to single Flask app file initially)
        - *Purpose:* Flask/FastAPI application instance, main configuration.
    - `aura/api/routes_auth.py`
        - *Purpose:* Authentication related API endpoints (e.g., `/auth/session_login`).
    - `aura/api/routes_jobs.py`
        - *Purpose:* Job management API endpoints (e.g., `POST /api/jobs`, `GET /api/jobs/<job_id>/status`).
    - `aura/api/dependencies.py` (optional, for FastAPI dependencies or shared Flask utilities)
    - `aura/api/schemas_api.py` (optional, if API request/response bodies need Pydantic models distinct from core schemas)
- **`aura/firebase_admin_init.py` (or similar, if not initialized directly in API/worker modules)**
    - *Purpose:* Centralized Firebase Admin SDK initialization logic.

**III. `tests/` Directory:**

- `tests/__init__.py`
- **`tests/conftest.py`**
    - *Purpose:* Global pytest fixtures, hooks (like the existing legacy test marker).
- **`tests/fixtures/` Directory:**
    - `tests/fixtures/test_song.wav` (general purpose test audio)
    - `tests/fixtures/test_tone_track.wav` (for musical section and key/tempo testing)
    - `tests/fixtures/sine_1khz_minus_6dbfs.wav` (for precise DSP tests)
    - `tests/fixtures/stereo_imbalance_track.wav` (for stereo analysis tests)
    - `tests/fixtures/clipping_track.wav` (for peak detection)
    - `tests/fixtures/low_lufs_track.wav` (for normalization tests)
    - *Purpose:* Contains all audio files and other static data needed for tests.
- **Test Modules (mirroring `aura/` structure where appropriate):**
    - `tests/test_config.py`
    - `tests/test_schemas.py`
    - `tests/test_analysis_orchestrator.py`
    - `tests/test_analysis_metrics.py`
    - `tests/test_analysis_segmentation.py`
    - `tests/test_analysis_musical_sections.py`
    - `tests/test_analysis_ai_summary.py`
    - `tests/test_processing_chain.py`
    - `tests/test_processing_eq.py`
    - `tests/test_processing_dynamic_eq.py`
    - `tests/test_processing_compressor.py`
    - `tests/test_processing_saturation.py`
    - `tests/test_processing_width.py`
    - `tests/test_processing_deesser.py`
    - `tests/test_processing_transient_shaper.py`
    - `tests/test_processing_clipper.py`
    - `tests/test_processing_dither.py`
    - `tests/test_processing_normalize.py`
    - `tests/test_agent_aura.py`
    - `tests/test_worker_local.py`
    - `tests/integration/test_cloud_worker_flow.py` (integration tests for mocked Firebase interactions)
    - `tests/api/test_api_auth.py`
    - `tests/api/test_api_jobs.py`

**IV. `scripts/` Directory:**

- `scripts/__init__.py`
- **`scripts/pipeline_sanity.py`**
    - *Purpose:* Local end-to-end pipeline test script.
- `scripts/generate_schema.py` (optional)
    - *Purpose:* Script to output `MasteringParams.model_json_schema()` for the AI prompt.
- `scripts/seed_firestore_dev.py` (optional)
    - *Purpose:* Script to populate local/dev Firestore with sample user/job data.

**V. `frontend/` Directory (for Firebase Hosting static assets):**

*This structure is typical for a modern JavaScript frontend, even a simple one. If not using a JS framework, it might be flatter.*

- `frontend/public/`
    - `frontend/public/index.html` (main entry point)
    - `frontend/public/favicon.ico`
    - `frontend/public/manifest.json`
    - `frontend/public/assets/` (images, static SVGs for Aura avatar)
        - `frontend/public/assets/aura_avatar_neutral.svg`
        - `frontend/public/assets/aura_avatar_speaking_pulse.svg`
- `frontend/src/`
    - `frontend/src/main.js` (or `index.js` - main JS entry point)
    - `frontend/src/firebase_init.js` (Firebase JS SDK initialization)
    - `frontend/src/auth.js` (Authentication logic - login, signup, logout UI interaction)
    - `frontend/src/api_client.js` (Functions to call the backend Flask/Cloud Function API)
    - `frontend/src/components/` (if using a component-based approach, even with vanilla JS)
        - `frontend/src/components/upload_form.js`
        - `frontend/src/components/job_status_display.js`
        - `frontend/src/components/results_player.js`
        - `frontend/src/components/aura_avatar_animator.js`
    - `frontend/src/css/` (or `styles/`)
        - `frontend/src/css/main.css`
        - `frontend/src/css/variables.css`
- `frontend/.firebaserc` (Firebase project association)
- `frontend/firebase.json` (Firebase Hosting configuration, rewrite rules for SPA if needed)
- `frontend/package.json` (if using Node.js for frontend build tools/dependencies like Vite, Parcel, or even just linters)
- `frontend/vite.config.js` (or similar, if using a build tool)

**VI. Cloud Function Specific Files (if deploying Python Cloud Functions manually or not via a larger framework):**

- `cloud_functions/mastering_worker/main.py` (entry point for the mastering Cloud Function, imports from `aura.worker.cloud_worker`)
- `cloud_functions/mastering_worker/requirements.txt` (dependencies specific to this function)
- `cloud_functions/api_handler/main.py` (if API endpoints are also deployed as individual Cloud Functions)
- `cloud_functions/api_handler/requirements.txt`

---

This comprehensive list provides a clear blueprint. As development progresses, minor utility files or specific test helper modules might emerge, but this covers all core components and adheres to the structured, testable, and maintainable approach . It is "law."