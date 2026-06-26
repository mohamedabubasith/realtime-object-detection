---
title: Object Detection
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Object Detection — Hugging Face Spaces deploy

This folder contains everything needed to run the app on a **Hugging Face Docker
Space**. The container builds the React frontend, runs the FastAPI backend, and
serves **both on one port (7860)** — which is all HF exposes.

> Hardware: the free **CPU basic** tier (2 vCPU / 16 GB RAM) is plenty. No GPU.

## Deploy in 4 steps

1. **Create the Space**
   - https://huggingface.co/new-space → choose **SDK: Docker** → **Blank** template.
   - This also creates a `README.md` in the Space with the YAML header above
     (you can copy the header from this file; `app_port: 7860` is what matters).

2. **Add the Dockerfile**
   - In the Space, create a file named **`Dockerfile`** and paste in the contents
     of [`Dockerfile`](./Dockerfile) from this folder.

3. **Point it at your GitHub repo** (already set to this repo)
   - The Dockerfile is pre-filled with:
     ```dockerfile
     ARG REPO_URL=https://github.com/mohamedabubasith/realtime-object-detection.git
     ARG REPO_REF=main
     ```
   - The repo must be **public** (or add a token — see below).

4. **Commit**
   - The Space builds automatically (first build ~5–10 min: it pulls PyTorch CPU,
     OpenCV, ffmpeg, downloads the YOLO weights, and exports ONNX). When it's
     done, the app is live at your Space URL.

## Updating

The Dockerfile clones a fixed snapshot at build time, so after you push changes
to GitHub, trigger a rebuild: in the Space, **Settings → Factory rebuild** (or
push any commit to the Space). Pin a tag/commit via `REPO_REF` for reproducible
builds.

## Private GitHub repo

If your repo is private, add your token as a Space **Secret** named `GH_TOKEN`,
then change the clone line to use it:

```dockerfile
RUN --mount=type=secret,id=GH_TOKEN \
    git clone --depth 1 --branch "${REPO_REF}" \
    "https://oauth2:$(cat /run/secrets/GH_TOKEN)@github.com/mohamedabubasith/realtime-object-detection.git" /src
```

## Notes & tuning

- **One active stream.** `MAX_SESSIONS=1` is set for the 2-core box — it handles
  one live source comfortably. Raise it in the Dockerfile `ENV` if needed.
- **Storage is ephemeral.** Uploaded videos and the exported model live inside
  the container and reset on rebuild/restart. That's fine for this app.
- **Speed knobs** (Dockerfile `ENV`): `IMGSZ=416` (drop to `320` for more FPS),
  `MODEL_PATH=models/yolo26n.onnx`. See the project README for the full list.
- **First request** may be slightly slow as the model warms up; after that it's
  steady.

## Alternative: push the whole repo to the Space

Instead of cloning from GitHub, you can push this project directly into the Space
git repo and use a `Dockerfile` that `COPY`s the local context instead of
`git clone`. The clone approach above is recommended so GitHub stays the single
source of truth.
