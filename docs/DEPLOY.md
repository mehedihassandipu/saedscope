# Putting SAEDscope online

## Hugging Face Spaces (free, public link)

1. Create a new Space at huggingface.co → **New Space** → SDK **Docker** → blank template.
2. Clone it, copy the whole project into it, and replace the Space's `README.md` with `deploy/huggingface/README.md`
   (its header tells Hugging Face to build the `Dockerfile` and serve port 7860).
3. Leave out the benchmark images (`bench/data/*/images`) and anything you are not allowed to publish.
4. `git add . && git commit -m "SAEDscope" && git push`. The first build takes about 5 minutes.

```bash
git clone https://huggingface.co/spaces/<your-name>/saedscope space
rsync -a --exclude .venv --exclude '__pycache__' --exclude 'bench/data/*/images' ./ space/
cp deploy/huggingface/README.md space/README.md
cd space && git add . && git commit -m "SAEDscope" && git push
```

API keys: never commit them. If you want the AI explanation on by default, add `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`)
as a **secret** in the Space settings; otherwise visitors can paste their own key in the app.

## Streamlit Community Cloud

Push the project to a GitHub repository, then at share.streamlit.io choose the repository and `app.py`.
`requirements.txt` is picked up automatically.

## Local container

```bash
docker build -t saedscope .
docker run -p 7860:7860 saedscope    # open http://localhost:7860
```

## Privacy

A public Space shows whatever is uploaded to the person who uploaded it only (nothing is stored), but the built-in examples
and anything you commit are public. Do not commit unpublished lab data without your advisor's permission.
