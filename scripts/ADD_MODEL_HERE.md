# Adding the trained EduLex checkpoint

This API serves the fine-tuned BERT token classifier from the EduLex project
when a Hugging Face checkpoint is present.

1. Copy the checkpoint files (config.json, model weights, tokenizer files)
   into this `model/` directory, or point `MODEL_DIR` at wherever they live.
2. Install the model dependencies: `pip install torch transformers`
3. Restart the server. `GET /model-info` should report
   `bert-token-classifier`.

Until then the API runs a clearly-labelled heuristic fallback so every
endpoint, test and the Docker image work without the weights.
