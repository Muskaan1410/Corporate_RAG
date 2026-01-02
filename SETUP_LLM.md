# LLM Setup Guide (Ollama + LLaMA)

## Step 1: Install Ollama

1. Download and install Ollama from: https://ollama.ai
2. For Windows: Download the installer and run it
3. Verify installation: Open terminal and run `ollama --version`

## Step 2: Start Ollama Server

```bash
ollama serve
```

This starts the Ollama server on `http://localhost:11434`

## Step 3: Pull LLaMA Model

Choose one of these models:

```bash
# Recommended: LLaMA 3.2 (smaller, faster)
ollama pull llama3.2

# Or LLaMA 3.1 (better quality, larger)
ollama pull llama3.1

# Or LLaMA 2 (older but stable)
ollama pull llama2
```

## Step 4: Install Python Package

```bash
pip install ollama
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Step 5: Test the Setup

```bash
python test_rag.py
```

## Troubleshooting

**Issue: "Could not connect to Ollama"**
- Make sure `ollama serve` is running
- Check if server is on `http://localhost:11434`

**Issue: "Model not found"**
- Run `ollama pull llama3.2` (or your chosen model)
- Verify with `ollama list`

**Issue: Slow responses**
- Use smaller model (llama3.2 instead of llama3.1)
- Reduce `max_context_chunks` in the code
- Check system resources (RAM, CPU)

## Available Models

- `llama3.2` - 3B parameters, fast, good for testing
- `llama3.1` - 8B parameters, better quality
- `llama2` - 7B parameters, stable
- `mistral` - Alternative, good quality
- `phi3` - Microsoft's small model

Check all available: `ollama list`

