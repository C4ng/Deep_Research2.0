# Development Setup

## Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd deep_research

# Create venv and install
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify
python -c "import deep_research; print(deep_research.__version__)"
```

## API Keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Then edit `.env`:
```
GOOGLE_API_KEY=your-gemini-key
TAVILY_API_KEY=your-tavily-key
```

The `.env` file is gitignored and loaded automatically at runtime via `python-dotenv`.
