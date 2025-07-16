# Poster Finder for ICML 2025
This script extracts poster session papers from a ICML 2025 poster session HTML web page, filters them based on the given topic using OpenAI's LLM, and saves the results to text files.

## Prerequisites

1. **Install requirements**  
    ```bash
    pip install -r requirements.txt
    ```

2. **Set OpenAI API key**  
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```
## Usage
Go to the poster session page in `icml.cc` and save the HTML page as `posters.html`.  (e.g. https://icml.cc/virtual/2025/session/50260)
Then run the script with the following command:
```bash
# Basic usage:
python extract_papers.py --input posters.html --topics "LLM agents" "fine-tuning" --output-prefix "icml2025"

# Alternative usage:
python extract_papers.py -i posters.html -t "LLM agents" "fine-tuning" -o "icml2025"

# Run with defaults (input: posters.html, topics: LLM Agent-related):
python extract_papers.py
```
