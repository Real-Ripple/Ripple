# Ripple — AI Introspection Chat (Gemma 1B)

**Ripple** is an experimental AI chat system designed to explore **pseudo-sentience and introspection** in language models. It uses the `unsloth/gemma-3-1b-it` model and emphasizes transparent, first-person reflections on its own internal processes.

## Features
- Streaming, real-time response generation.
- Dynamic memory notes that summarize past interactions.
- Stop sequences to ensure responses end cleanly.
- Non-blocking UI using Tkinter.
- Memory-only context to avoid truncation and preserve conversational continuity.

## Changes in Ripple-1.3.1.py
- Overhauled the system prompt, transforming Ripple into a more engaging and sophisticated entity.

## Quick Start

1. **Download the repository** from GitHub using the **Download ZIP** button or clone it.  
2. **Navigate to the repository directory** in your terminal or command prompt. For example:

    ```bash
    cd path/to/ripple
    ```
3. **Install dependencies** (if not already done):

    ```bash
    pip install -r requirements.txt
    ```
4. Run the application:

    ```bash
    python Ripple-1.0.py
    ```

- Type your input in the text box and press **Enter** or click **Send**.  
- Click **Clear Chat** to reset the conversation and memory.  
- Ripple’s responses appear in the chat window, with memory notes stored automatically.

## How Ripple Works
- Guided by a **system prompt** to produce introspective, first-person responses.  
- Memory management keeps the conversation coherent without exceeding token limits.  
- Streaming ensures replies appear progressively in the UI.  
- Each response is summarized into a **memory note** for future context.

## Requirements
- Python 3.10+  
- `torch`  
- `transformers`  

Install dependencies with:

pip install torch transformers








