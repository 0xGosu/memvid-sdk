# memvid-ask-model

LLM inference module for Memvid Q&A with local and cloud model support.

[![Crates.io](https://img.shields.io/crates/v/memvid-ask-model.svg)](https://crates.io/crates/memvid-ask-model)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE-APACHE)

## About

memvid-ask-model provides LLM inference capabilities for Memvid's Q&A functionality. It supports both local inference using llama.cpp and cloud APIs (OpenAI, Claude, Gemini).

## Features

- **Local Inference** - Built-in llama.cpp with phi1.5 model (no API key needed)
- **OpenAI** - GPT-4 and GPT-3.5 support
- **Anthropic** - Claude models
- **Google** - Gemini models
- **RAG Integration** - Seamlessly works with memvid-core search results

## Installation

```toml
[dependencies]
memvid-ask-model = "2.0.102"
```

## Usage

```rust
use memvid_ask_model::run_model_inference;
use memvid_core::Memvid;

// Get search results from memvid-core
let mem = Memvid::open("knowledge.mv2")?;
let hits = mem.find("topic", 5)?;

// Run inference with local model
let answer = run_model_inference(
    "What is this about?",
    &hits,
    None, // Use local model
)?;

// Or use cloud API
let answer = run_model_inference(
    "Summarize the findings",
    &hits,
    Some("openai"), // Requires OPENAI_API_KEY
)?;
```

## Environment Variables

For cloud models, set the appropriate API key:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
```

## Documentation

- [Full Documentation](https://docs.memvid.com)
- [GitHub Repository](https://github.com/memvid/memvid)

## License

Licensed under Apache 2.0

- [Apache License 2.0](LICENSE-APACHE)
