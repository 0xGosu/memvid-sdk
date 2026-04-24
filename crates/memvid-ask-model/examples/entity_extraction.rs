//! Example: Extract entities from text using OpenAI
//!
//! Run with: OPENAI_API_KEY=sk-... cargo run -p memvid-ask-model --example entity_extraction

use memvid_ask_model::{ENTITY_EXTRACTION_PROMPT, extract_entities};

fn main() {
    let text = "John Smith met with Microsoft CEO Satya Nadella in Seattle last Tuesday. \
                They discussed the partnership with S&P Global and CRISIL.";

    println!("Input text: {}\n", text);
    println!(
        "Using default prompt:\n{}\n",
        &ENTITY_EXTRACTION_PROMPT[..200]
    );

    match extract_entities("openai:gpt-4o-mini", text, None, None) {
        Ok(response) => {
            println!(
                "✓ Extracted {} entities from {} chars using {}:\n",
                response.entities.len(),
                response.text_chars,
                response.model
            );
            for e in &response.entities {
                println!(
                    "  • {} ({}) - {:.0}% confidence",
                    e.name,
                    e.entity_type,
                    e.confidence * 100.0
                );
            }
        }
        Err(e) => {
            eprintln!("✗ Error: {}", e);
            std::process::exit(1);
        }
    }
}
