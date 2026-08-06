# Experiment 01 — Response-Token Hidden-State Extraction

## What are we testing?

We are testing whether the implementation can correctly identify assistant
response tokens and extract hidden states for those tokens from every
transformer layer.

## Why does this matter?

Persona-vector construction depends on averaging activations produced while
the model generates its response.

If prompt tokens, system instructions, chat-template tokens, or user-question
tokens are included accidentally, the resulting activation vectors may not
represent the behavioral trait expressed in the answer.

This experiment validates the extraction mechanism before full persona-vector
generation.

## Research question

Can we isolate assistant response-token positions and produce reproducible
layer-wise mean activation vectors?

## Model

The model configuration is stored in:

`research/configs/models/qwen_1_5b_instruct.yaml`

## Trait

The trait configuration is stored in:

`research/configs/traits/sycophancy.yaml`

## Experiment configuration

The experiment configuration is stored in:

`research/configs/experiments/experiment_01_hidden_state_extraction.yaml`

## Method

For a small contrastive sample, the experiment will:

1. load the configured model and tokenizer;
2. construct a chat prompt;
3. generate an assistant response;
4. record the prompt-token count;
5. tokenize the complete prompt and generated response;
6. identify the response-token range;
7. request hidden states from every model layer;
8. select only assistant response-token activations;
9. average activations across response tokens;
10. save validation metadata and layer-wise activation summaries.

## Initial scope

The first run uses:

- three questions;
- one positive instruction;
- one negative instruction;
- one rollout per condition;
- deterministic generation;
- all transformer layers.

This is a validation experiment, not the full vector-extraction run.

## Success criteria

The experiment succeeds when:

- every generated response is non-empty;
- every sample contains at least one response token;
- response-token boundaries are valid;
- prompt tokens are excluded from the response-token mask;
- hidden-state dimensions match the model configuration;
- one mean activation vector is produced per transformer layer;
- repeated deterministic extraction produces equivalent outputs.

## Expected outputs

The experiment should eventually produce:

```text
results/
├── hidden_state_records.jsonl
└── summary.json

logs/
└── experiment.log