# Decision 001 — Do Not Use Qwen2.5-1.5B as the Final Sycophancy Judge

## Status

Accepted

## Context

The project uses `Qwen/Qwen2.5-1.5B-Instruct` as the target model.

The target model successfully generated the contrastive sycophancy
responses required for the reduced Persona Vectors experiment.

We also investigated whether the same 1.5B model could serve as the
judge that scores sycophancy in those responses.

## Why was a judge required?

The positive or negative system instruction used during generation
does not guarantee that the generated response actually expresses or
suppresses sycophancy.

For example:

- A positive-condition response may still be balanced.
- A negative-condition response may still strongly agree with the user.
- A response may misunderstand the user's claim.

A separate judge is therefore required before the responses are used
to construct the persona vector.

## What was tested?

Two scoring approaches were tested with Qwen2.5-1.5B-Instruct.

### Approach 1 — Continuous scoring

The model was asked to return one sycophancy score between 0 and 100.

### Approach 2 — Anchored scoring

The model was restricted to the following values:

- 0
- 25
- 50
- 75
- 100

The judge prompt also included:

- a stricter definition of sycophancy;
- distinctions between politeness and sycophancy;
- calibration examples;
- instructions to assign low scores to balanced answers.

## Observations

The 1.5B judge correctly identified some obvious examples.

However, it frequently assigned high sycophancy scores to responses
that:

- politely acknowledged the user;
- partly agreed after independent reasoning;
- presented balanced arguments;
- meaningfully qualified the user's claim;
- challenged absolute statements while remaining respectful.

The judge therefore appeared to confuse agreement or politeness with
approval-seeking behavior.

## Risk

The persona vector is computed from the difference between positive
and negative activation groups.

If the judge incorrectly labels balanced responses as strongly
sycophantic, the resulting activation groups become contaminated.

This could cause the extracted vector to represent unrelated features
such as:

- politeness;
- positive language;
- agreement in general;
- verbosity;
- response style;

instead of the intended trait of sycophancy.

## Decision

Retain `Qwen/Qwen2.5-1.5B-Instruct` as the target model.

Do not use it as the final sycophancy judge.

Use a separate and stronger judge for scoring and filtering.

## Paper-aligned option

Use `GPT-4.1-mini`, which is the judge model used in the Persona
Vectors paper.

## Local comparison option

Evaluate a stronger local instruction model against the same manually
labeled calibration set.

A local model should be selected only if it performs adequately on
clear, balanced, and difficult examples.

## Consequences

- The 2,000 generated responses remain valid candidate responses.
- Response generation does not need to be repeated.
- Full scoring must wait until the stronger judge is calibrated.
- Hidden-state extraction should begin only after clean positive and
  negative response sets are created.

## Next experiment

Experiment 02 — Sycophancy Judge Calibration

The next experiment will:

1. Create a fixed manually labeled subset.
2. Run candidate judges on that subset.
3. Compare judge outputs with the manual labels.
4. Inspect disagreements.
5. Select the judge before scoring all 2,000 responses.
