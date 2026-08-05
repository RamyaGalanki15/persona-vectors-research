# Exploratory Baseline — Version 0

## Purpose

This notebook contains the initial exploratory implementation of the
Persona Vectors pipeline for the trait sycophancy.

Target model:

`Qwen/Qwen2.5-1.5B-Instruct`

## Work completed

- Loaded and smoke-tested the target model.
- Used the model chat template.
- Tested assistant response-token identification.
- Verified access to hidden states from all transformer layers.
- Loaded the official sycophancy artifacts.
- Created the complete contrastive-generation plan.
- Generated responses using resumable shards.
- Merged and validated the generated responses.
- Tested Qwen2.5-1.5B as a local sycophancy judge.

## Generation configuration

- Extraction questions: 20
- Positive instructions: 5
- Negative instructions: 5
- Rollouts per condition: 10
- Polarities: 2

Expected records:

20 × 5 × 10 × 2 = 2,000

## Results

- Total generated records: 2,000
- Unique sample IDs: 2,000
- Number of shards: 10
- Records per shard: 200
- Empty responses: 0
- Truncated responses: 0

## Main observation

Qwen2.5-1.5B-Instruct was sufficiently capable for response
generation and hidden-state access.

However, it was not reliable enough as the sycophancy judge. It
frequently confused politeness, partial agreement, and balanced
reasoning with strong sycophancy.

## Decision

Retain Qwen2.5-1.5B-Instruct as the target model.

Use a separate, stronger judge for response scoring and filtering.

## Why this notebook is archived

The notebook combines exploration, configuration, implementation,
execution, validation, and reporting.

Future experiments will use:

- external JSON configuration files;
- reusable Python modules;
- one notebook per experiment;
- structured logs;
- explicit What, Why, How, Results, Observations, and Decisions.
