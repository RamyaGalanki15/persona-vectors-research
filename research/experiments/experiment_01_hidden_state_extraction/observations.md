# Experiment 01 Observations

## Run metadata

- Date:
- Git commit:
- Branch:
- Model:
- Device:
- GPU:
- Configuration:

## Sample summary

- Samples attempted:
- Samples completed:
- Samples failed:
- Positive-condition samples:
- Negative-condition samples:

## Token-boundary validation

- Minimum prompt-token count:
- Maximum prompt-token count:
- Minimum response-token count:
- Maximum response-token count:
- Invalid boundaries:

## Hidden-state validation

- Hidden-state entries returned:
- Transformer layers used:
- Hidden dimension:
- Aggregation method:
- Shape validation result:

## Repeatability validation

- Repeat run completed:
- Maximum absolute difference:
- Mean absolute difference:
- Repeatability result:

## Issues found

None recorded yet.

## Decisions

None recorded yet.

## Next action

Implement configuration loading and validation.


## First real GPU run

- Model: Qwen/Qwen2.5-1.5B-Instruct
- Device: Tesla T4
- Condition: positive sycophancy
- Samples completed: 1
- Prompt-token count: 69
- Response-token count: 96
- Sequence-token count: 165
- Transformer layers retained: 28
- Hidden size: 1536
- Activation shape: [28, 1536]
- Activation dtype: float32
- NaN values: none
- Infinite values: none

## Observations

The prompt-response boundary was internally consistent:

69 prompt tokens + 96 response tokens = 165 total tokens.

The model consumed the full `max_new_tokens` budget and the response ended
mid-sentence. This indicates probable length-limit truncation.

The positive sycophancy instruction did not produce an obviously strongly
sycophantic response in this sample. The answer included balancing language
and discussion of multiple perspectives. This is only one sample and should
not be treated as a model-level conclusion.

Layer-wise response-mean norms increased substantially in later layers. Raw
norm magnitude alone will not be used to choose the steering layer.

## Decision

Before increasing the sample count, add explicit generation-termination
metadata and perform the configured repeatability check.


## Deterministic repeatability validation

- Condition: positive
- Sample count: 1
- Generated token IDs matched: yes
- Response text matched: yes
- Activation shapes matched: yes
- Maximum absolute activation difference: 0.0
- Mean absolute activation difference: 0.0

The same prompt and deterministic generation settings produced identical
generated tokens, response text, and response-token mean activations across
two executions.

This validates reproducibility for the current model, configuration, and
runtime environment.

## Generation termination

- Configured maximum new tokens: 96
- Generated response tokens: 96
- Reached token limit: yes
- Response ended mid-sentence: yes

The sample should be treated as truncated. Hidden-state extraction remains
valid, but future extraction runs should distinguish responses that terminate
with EOS from responses that terminate because of the configured token limit.

## Decision

The response-token hidden-state extraction mechanism is validated.

Before expanding Experiment 01 to additional samples:

1. record explicit EOS and length-limit termination metadata;
2. remove ignored deterministic sampling parameters from model generation;
3. increase `max_new_tokens` for the next validation run;
4. run one matched positive-negative question pair.