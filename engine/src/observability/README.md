# src/observability/

Token accounting and USD cost metering for LLM calls.

## Components

| File | Purpose |
|------|---------|
| `metering.py` | Token counting, cost calculation, and pricing registry |

## Data Classes

### `TokenUsage`

Raw token counts for a single LLM call or an aggregation.

- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `is_estimated` — true when exact counts aren't available (falls back to chars/4 heuristic)
- Supports `+` operator for accumulation

### `CostBreakdown`

USD cost breakdown for a single call or aggregation.

- `input_cost`, `output_cost`, `total_cost`
- `currency` (always `"USD"`)
- Supports `+` operator for accumulation

### `Metering`

Combined container holding both `TokenUsage` and `CostBreakdown`, plus `provider` and `model` metadata. Also supports `+` for session-level accumulation.

## Cost Log & Source Tracking

Every metered LLM call is appended to `data/orion/cost_log.jsonl` via `log_cost_event()`. Events include a `source` field — `"platform"` for platform-hosted keys or `"user"` for BYOK keys. Falls back to the `ORION_COST_SOURCE` environment variable when not explicitly provided.

| Function | Purpose |
|----------|---------|
| `log_cost_event(metering, agent, chat_id, source)` | Appends a cost event with source tagging to the JSONL log |
| `read_cost_log(since, until, agent, source, limit)` | Reads events with date-range (`since`/`until`), agent, and source filtering |
| `aggregate_costs(events)` | Summarizes events by model, agent, and source — returns `by_source` breakdown |

## Key Functions

| Function | Purpose |
|----------|---------|
| `meter_response(response, provider, messages)` | Creates a `Metering` from an `LLMResponse`. Uses exact token counts when available, otherwise estimates via chars/4. |
| `get_price(provider, model)` | Looks up per-million-token pricing from `config/pricing.yaml`. Supports exact match → prefix match (e.g. `gpt-5.2` matches `gpt-5.2-2025-12-11`) → `_default` per provider. |
| `compute_cost(usage, provider, model)` | Builds a `CostBreakdown` from token counts and pricing. |
| `zero_metering()` | Returns a zeroed `Metering` instance for initializing session accumulators. |
| `log_cost_event()` | Persists a cost event to the JSONL log with agent, chat_id, and source fields |
| `read_cost_log()` | Reads events with optional `since`, `until`, `agent`, `source`, and `limit` filters |
| `aggregate_costs()` | Aggregates events into summary stats with `by_model`, `by_agent`, and `by_source` breakdowns |

## Pricing Configuration

Pricing is defined in `config/pricing.yaml`:

```yaml
openai:
  gpt-5.2:
    input: 2.50    # USD per 1M input tokens
    output: 10.00  # USD per 1M output tokens
  _default:
    input: 3.00
    output: 15.00

ollama:
  _default:
    input: 0.0
    output: 0.0
```

## Usage

```python
from src.observability.metering import meter_response, zero_metering

m = meter_response(response, provider="openai", messages=messages)
session = zero_metering()
session = session + m  # accumulate across calls
```
