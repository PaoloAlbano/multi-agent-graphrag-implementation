# Results recap

_Generated 2026-10-24T17:50:28.687634+00:00_

_LLM calls columns count every successful agent call (query generation + evaluation + interpretation + recheck/refix), excluding failed calls; query_evaluator is derived from query_generator._

## Qwen/Qwen3.5-27B (temp0.0-reasoning-medium)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 61.0% | 59.0% | -2.0% | 56.0% | 54.0% | -2.0% | 2.4 | 7.0 |
| company | 74.0% | 80.0% | +6.0% | 69.0% | 73.0% | +4.0% | 2.2 | 5.1 |
| fictional_character | 79.0% | 81.0% | +2.0% | 78.0% | 79.0% | +1.0% | 2.5 | 6.5 |
| flight_accident | 89.0% | 92.0% | +3.0% | 88.0% | 90.0% | +2.0% | 2.1 | 4.2 |
| geography | 50.0% | 59.0% | +9.0% | 47.0% | 55.0% | +8.0% | 2.7 | 9.3 |
| **Average** | 70.6% | 74.2% | +3.6% | 67.6% | 70.2% | 2.6% | 2.4 | 6.4 |

## Qwen/Qwen3.5-27B (temp0.0-reasoning-off)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 42.0% | 45.0% | +3.0% | 40.0% | 44.0% | +4.0% | 3.1 | 10.4 |
| company | 55.0% | 61.0% | +6.0% | 55.0% | 59.0% | +4.0% | 3.2 | 8.6 |
| fictional_character | 53.0% | 59.0% | +6.0% | 50.0% | 58.0% | +8.0% | 3.1 | 9.4 |
| flight_accident | 85.0% | 85.0% | +0.0% | 83.0% | 85.0% | +2.0% | 2.4 | 5.4 |
| geography | 33.0% | 47.0% | +14.0% | 32.0% | 44.0% | +12.0% | 3.4 | 10.9 |
| **Average** | 53.6% | 59.4% | +5.8% | 52.0% | 58.0% | 6.0% | 3.0 | 8.9 |

## Qwen/Qwen3.5-27B (temp1.0-reasoning-medium)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 58.0% | 57.0% | -1.0% | 53.0% | 55.0% | +2.0% | 2.4 | 8.8 |
| company | 70.0% | 72.0% | +2.0% | 68.0% | 68.0% | +0.0% | 2.4 | 5.5 |
| fictional_character | 87.0% | 83.0% | -4.0% | 86.0% | 80.0% | -6.0% | 2.2 | 6.6 |
| flight_accident | 93.0% | 89.0% | -4.0% | 90.0% | 88.0% | -2.0% | 2.2 | 5.0 |
| geography | 49.0% | 59.0% | +10.0% | 45.0% | 56.0% | +11.0% | 2.7 | 9.7 |
| **Average** | 71.4% | 72.0% | +0.6% | 68.4% | 69.4% | 1.0% | 2.4 | 7.1 |

## Qwen/Qwen3.5-27B (temp1.0-reasoning-off)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 45.0% | 44.0% | -1.0% | 42.0% | 43.0% | +1.0% | 2.9 | 11.9 |
| company | 52.0% | 63.0% | +11.0% | 50.0% | 64.0% | +14.0% | 3.0 | 9.6 |
| fictional_character | 63.0% | 65.0% | +2.0% | 60.0% | 64.0% | +4.0% | 3.0 | 9.4 |
| flight_accident | 87.0% | 86.0% | -1.0% | 87.0% | 86.0% | -1.0% | 2.3 | 5.9 |
| geography | 42.0% | 45.0% | +3.0% | 42.0% | 41.0% | -1.0% | 3.1 | 12.2 |
| **Average** | 57.8% | 60.6% | +2.8% | 56.2% | 59.6% | 3.4% | 2.9 | 9.8 |

## deepseek-ai/DeepSeek-V4-Flash (temp0.0-reasoning-medium)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 58.0% | 64.0% | +6.0% | 56.0% | 67.0% | +11.0% | 2.5 | 6.5 |
| company | 77.0% | 73.0% | -4.0% | 75.0% | 73.0% | -2.0% | 2.1 | 4.5 |
| fictional_character | 83.0% | 82.0% | -1.0% | 83.0% | 83.0% | +0.0% | 2.1 | 4.7 |
| flight_accident | 90.0% | 87.0% | -3.0% | 92.0% | 87.0% | -5.0% | 2.1 | 3.8 |
| geography | 44.0% | 23.0% | -21.0% | 43.0% | 21.0% | -22.0% | 2.7 | 14.8 |
| **Average** | 70.4% | 65.8% | -4.6% | 69.8% | 66.2% | -3.6% | 2.3 | 6.8 |

## deepseek-ai/DeepSeek-V4-Flash (temp0.0-reasoning-off)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 57.0% | 59.0% | +2.0% | 58.0% | 60.0% | +2.0% | 2.6 | 7.7 |
| company | 67.0% | 70.0% | +3.0% | 66.0% | 65.0% | -1.0% | 2.5 | 6.3 |
| fictional_character | 57.0% | 70.0% | +13.0% | 58.0% | 71.0% | +13.0% | 2.9 | 7.9 |
| flight_accident | 87.0% | 85.0% | -2.0% | 86.0% | 84.0% | -2.0% | 2.3 | 4.8 |
| geography | 45.0% | 59.0% | +14.0% | 46.0% | 56.0% | +10.0% | 3.3 | 10.0 |
| **Average** | 62.6% | 68.6% | +6.0% | 62.8% | 67.2% | 4.4% | 2.7 | 7.4 |

## deepseek-ai/DeepSeek-V4-Flash (temp1.0-reasoning-medium)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 51.0% | 60.0% | +9.0% | 51.0% | 57.0% | +6.0% | 2.4 | 7.1 |
| company | 78.0% | 75.0% | -3.0% | 78.0% | 81.0% | +3.0% | 2.1 | 4.6 |
| fictional_character | 84.0% | 82.0% | -2.0% | 84.0% | 81.0% | -3.0% | 2.1 | 4.4 |
| flight_accident | 86.0% | 89.0% | +3.0% | 87.0% | 91.0% | +4.0% | 2.0 | 4.3 |
| geography | 48.0% | 37.0% | -11.0% | 47.0% | 36.0% | -11.0% | 2.6 | 12.1 |
| **Average** | 69.4% | 68.6% | -0.8% | 69.4% | 69.2% | -0.2% | 2.3 | 6.5 |

## deepseek-ai/DeepSeek-V4-Flash (temp1.0-reasoning-off)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 57.0% | 51.0% | -6.0% | 62.0% | 49.0% | -13.0% | 2.4 | 8.6 |
| company | 65.0% | 56.0% | -9.0% | 67.0% | 58.0% | -9.0% | 2.4 | 7.7 |
| fictional_character | 58.0% | 64.0% | +6.0% | 60.0% | 67.0% | +7.0% | 2.7 | 8.3 |
| flight_accident | 82.0% | 85.0% | +3.0% | 83.0% | 84.0% | +1.0% | 2.3 | 5.1 |
| geography | 43.0% | 49.0% | +6.0% | 43.0% | 48.0% | +5.0% | 3.0 | 10.9 |
| **Average** | 61.0% | 61.0% | +0.0% | 63.0% | 61.2% | -1.8% | 2.5 | 8.1 |

## openai/gpt-oss-120b (temp0.0-reasoning-medium)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 42.0% | 48.0% | +6.0% | 41.0% | 48.0% | +7.0% | 2.9 | 9.4 |
| company | 69.0% | 69.0% | +0.0% | 67.0% | 69.0% | +2.0% | 2.2 | 4.6 |
| fictional_character | 78.0% | 80.0% | +2.0% | 79.0% | 76.0% | -3.0% | 2.2 | 5.8 |
| flight_accident | 84.0% | 83.0% | -1.0% | 84.0% | 83.0% | -1.0% | 2.1 | 4.6 |
| geography | 41.0% | 51.0% | +10.0% | 41.0% | 49.0% | +8.0% | 2.8 | 9.4 |
| **Average** | 62.8% | 66.2% | +3.4% | 62.4% | 65.0% | 2.6% | 2.4 | 6.8 |

## openai/gpt-oss-120b (temp0.0-reasoning-off)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 49.0% | 39.0% | -10.0% | 48.0% | 40.0% | -8.0% | 3.0 | 9.9 |
| company | 67.0% | 70.0% | +3.0% | 68.0% | 70.0% | +2.0% | 2.2 | 5.4 |
| fictional_character | 82.0% | 78.0% | -4.0% | 82.0% | 80.0% | -2.0% | 2.2 | 5.3 |
| flight_accident | 84.0% | 76.0% | -8.0% | 85.0% | 77.0% | -8.0% | 2.1 | 4.9 |
| geography | 46.0% | 52.0% | +6.0% | 45.0% | 48.0% | +3.0% | 2.6 | 9.9 |
| **Average** | 65.6% | 63.0% | -2.6% | 65.6% | 63.0% | -2.6% | 2.4 | 7.1 |

## openai/gpt-oss-120b (temp1.0-reasoning-medium)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 50.0% | 54.0% | +4.0% | 49.0% | 52.0% | +3.0% | 3.0 | 9.1 |
| company | 68.0% | 72.0% | +4.0% | 68.0% | 69.0% | +1.0% | 2.2 | 5.3 |
| fictional_character | 80.0% | 79.0% | -1.0% | 79.0% | 80.0% | +1.0% | 2.2 | 5.3 |
| flight_accident | 84.0% | 81.0% | -3.0% | 83.0% | 81.0% | -2.0% | 2.2 | 4.4 |
| geography | 47.0% | 45.0% | -2.0% | 46.0% | 46.0% | +0.0% | 2.6 | 10.0 |
| **Average** | 65.8% | 66.2% | +0.4% | 65.0% | 65.6% | 0.6% | 2.4 | 6.8 |

## openai/gpt-oss-120b (temp1.0-reasoning-off)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 43.0% | 46.0% | +3.0% | 46.0% | 46.0% | +0.0% | 2.9 | 9.4 |
| company | 67.0% | 70.0% | +3.0% | 69.0% | 69.0% | +0.0% | 2.1 | 4.6 |
| fictional_character | 82.0% | 81.0% | -1.0% | 84.0% | 81.0% | -3.0% | 2.2 | 5.0 |
| flight_accident | 82.0% | 81.0% | -1.0% | 82.0% | 79.0% | -3.0% | 2.1 | 4.6 |
| geography | 44.0% | 42.0% | -2.0% | 45.0% | 41.0% | -4.0% | 2.6 | 10.3 |
| **Average** | 63.6% | 64.0% | +0.4% | 65.2% | 63.2% | -2.0% | 2.4 | 6.8 |

## openai/gpt-oss-20b (temp0.0-reasoning-medium)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 42.0% | 46.0% | +4.0% | 45.0% | 48.0% | +3.0% | 2.7 | 8.9 |
| biology | 35.0% | 30.0% | -5.0% | 36.0% | 30.0% | -6.0% | 2.6 | 10.6 |
| company | 60.0% | 55.0% | -5.0% | 60.0% | 56.0% | -4.0% | 2.5 | 8.0 |
| fictional_character | 67.0% | 70.0% | +3.0% | 67.0% | 66.0% | -1.0% | 2.5 | 6.3 |
| flight_accident | 78.0% | 71.0% | -7.0% | 78.0% | 70.0% | -8.0% | 2.4 | 7.6 |
| geography | 43.0% | 42.0% | -1.0% | 43.0% | 41.0% | -2.0% | 2.7 | 10.7 |
| movie | 61.0% | 62.0% | +1.0% | 62.0% | 60.0% | -2.0% | 2.4 | 7.8 |
| nba | 65.0% | 68.0% | +3.0% | 64.0% | 67.0% | +3.0% | 2.1 | 6.5 |
| politics | 26.0% | 24.0% | -2.0% | 26.0% | 25.0% | -1.0% | 2.8 | 14.6 |
| soccer | 34.0% | 37.0% | +3.0% | 35.0% | 38.0% | +3.0% | 2.3 | 11.8 |
| terrorist_attack | 73.0% | 73.0% | +0.0% | 73.0% | 74.0% | +1.0% | 2.2 | 5.6 |
| **Average** | 53.1% | 52.5% | -0.5% | 53.5% | 52.3% | -1.3% | 2.5 | 9.0 |

## openai/gpt-oss-20b (temp0.0-reasoning-off)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 48.0% | 51.0% | +3.0% | 48.0% | 53.0% | +5.0% | 2.7 | 9.2 |
| biology | 35.0% | 38.0% | +3.0% | 33.0% | 38.0% | +5.0% | 2.6 | 10.6 |
| company | 60.0% | 61.0% | +1.0% | 60.0% | 62.0% | +2.0% | 2.4 | 7.4 |
| fictional_character | 72.0% | 68.0% | -4.0% | 71.0% | 66.0% | -5.0% | 2.3 | 7.2 |
| flight_accident | 78.0% | 67.0% | -11.0% | 77.0% | 68.0% | -9.0% | 2.6 | 6.7 |
| geography | 39.0% | 41.0% | +2.0% | 38.0% | 38.0% | +0.0% | 2.5 | 10.7 |
| movie | 61.0% | 56.0% | -5.0% | 62.0% | 56.0% | -6.0% | 2.4 | 8.0 |
| nba | 68.0% | 66.0% | -2.0% | 66.0% | 65.0% | -1.0% | 2.1 | 6.6 |
| politics | 26.0% | 25.0% | -1.0% | 25.0% | 25.0% | +0.0% | 2.6 | 14.5 |
| soccer | 36.0% | 43.0% | +7.0% | 36.0% | 43.0% | +7.0% | 2.3 | 11.4 |
| terrorist_attack | 72.0% | 74.0% | +2.0% | 71.0% | 74.0% | +3.0% | 2.4 | 6.1 |
| **Average** | 54.1% | 53.6% | -0.5% | 53.4% | 53.5% | 0.1% | 2.4 | 8.9 |

## openai/gpt-oss-20b (temp1.0-reasoning-medium)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 47.0% | 50.0% | +3.0% | 45.0% | 52.0% | +7.0% | 2.7 | 8.7 |
| biology | 36.0% | 32.0% | -4.0% | 33.0% | 34.0% | +1.0% | 2.7 | 11.5 |
| company | 63.0% | 59.0% | -4.0% | 63.0% | 61.0% | -2.0% | 2.4 | 7.4 |
| fictional_character | 71.0% | 71.0% | +0.0% | 68.0% | 69.0% | +1.0% | 2.4 | 6.8 |
| flight_accident | 75.0% | 76.0% | +1.0% | 75.0% | 77.0% | +2.0% | 2.4 | 6.4 |
| geography | 37.0% | 49.0% | +12.0% | 36.0% | 46.0% | +10.0% | 2.6 | 10.5 |
| movie | 74.0% | 60.0% | -14.0% | 72.0% | 60.0% | -12.0% | 2.3 | 7.9 |
| nba | 61.0% | 70.0% | +9.0% | 61.0% | 68.0% | +7.0% | 2.2 | 5.8 |
| politics | 22.0% | 18.0% | -4.0% | 21.0% | 18.0% | -3.0% | 2.7 | 14.7 |
| soccer | 42.0% | 47.0% | +5.0% | 42.0% | 48.0% | +6.0% | 2.6 | 12.2 |
| terrorist_attack | 74.0% | 75.0% | +1.0% | 70.0% | 75.0% | +5.0% | 2.4 | 5.8 |
| **Average** | 54.7% | 55.2% | +0.5% | 53.3% | 55.3% | 2.0% | 2.5 | 8.9 |

## openai/gpt-oss-20b (temp1.0-reasoning-off)

| Domain | single | agentic | delta | ext single | ext agentic | ext delta | avg LLM calls (single) | avg LLM calls (agentic) |
|---|---|---|---|---|---|---|---|---|
| art | 45.0% | 53.0% | +8.0% | 45.0% | 51.0% | +6.0% | 2.6 | 8.4 |
| biology | 34.0% | 39.0% | +5.0% | 34.0% | 38.0% | +4.0% | 2.6 | 11.6 |
| company | 59.0% | 62.0% | +3.0% | 58.0% | 61.0% | +3.0% | 2.4 | 7.2 |
| fictional_character | 74.0% | 72.0% | -2.0% | 76.0% | 71.0% | -5.0% | 2.5 | 7.5 |
| flight_accident | 82.0% | 72.0% | -10.0% | 82.0% | 71.0% | -11.0% | 2.5 | 6.3 |
| geography | 33.0% | 43.0% | +10.0% | 33.0% | 43.0% | +10.0% | 2.7 | 10.1 |
| movie | 59.0% | 63.0% | +4.0% | 58.0% | 60.0% | +2.0% | 2.5 | 7.7 |
| nba | 66.0% | 68.0% | +2.0% | 64.0% | 66.0% | +2.0% | 2.2 | 6.5 |
| politics | 28.0% | 24.0% | -4.0% | 27.0% | 23.0% | -4.0% | 2.6 | 14.9 |
| soccer | 41.0% | 48.0% | +7.0% | 40.0% | 47.0% | +7.0% | 2.5 | 11.6 |
| terrorist_attack | 76.0% | 75.0% | -1.0% | 72.0% | 74.0% | +2.0% | 2.4 | 6.2 |
| **Average** | 54.3% | 56.3% | +2.0% | 53.5% | 55.0% | 1.5% | 2.5 | 8.9 |
