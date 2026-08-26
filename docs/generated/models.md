# Model matrix

| Model | Provider | Transport | Status |
|---|---|---|---|
| `MiniMax-M3` | `minimax` | `qwen_minimax_cli` | `available` |
| `abacus-glm-5.3` | `abacus` | `abacus_cli` | `candidate` |
| `abacus-grok-4.6` | `abacus` | `abacus_cli` | `available` |
| `abacus-qwen3.8-27b` | `abacus` | `abacus_cli` | `candidate` |
| `abacus-qwen3.8-max` | `abacus` | `abacus_cli` | `available` |
| `claude-opus` | `anthropic` | `claude_cli` | `available` |
| `claude-sonnet` | `anthropic` | `claude_cli` | `available` |
| `deepseek-v4-flash` | `deepseek` | `deepseek_harness` | `quota_unavailable` |
| `deepseek-v4-pro` | `deepseek` | `deepseek_harness` | `quota_unavailable` |
| `fx-glm-5.2-promo` | `fx` | `fx_cli` | `available` |
| `gemini-free-fast` | `antigravity` | `qwen_omniroute_cli` | `available` |
| `gpt-5.6-luna` | `openai` | `codex_cli` | `available` |
| `gpt-5.6-sol` | `openai` | `codex_cli` | `available` |
| `gpt-5.6-terra` | `openai` | `codex_cli` | `available` |
| `kat-coder-free` | `zenmux-free` | `qwen_omniroute_cli` | `candidate` |
| `kimi-free-code` | `huggingchat` | `qwen_omniroute_cli` | `candidate` |
| `kimi-official` | `moonshot` | `kimi_cli` | `available` |
| `local-qwen36-agent` | `local-ai` | `qwen_omniroute_cli` | `candidate` |
| `qwen-web-max` | `qwen-web` | `qwen_omniroute_cli` | `candidate` |

## Components

- `abacus-cli` current `2.6.9`; channel=`latest`; policy=`manual_promote`; active=true
- `claude-cli` current `2.1.245`; channel=`stable`; policy=`manual_promote`; active=true
- `codex-cli` current `0.149.1`; channel=`stable`; policy=`manual_promote`; active=true
- `deepseek-harness` current `0.1.1-rc.2`; channel=`latest`; policy=`manual_promote`; active=true
- `fx-cli` current `0.0.4`; channel=`stable`; policy=`manual_promote`; active=false
- `kimi-cli` current `0.38.0`; channel=`stable`; policy=`manual_promote`; active=true
- `opencode-cli` current `1.18.23`; channel=`stable`; policy=`manual_promote`; active=true
- `qwen-cli` current `0.22.1`; channel=`latest`; policy=`manual_promote`; active=true
