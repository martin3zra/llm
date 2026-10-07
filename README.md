# llm

A small, vendor-neutral interface for talking to an LLM with tool calling, plus
adapters for Anthropic (Messages API) and OpenAI (Responses API).

The caller runs its own tool-calling loop against `llm.Provider` and never
touches a vendor SDK type, so switching or adding a provider doesn't change
that loop. Every adapter is built from one user's own key (`anthropic.New`,
`openai.New`) and never falls back to a server-wide key from the environment.

- `llm.Provider` streams one model turn: text deltas, tool calls, then done or
  an error.
- `llm.ChatRequest.ToolChoice` forces a named tool, for structured extraction.
- `llm.Verifier` checks a key and a model without spending tokens, for when
  they're saved.
- `EventDone` carries the turn's `llm.Usage` (input, output, cache write and
  cache read tokens), for metering.
- `agent.Run` (in `llm/agent`) runs one user turn of the tool-calling loop:
  stream the answer, call the tools the model asks for, feed results back,
  and repeat. The caller keeps the history: `Run` takes it and reports the
  messages the turn added, plus summed usage, in its done event. A tool's
  `Result.Payload` reaches the caller but never the model.
- `llm.Error` sorts vendor failures into `KindAuth`, `KindRateLimit`,
  `KindInvalidReq`, `KindUnavailable` and `KindUnknown`. Show the `Kind` or
  `Message`, never the wrapped cause.

Used by `cactus-agent` and `tallyo` as a sibling checkout, through a `replace`
directive in each one's `go.mod`:

```
replace github.com/martin3zra/llm => ../llm
```

## Testing

```bash
go test ./...
```

The tests cover request mapping and error normalization, with `Verify` run
against a local test server. They don't call a real API.
