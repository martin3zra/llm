package llm

import "fmt"

// Kind normalizes provider-specific errors so callers can react without
// knowing which vendor is behind the request. Never surface a raw provider
// error (it may embed the API key or other request details) to an end user —
// translate to Kind first.
type Kind string

const (
	KindAuth        Kind = "auth"        // invalid or revoked API key
	KindRateLimit   Kind = "rate_limit"  // provider is throttling
	KindInvalidReq  Kind = "invalid_req" // bad model name, malformed request, etc.
	KindUnavailable Kind = "unavailable" // provider outage / timeout
	KindUnknown     Kind = "unknown"
)

// Error wraps a provider-specific failure with a normalized Kind. Message is
// safe to show to the key's owner; it must never contain the API key.
type Error struct {
	Kind    Kind
	Message string
	cause   error
}

func (e *Error) Error() string {
	return fmt.Sprintf("provider: %s: %s", e.Kind, e.Message)
}

func (e *Error) Unwrap() error { return e.cause }

func NewError(kind Kind, message string, cause error) *Error {
	return &Error{Kind: kind, Message: message, cause: cause}
}
