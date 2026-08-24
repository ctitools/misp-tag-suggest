# GPT-5.6 Sol Flex canary report

**Date:** 2026-08-24  
**Outcome:** fail-closed halt before generation

## M0 result

The direct Responses API runner and its mocked emergency-brake suite were completed
before any live request. The runner:

- reads the credential only from the fixed mode-`0600` file;
- sends only to `https://api.openai.com/v1/responses`;
- requests exactly `gpt-5.6-sol` and `service_tier="flex"`;
- uses a 900-second timeout, strict Structured Outputs, explicit GPT-5.6 prompt
  caching, `store=false`, and sequential requests;
- performs a structural privacy gate before opening HTTP;
- pessimistically reserves spend before each attempt;
- retries only HTTP 408 and an exact HTTP 429 Resource Unavailable response;
- persists a global brake on refusal, filtering, policy errors, malformed output,
  unexpected service tier/model, privacy failures, or budget failures.

Mocked tests demonstrated that a refusal or content filter on request N prevents
request N+1. Privacy, malformed JSON, wrong model/tier, budget, seed-contract and
policy-error cases were also covered.

## Live canary result

The selected canary contained 20 taxonomy-only bundles, balanced five per target
family. It contained no event text, UUID, URL, event ID, organization or analyst
field. The stable prompt hash and strict-schema hash were recorded before transport.

At `2026-08-24T19:23:31Z`, the first request returned HTTP 400. The response body did
not match the runner's policy/safety/filter signal detector, but its detailed message
was intentionally not persisted because API errors can echo request material. The
runner wrote the global emergency brake immediately.

Consequences:

- exactly one HTTP attempt;
- zero generated bundles;
- request 2 was not sent;
- no usage or settlement object was returned;
- no OpenAI model cost is asserted;
- the ledger conservatively retains the request's USD `0.090835` pessimistic
  reservation;
- pilot generation, auditing, synthetic training, adoption evaluation, test read and
  integration were not performed;
- the real test split remains untouched.

This is an invalid-request/compatibility failure, not evidence that Sol accepted or
rejected the defensive content. The run was not retried and the prompt was not
changed.

## Credential disposition

The local ephemeral key file was removed after the halt. The operator must also
revoke the credential in the OpenAI project dashboard; local deletion cannot revoke a
server-side key.

## Required recovery

A future run requires a fresh credential and explicit authorization after the HTTP
400 is diagnosed from provider-side request logs. It must be treated as a new canary,
not a continuation. If provider logs indicate any policy or filtering cause, the
experiment must remain stopped and no prompt mutation is permitted.
