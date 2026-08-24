# Synthetic scenario specification

**Status:** schema and seed preparation only; no API generation is performed by these files.

## Purpose

Synthetic rows may vary the language and presentation of fictional defensive CTI
reports. They do not invent the applicable taxonomy target. Every label contract is
selected locally from the frozen taxonomy before generation.

Synthetic data is training-only. Chronological analyst-authored validation and test
remain the only evidence used for model adoption.

## Privacy boundary

Material permitted in an external generation request:

- public taxonomy and galaxy tag identifiers;
- public taxonomy descriptions, aliases and external IDs;
- aggregate label-frequency buckets;
- locally selected candidate and confusing-tag sets;
- fictional scenario IDs and standardized reserved indicators.

Material forbidden in an external generation request:

- real event text or titles;
- real attributes, indicators, comments or object values;
- event IDs or UUIDs;
- organization or analyst names;
- retrieved neighbors, predictions or rationales tied to real events;
- examples copied or paraphrased from validation or test.

The seed builder enforces this boundary by reading tag documents and aggregate
frequencies only. Its output contains no event row.

## Fixed six-scenario bundle

Each target produces one counterfactual group that must stay together across any
split or sampling operation:

1. `explicit_positive`: direct evidence for the target, relevance 2;
2. `implicit_positive`: paraphrased evidence without copying the tag name, relevance 2;
3. `partial_positive`: incomplete but useful evidence for the target, relevance 1;
4. `same_family_confuser`: evidence for a preselected competing candidate and not the
   target, target relevance 0;
5. `unrelated_negative`: no candidate is supported, relevance 0;
6. `insufficient_evidence`: plausible report with too little evidence to tag,
   relevance 0.

The local seed fixes target, candidates, confuser and relevance values. Generation
fills only the fictional event and evidence paths. A verifier may reject an invalid
scenario, but may not replace the target with an arbitrary tag.

## Fictional indicator policy

Only these indicator forms are accepted:

- domains and hostnames below `.example`;
- HTTP(S) URLs whose host is below `.example`;
- IPv4 from `192.0.2.0/24`, `198.51.100.0/24` or `203.0.113.0/24`;
- IPv6 from `2001:db8::/32`;
- email addresses below `.example`;
- placeholder SHA-256 values consisting of 64 hexadecimal characters and beginning
  with 32 zeroes.

UUID-like strings are forbidden. Generated rows are not intended for import into
MISP and must remain marked as fictional training data.

## Defensive content boundary

The request asks for short analyst-facing incident summaries and inert indicators.
It must not request executable code, payloads, exploit steps, credentials,
persistence instructions or operational attack guidance. Descriptions may name
high-level observed behavior when needed to support a defensive taxonomy tag.

A refusal, `content_filter`, policy-related API error, or warning that suggests the
request was filtered stops the entire run. It is not retried with euphemisms or an
evasion-oriented prompt.

## Generation prompt contract

A future direct-OpenAI request will use `gpt-5.6-sol` with
`service_tier="flex"` and strict structured output. The stable developer prefix will
contain:

- this privacy and defensive-content policy;
- the six variant definitions;
- reserved-indicator rules;
- the JSON Schema;
- concise positive and negative format examples containing fictional data only.

The stable prefix must be at least 1,024 tokens and end at an explicit cache
breakpoint. Use `prompt_cache_options.mode="explicit"` and one stable
`prompt_cache_key`. Variable taxonomy seed data follows the breakpoint. Warm the
cache once before bounded concurrency and record `cached_tokens` and
`cache_write_tokens`.

The variable request provides one seed and directs the model to:

1. preserve all IDs, candidate tags and fixed labels exactly;
2. create six mutually distinct fictional events in the requested language;
3. avoid spelling the target tag in positive event text unless ordinary prose
   independently requires a public family name;
4. provide only evidence paths that resolve to generated fields;
5. use only reserved indicators;
6. return no prose outside the schema.

## Admission checks

A generated bundle is admitted only when all checks pass:

- strict JSON Schema validation;
- each required variant occurs exactly once;
- target appears in the supplied candidates;
- expected labels are candidate-only and match the fixed local contract;
- evidence paths resolve and are empty for negative/abstention variants;
- every typed indicator passes the reserved-indicator validator;
- no UUID-like values or duplicate events occur;
- no exact or near duplicate exists in real train, validation or test;
- blinded verification confirms that cited evidence supports the fixed label;
- provenance contains request ID, model, service tier, prompt/schema hashes, usage,
  cache reads/writes, cost and all verifier decisions.

Rejected attempts remain in the private append-only ledger. They are never silently
rewritten into accepted data.

## Shortcut and release controls

Before scale-up:

- train response-only and metadata-only lexical baselines;
- evaluate template-held-out bundles;
- measure whether a classifier can distinguish real from synthetic text;
- compare real-only LFM with real plus 25% synthetic on real validation;
- manually audit a stratified sample and every model disagreement.

Synthetic data and model-reviewed labels remain silver. Generated data, trained
weights and provider terms require a separate redistribution review before public
release.
