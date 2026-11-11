# Incident scenarios (still wanted)

The eval set needs about ten scenarios taken from **real incidents** that only the project owner can supply: a time an agent used an outdated fact, mixed up two businesses, showed a private note to the wrong agent, missed a fact because of an Indonesian word form, and so on (NFR-7, risk R-13). The scenarios in the parent folder are all `"source": "pattern"`. Ten incident scenarios are still wanted, and none have been written: they must not be invented.

## What to send

For each incident, in plain words and in the language it happened in:

1. What the agent did or answered wrong, and what it should have done.
2. The facts that were stored at that time (the text, as close to the original as you can), who stored each one (business and agent), and the tier.
3. What changed between storing the facts and the mistake (a price change, 31 days passing, a supersede, a proposal waiting).
4. The question or query that went wrong, word for word.

Remove secrets and customer names before sending. Redaction does not run on these files.

## How an incident becomes a scenario

Put one file here per incident, named `<id>.json`, in the same format as the other scenarios (see `evals/README.md`), with:

- `"source": "incident"`
- an `id` that starts with the category and a short topic, for example `stale-incident-07-tarif-vendor`
- the original wording of the texts and queries where possible, so the scenario keeps the real difficulty (word forms, abbreviations, mixed Indonesian and English)

The runner reads this folder automatically. A scenario that fails because of real product behaviour stays in the set with an `xfail` reason; it is not softened.
