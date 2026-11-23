# Retraction: a design for tombstones on wrong values

Status: **proposal, not built.** [ROADMAP.md](ROADMAP.md) §5 asks for this design to be written
down and agreed before any code. Nothing here is implemented. The six questions at the end
are the owner's to answer; everything above them is what the code will do once they are.

## The problem

A value is recorded, turns out to be wrong, and is taken out. Later an agent reads the same
wrong source (an old document, an earlier conversation, an ingested note) and records it
again. Nothing stops it: the new fact is valid, and nobody is told that this exact statement
was already withdrawn once.

What exists today does not cover it:

- `forget` removes the fact and every trace of it. That is its job, and it means the memory
  cannot know the statement was ever there.
- `supersede(old, new)` closes the old fact and points it at the new one. A later `remember`
  of the old text creates a fresh valid fact. If it shares an entity and enough words with
  the new one it is flagged `possible conflict`; otherwise it is simply saved.
- `ingatan undo` of a `create` closes the fact. It carries no reason and no later `remember`
  looks at it.

## What retracting is

A separate operation on a fact that **stays in the file**: "this was recorded, and it was
never true". It is not a softer `forget` and shares no code with it.

`facts.retract(ctx, fact_id, reason)`:

1. requires a non-empty `reason` (redacted like any text), as `confirm` does;
2. closes the fact (`valid_to = now`) and sets two new columns, `retracted_at` and
   `retract_reason`, in one transaction;
3. resolves the fact's open conflict pairs (resolution `retract`): a withdrawn fact
   contradicts nothing any more;
4. writes one log entry with a new op, `retract`, whose `before` holds the old `valid_to`.

`ingatan undo` of that entry clears both columns, restores `valid_to` and reopens the pairs
it resolved, the way undoing a `supersede` does today. So a retraction made by mistake is
taken back like any other change.

Schema, as the next free migration:

```sql
ALTER TABLE facts ADD COLUMN retracted_at TEXT;
ALTER TABLE facts ADD COLUMN retract_reason TEXT;
```

No new index: the lookup below goes through `facts_text_key`, which exists.

### Why the tombstone is the fact row and not a table of its own

A `tombstones(text_key, ...)` table was the other candidate. It is rejected for one reason:
it would outlive the fact. `forget` would have a second place to scrub, and the key is a hash
of the fact's text, which is a trace of a forgotten fact. The roadmap's constraint is that a
tombstone can never be how a forgotten fact is remembered.

With the tombstone being the row, there is nothing to get wrong: `forget` deletes the row
(with `secure_delete` on, the FTS entries, the vector, the queue row and the log content
going with it, as today), and the tombstone is gone because it never was anything else.
After a `forget`, the same text can be remembered again and nothing says it was ever there.

## What a later `remember` does

Before the repeat check that `remember` already makes, it looks for a retracted fact that
says the same thing:

- same `text_key` (the text with case, spacing and trailing punctuation ignored),
- same set of entities, same tier and category,
- visible to the caller, with the visibility clause every query uses.

That is exactly the test `_find_duplicate` applies to open facts, pointed at retracted ones.
One index lookup, whatever the size of the store.

If one is found, nothing is saved, and the result says so:

> Not saved. This was recorded before and withdrawn on 2026-11-20: "the supplier confirmed
> the old price list was wrong" (f_3a9c…). If it is true after all, tell the user.

`RememberResult.fact_id` and `proposal_id` are both `None`, and `similar` names the retracted
fact, so the caller has an id to show.

### Why the match is exact

Anything looser needs a similarity score, and [EMBEDDINGS.md](EMBEDDINGS.md) records why
that cannot work here: for every model measured, two statements that differ only in a value
("Rp 15.000" against "Rp 18.000") score higher than two true paraphrases. A fuzzy tombstone
would block the corrected value along with the wrong one. That is the opposite of what it is
for.

The cost is stated plainly: a paraphrase of a withdrawn statement is saved. An agent that
reads the same wrong sentence from the same wrong source produces the same text, and that is
the case this catches. An agent that rewords it does not get caught.

### What visibility means for the match

- A retracted `agent`-tier fact blocks only the agent that owned it. Another agent that
  remembers the same text in its own notes is not told: it could not see the fact, and the
  answer would reveal that it existed.
- A retracted `business` fact blocks every agent of that business, in the `business` tier.
- A retracted `global` fact blocks the `global` tier in every business.
- A retraction in one tier does not block the same text in another tier.

## Where a retracted fact shows

- `recall` never returns it by default: it is closed.
- `recall(include_closed=true)` returns it with the status `retracted` (a third closed
  status beside `superseded` and `expired`), so the eval's stale check keeps working: a
  retracted fact shown as anything else is a stale result.
- `history(id)` shows `create` and then `retract` with its reason. `history(entity)` lists
  it with the entity's other facts.
- `brief` never shows it.
- `ingatan log` and `ingatan diff` show the entry like any other.

## What this deliberately leaves out

- **No tombstone for a forgotten fact.** Ever. See above.
- **No fuzzy matching.** See above.
- **`undo` of a `create` is not a retraction.** It has no reason, and "I did not mean to
  write that" is not "that is false". A fact closed by `undo` blocks nothing.
- **Nothing crosses a business.** The lookup runs in the caller's business file and
  `global.db`, like every other query.

## How it will be tested

- `retract` closes the fact, sets both columns, resolves its pairs, writes one entry; a
  missing reason is refused; a closed fact cannot be retracted twice.
- `remember` of the same text is answered and not saved; of a changed value is saved; of the
  same text with other entities is saved; in another tier is saved.
- Another agent's retracted private fact blocks nothing and is not mentioned.
- `undo` of a `retract` reopens the fact and its pairs, and the text can then be remembered
  as a repeat of the open fact.
- `forget` of a retracted fact leaves no byte of its text or its reason in either database
  file or its WAL (the byte-scan of `tests/facts/test_forget.py`), and the same text is then
  saved without a word.
- `recall(include_closed=true)` shows `retracted`; default `recall` and `brief` do not show
  the fact.
- Eval: a new category `retract` and `evals/scenarios/retract-01-nilai-salah-kembali.json`:
  a wrong price is remembered, retracted, and remembered again a week later by an agent that
  reads the same source. The scenario needs two additions to the format: an op `retract`,
  and on `remember` the expectation `expect_not_saved`.

## Questions for the owner

Each has a recommendation. The design above assumes the recommendations.

1. **Refuse, or save and warn?** The roadmap says the later `remember` is "answered with
   'this was recorded before and withdrawn' instead of being saved silently". That allows
   both. *Recommended: refuse.* A fact that is saved with a warning is valid, is recalled,
   and the warning is read once by an agent and then gone.

2. **How does a statement come back when it turns out to be true after all?** With a refusal
   there must be a way. *Recommended: only the user, by `ingatan undo` of the retraction.*
   An override parameter on `remember` would be used by exactly the agent this is meant to
   stop: it reads the wrong source, is told no, and tries again with the flag.

3. **Who may retract?** *Recommended: the rule `supersede` follows.* An agent retracts its own
   `agent`-tier facts. For `business` and `global` facts it retracts directly in the default
   write-then-undo mode, and queues the change for approval when `require_approval` is on.
   The user retracts anything from the CLI.

4. **An eighth MCP tool, or CLI only?** Today there are seven tools and the README says so.
   *Recommended: an eighth tool, `retract(fact_id, reason)`.* The person who notices a wrong
   value usually says so to an agent in a conversation. If the agent has no tool for it, it
   will reach for `forget` (which destroys the evidence, and works only on its own facts) or
   leave the fact standing. The cost is one more tool description in every session's
   context and a README that says eight.

5. **Should a superseded text get a note as well?** `supersede` means "was true, then
   changed", and values do go back to what they were, so refusing would be wrong.
   *Recommended: save it as today, and add one sentence to the result:* "the same statement
   was replaced by f_… on DATE". It costs one more index lookup per `remember` and no
   schema change. It can also be left out of the first version entirely.

6. **Is the status name `retracted` right, and should the reason be shown in `recall`?**
   *Recommended: `retracted`, and the reason only in `history`,* so that a closed fact's line
   in `recall(include_closed=true)` stays one line.
