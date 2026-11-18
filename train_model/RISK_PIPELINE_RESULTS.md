# Risk pipeline prototype: first check (18 Nov 2026)

**Status: prototype. The playbook is a draft that no lawyer has reviewed, the
example labels are Claude's reading of that draft, and the check can be wrong.**

## What it does
`risk_pipeline.py`, two clause types, freelancer side, draft India playbook:

1. **Detect** the clause type with the existing embeddings-flag, LLM-confirm
   detector (few-shot, trained on CUAD training contracts only).
2. **Extract** a small form per type with the LLM (Groq `openai/gpt-oss-120b`).
   Every answer must quote the clause. An answer whose quote is not in the
   clause, or that falls outside the allowed list, is dropped.
3. **Rules** from the playbook turn the form into low / medium / high
   (= preferred / acceptable / unacceptable), or **unsure**. A high flag needs
   one quoted fact. Low needs every fact that could make it high; a missing fact
   never turns into low.

Types: IP Ownership Assignment and Termination For Convenience. Payment terms
was dropped because CUAD has no payment type and the playbook has no entry.

## Examples
19 clauses in `risk_examples.jsonl`: 14 written by Claude to cover each rule
(7 per type) and 5 real CUAD test-contract chunks. The expected risk is
Claude's reading of the draft playbook, not a lawyer's.

| | Result |
|---|---|
| Steps 2-3 alone (true type given), 3 runs after the fixes | 19/19 twice, 18/19 once |
| Full pipeline (steps 1-3), 1 run | 15/19 right, 1 unsure where high was expected, 3 not detected |

The 3 not detected:
- `ip1`, a licence-only clause (the freelancer keeps ownership). CUAD would not
  call this an IP Ownership Assignment, so the detector is arguably right, but
  it means the "preferred" IP position is never shown as low.
- `ip6` (vague IP wording) and `cuad_paxmedic_16` (a duty to help register IP):
  both expected unsure, so nothing wrong was shown, but they come out as "not
  checked" instead of "unsure".

## What works
- On clear clauses the rules match the playbook every time, and every reason
  quotes the clause.
- The quote check catches bad extraction. `cuad_hfenterp_8` (developer assigns
  present and future rights) gave unsure in 2 of 5 runs, because the quotes
  for two facts did not match the text. Once it also called the scope "project
  deliverables" when it covers future work. The same clause went high in the
  other runs. So the failure is unsure, not a wrong low.
- Two rule bugs that the examples found, both fixed before the runs above:
  a freelancer-only termination was called low when the penalty fact was
  missing, and a clause that only refers to another section was read as
  "licence only" and called low.

## What doesn't, or isn't known yet
- **19 examples, 14 written by the same Claude that wrote the rules.** They are
  clean and easy. They show the rules do what the playbook says, but they do
  not show accuracy on real contracts. No accuracy figure should be quoted from
  this.
- The same input at temperature 0 does not always give the same output
  (see `cuad_hfenterp_8`).
- Real CUAD chunks are business-to-business. Mapping "Developer" or "CRO" to
  the freelancer role is a stretch.
- A clause is judged alone. A termination clause that does not mention payment
  gets medium ("check the payment clause"), even if another clause covers it.
- The playbook sets no fair notice period, so notice length is reported but
  never judged.
- Two prompt tweaks (the `not_stated` option and the payment and penalty
  wording) were made after seeing these examples, so these results are
  optimistic.

## Next, if this direction holds
1. Run it on 20 to 30 real freelancer contracts (the Indian test set), with a
   person labelling the expected risk before seeing the output.
2. Pass neighbouring clauses (definitions, payment) into step 2.
3. Ask step 2 two or three times and call disagreement unsure.
4. Add a licence route so "you keep ownership" clauses can show as low.
