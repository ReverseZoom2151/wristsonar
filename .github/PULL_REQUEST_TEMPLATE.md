# What this changes

<!-- One or two sentences. The diff shows what changed, so say why. -->

Closes #

## Checks

- [ ] `make check` passes locally, on Python 3.11 and 3.12 if you can
- [ ] `pytest` passes on a clean checkout with no dataset downloaded
- [ ] New behaviour is covered by tests that use synthetic fixtures
- [ ] Commit subjects are imperative and the bodies explain why

## Does this touch evaluation

<!--
Answer yes if the change affects any reported number, or anything upstream of
one: signal processing, features, model, training, splitting, metrics, or the
report layer. If yes, the section below is required.
-->

- [ ] No, this change cannot affect any reported number
- [ ] Yes, and the section below is filled in

### Splits run

<!--
List every split you ran and the number it produced, with the protocol that
produced it. Paste protocol.describe() output where you can. Leave a split as
"not run" rather than omitting it, so the gaps are visible.
-->

| Split | Ran | Number | Protocol |
| --- | --- | --- | --- |
| within-session | | | |
| cross-session | | | |
| cross-user | | | |
| cross-device | | | |

Ground truth source:

Calibration budget used:

Held-out poses in the test set:

<!--
A within-session number is not a headline. If it is the only split you ran,
say so plainly and explain why the honest splits were not available.
-->

## Anything reviewers should look at closely

<!-- Tradeoffs you made, things you are unsure about, follow-up work. -->
