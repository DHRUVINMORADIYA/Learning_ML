# Writing in the user's voice

The permanent note should read like the user wrote it on a good day: the same words
and the same way of thinking out loud, just cleaner and in a better order. A quick
test: if the user reads a paragraph and feels "someone explained this to me", it
missed. If they feel "yes, that's how I see it", it worked.

## The current rough note is the ground truth

The examples here are calibration taken from earlier notes. The user's style keeps
evolving, so before drafting, re-read the rough note being converted and pick out a
handful of its characteristic phrases, analogies and hedges. Those win over
anything below.

## What the voice sounds like

All examples are from the user's own rough notes.

- **"We" for doing the ML, "I" for the user's own understanding, judgment and
  plans.** "We let negatives come in but in proportion of a." / "So I suppose this
  should be the only thing that matters in practical case." / "Spent some time
  understanding kernels and SVM, parking it after getting high level idea of them."
- **Thinking-out-loud questions.** "Why layers?" / "another option?" / "Better
  approach?" / "what then?" / "Why not one single model that we can train on 80%
  of total available labeled data?" These make good headings and openers.
- **The user's own analogies, kept as they are.** "w2 is like a gatekeeper", "all
  neurons looking for their 'thing' in previous layer", "a flat line vs a flexible
  line trying to find pattern", "what if we went to some mysterious world where
  dogs look like cat?", "test set should be taken as sacred", autograd as turning
  "on a switch over a variable".
- **Everyday words.** punishment, squished, bumped up, fired, quirks, chaos,
  surprised, nudge, thing.
- **Honest hedges and judgments.** "It is straightforward." / "kinda" / "concept
  wise feels rich and might come back somewhere else too" / "doesn't seem like a
  standard industry practice" / "I have yet to see this in action" /
  "interesting!" / "f around and find out".
- **Cause → effect with arrows** for short chains: "for many multiplication, it
  would keep going near zero -> so we introduce log".
- **Connections said plainly.** "Concept learned about distribution shift connects
  here." / "Somewhere this relates to VC dimension topic."
- **Notes-to-self in brackets.** "[I will have to write and run a code for the same
  by myself.]"
- **Simple sentences, plain English.** Not formal, not academic.

## Cleaning up without changing the voice

- **Fix**: spelling (destribution → distribution), missing parentheses in formulas,
  sentences that run on, obvious grammar slips.
- **Keep**: the sentence level, the words, the hedges, the analogies, and the order
  of reasoning (unless the order itself is wrong).
- **Don't**: upgrade the vocabulary, smooth sentences into polished prose, or add
  flourishes.

> Rough: "Sigmoid converts any value in a value between 0 to 1. as values get bigger
> sigmoid lean towards 1 and as they get smaller sigmoid goes towards 0."
>
> Good: "Sigmoid converts any value into a value between 0 and 1. As values get
> bigger it leans towards 1, and as they get smaller it goes towards 0."
>
> Too far: "The sigmoid maps the real line onto (0, 1), saturating at both
> extremes."

## The tutor voice to avoid

The chapter 02 draft slipped into this, and it is the most likely way for a note to
go wrong. Signs:

- **Talking to a reader**: "Note that…", "Notice…", "Let's…", "Recall…",
  "Remember…", "You might wonder…", "As we'll see…", "Here's the key idea…",
  "This is the part that sounds strange the first time", "Note what this needs:".
- **Punchlines the user didn't write**: "Same destination, two roads." / "That is
  the entire behaviour." / "A cat is still a cat; we are just seeing different
  cats."
- **Textbook words the user doesn't use**: calibrated, shatter, saturate, monotonic,
  leverage, crucially, fundamentally, canonical, intuitively.
- **Over-formatting**: tables, deep nesting, bold on every other phrase. Bold is for
  the term a paragraph defines, and not much else.

Side by side (rough note → tutor draft → user's voice):

> Rough: "How to get beta(i) value is the next question. For that, (it sounds
> strange and new but) we have to create another simple binary classifier which
> will classify if a given example is from training data or population data."
>
> Tutor: "Getting βᵢ is the part that sounds strange the first time: **train a
> second, throwaway binary classifier** whose only job is to say whether an example
> came from the training pool or the real-world pool."
>
> User's voice: "How to get $\beta_i$ is the next question. It sounds strange and
> new, but we create another simple binary classifier that tells if a given example
> is from the training data or the real-world data."

> Rough: "Although theory says that way, in practice it is au contraire good at
> generalizing."
>
> Tutor: "Theory and practice disagree here, and both are worth holding."
>
> User's voice: "Theory says it will overfit, but in practice it is au contraire
> good at generalizing."

## New words

Prefer the user's word whenever one exists. If a new term is really needed (usually
a book term that will come up again), say it in the user's words first and give the
term second: "a separate column for each value (one-hot encoding)". Math symbols are
fine, but say what they mean right after: "where $\odot$ means element by element".

## Answering the user's own doubts

When the rough note says "correct if it is different" or "not really clear why", or
contains a mistake, the permanent note gives the right version in the same voice,
built on the user's own framing and analogies: keep the gatekeeper, fix what the
gatekeeper does. Keep the user's question as the opener when there was one. Don't
narrate the mistake inside the note ("I used to think…"). The correction goes in the
report instead, so the user knows what changed.
