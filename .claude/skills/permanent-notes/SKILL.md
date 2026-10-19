---
name: permanent-notes
description: Turns the user's hand-written rough notes (rough-notes/Rough_Note_NN.md, written while reading the D2L "Dive into Deep Learning" book) into a permanent chapter note (permanent-notes/NN-topic/Notes.md) that reads in the user's own voice and vocabulary, loses nothing from the rough note, links ideas across chapters, writes math as clean LaTeX, and lets the user re-learn the chapter fast with the Feynman method; also updates permanent-notes/Index.md. Use this whenever the user says they finished a chapter or section and the rough notes are ready, asks to create, convert, distill, clean up or promote rough notes into permanent notes (often spelled "permanant"), or wants an existing permanent note redone or extended from rough notes, even if they don't name this skill.
---

# Rough notes → permanent notes

## What this is for

The user reads the D2L book and keeps a working journal in `rough-notes/`: one
`Rough_Note_NN.md` per chapter or two, plus `Rough_Note_Random.md` for things from
outside the book. When a chapter is done it becomes a permanent note in
`permanent-notes/NN-topic/Notes.md`, published as a Jekyll (just-the-docs) site.

The permanent note has one job. If the user forgets this material some day but still
has their general mental models, reading the note should get them back on track
fast, to the point where they can explain every idea in plain words (Feynman
method). Every rule below serves that job.

## The user's eight rules

These come straight from the user. The notes under each say what they mean in
practice.

1. **Match the user's English level and writing style.** Simple sentences, everyday
   words, the user's way of thinking out loud. Read `references/voice.md` before
   drafting.
2. **No unnecessary long details.** Cut repetition, filler and false starts. Don't
   add mini-lessons on things the rough note didn't touch. Compressing never means
   dropping an idea; rule 7 wins.
3. **Connected, not isolated.** It should be possible to move between topics without
   feeling lost:
   - order sections so each leads into the next, with a one-line bridge where the
     rough note jumps;
   - when an idea from an earlier chapter shows up again, link that permanent
     note's section (relative link with anchor);
   - when the same idea appears twice in this chapter, explain it once and link the
     other place;
   - keep the user's own connections ("X connects here") and turn them into links;
   - update `permanent-notes/Index.md` (step 6).
4. **Math formulas and explanations, nicely.** LaTeX for every formula, symbols
   explained in words right after, a one-line derivation when the rough note
   derived something. Ascii math (`e**x`, `1 / 1 + exp(-x)`) and formula
   screenshots become proper LaTeX.
5. **Feynman-ready.** Every concept should answer: what is it, what problem it
   solves (why it exists), and how it works (mechanism or formula), with the user's
   example when there was one. If the rough note has the "what" but not the "why",
   add one plain sentence for the why. This is the main reason to add anything.
6. **The user's vocabulary, little new jargon** (soft rule). Use the user's words
   for things ("punishment" for a weight's share of the gradient, "squished",
   "bumped up"). If a new term is really needed, the user's words come first and
   the term second.
7. **Nothing lost.** Every idea, formula, example, number, analogy, opinion,
   connection, code snippet and open question in the rough note appears in the
   permanent note. Steps 2 and 5 make this checkable.
8. **The user's point of view, not a teacher's.** The note is the user reminding
   themselves, not someone explaining to a student. "We" for doing the ML, "I" for
   the user's own understanding, judgments and plans. See `references/voice.md`.

## Workflow

### 1. Gather

- Find the rough note(s) for the chapter. One rough note can hold one chapter or
  two, and the user's chapter numbers differ from the book's (permanent `03` was
  D2L chapter 5). If the mapping isn't clear from the headers, ask.
- Read the whole rough note. Open every embedded image (`![...](image-N.png)`) with
  the Read tool. They are usually formula screenshots from the book or from an
  explainer chat, and their content counts as rough-note material for rule 7.
- Skim `rough-notes/Rough_Note_Random.md`. Don't merge it in, but if an item there
  clearly connects to this chapter, plan a one-line pointer to it.
- Read `permanent-notes/Index.md` and skim every existing
  `permanent-notes/*/Notes.md` (headings at least) to know what earlier chapters
  already explain and which anchors can be linked.
- Look for practice code for this chapter (files in the chapter folder, or in
  `rough-notes/code/` made for it). Get the GitHub base URL from
  `git remote get-url origin`.
- If the target `Notes.md` already has content, it may hold the user's own edits.
  Extend it instead of rewriting, keep their wording, and say what changed.

### 2. Inventory

Before writing, list every item in the rough note and its images: concepts,
definitions, formulas, examples and numbers, analogies, the user's opinions and
judgments, connections they made, code snippets, and anything flagged as unclear,
parked or "more later". For a long rough note keep the list in a scratch file
outside the repo. This list is the checklist for step 5.

While listing, mark two kinds of item:

- **Wrong or shaky**: a formula slip, a misunderstanding, a "correct if it is
  different". The note states the correct version in the user's voice, and the
  report lists the change.
- **Open**: "not really clear why", "yet to learn", "parking it". If the book or
  basic math answers it, answer it in the note and keep the user's question as the
  opener ("Why the transpose? ..."). If it stays open, it goes to "To revisit".

### 3. Plan the storyline

Write the Summary first: the chapter's storyline in a few short lines, the way the
user would retell it. It is the re-entry point for someone who has forgotten the
chapter, so every later section should be reachable from it.

Then order the sections. Book order is a good default, but group things that
belong together. A drawback discussed under an activation function and again under
numerical stability gets explained once, with a link from the other place.

### 4. Write

Read `references/voice.md` now, then re-read a few paragraphs of the rough note to
tune into its current voice. Follow the format conventions below.

### 5. Verify

- **Coverage**: walk the inventory and find each item in the draft. Anything
  missing goes back in.
- **Voice**: read the draft as the user would. Anything that sounds like a tutor, or
  uses words the user never uses, gets rewritten (checklist in
  `references/voice.md`).
- **Feynman**: for each section, could the user explain it to a friend using only
  what's written? If a "why" is missing, add one plain sentence.
- **Mechanics**: run `python <this-skill-dir>/scripts/check_note.py <path/to/Notes.md>`.
  It checks relative links and #anchors, GitHub links to repo files, `$$` balance and
  leftover image references, and points at tutor-sounding phrases. Fix what it
  reports.

### 6. Update Index.md

- Add the chapter to **Path** in the same format as the existing entries:
  `N. **[NN · Title](folder/Notes.md)**:` followed by a one-line summary, wrapped
  near 85 characters with a 3-space indent.
- For each **Concepts** entry the note links to, add the note to its `Cited by:`
  line.
- When a topic now runs through three or more chapter notes and has no Concepts
  entry, add a short one (broad, a few lines, links into each chapter, `Cited by:`).
  Mention it in the report so the user can drop it.

### 7. Report back

Keep it short. The user needs:

- the files written or changed;
- **corrections**: every place where the note says something different from the
  rough note (rough said X, note says Y, short why). This matters because the note
  now states things the user didn't write;
- **additions**: the "why" sentences and links that weren't in the rough note;
- **coverage**: confirmation that every inventory item made it in, and anything
  left out on purpose (rare, with the reason);
- optionally, gaps noticed in older permanent notes (material from an older rough
  note that its permanent note lacks). Offer to fix them; don't fix them unasked.

## Format conventions

Match chapters 01 and 02.

- **Location**: `permanent-notes/NN-kebab-title/Notes.md`. `NN` is two digits, the
  next number after the highest existing folder. The title is the book chapter's
  name.
- **Front matter**:
  ```yaml
  ---
  title: NN Title
  parent: Permanent Notes
  nav_order: <NN + 1>
  ---
  ```
- **Opening**: `# NN · Title`, then
  `Builds on [NN-1 · Title](../folder/Notes.md). Book: D2L chapter N.`
- **`## Summary`** comes first.
- **Sections**: `##` for topics, `###` for sub-topics. Short plain headings; the
  user's own questions work well ("Why layers?"). Don't start a heading with a
  number, it breaks the anchor.
- **Math**: display math as `$$` on its own lines, with a blank line before and
  after; inline math as `$...$`. Keep notation consistent with earlier notes:
  $\mathbf{x}$, $\mathbf{W}$, $\mathbf{b}$, $\mathbf{o}$ for outputs/logits,
  $\hat{y}$, $\mathbf{h}$ for hidden values, $\phi$ for an activation function,
  $\sigma$ for standard deviation.
- **Images**: never link the rough-notes images from a permanent note. Formulas
  become LaTeX; a diagram becomes a short description in words.
- **Code**: short snippets only when they carry meaning (an API call the user
  noted), fenced with the language name.
- **Practice files**: only when the chapter has its own code. Copy the chapter
  01/02 format, `### [File.py](<github-base>/blob/main/<path>){:target="_blank"}`,
  plus a line or two on which
  [level of abstraction](../Index.md#levels-of-abstraction) it sits at. A single
  older code file can just be linked inline where it's relevant.
- **Links**: relative for notes (`../02-folder/Notes.md#anchor`), GitHub URLs for
  code files. Anchors follow kramdown: drop leading non-letters, keep only letters,
  digits, spaces and hyphens, turn spaces into hyphens, lowercase
  ("Softmax revisited: LogSumExp" → `#softmax-revisited-logsumexp`).
- **Ending**: a `---` line, then `## To revisit` with every thread still open in
  the rough note (parked topics, "more later", "not clear", "yet to learn").

## Don't

- Edit the rough notes or delete their images. They are the user's journal.
- Commit or push. The user commits on their own schedule.
- Write new practice code. That's a separate job the user asks for explicitly.
