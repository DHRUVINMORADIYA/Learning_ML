---
title: Rough Note Random
parent: Rough Notes
nav_order: 4
---

Note for writing down things encountered from outside of d2l book.

YT channels
Vizuara
StatQuest


1. t-test, p-value, central limit theorem 

- The great mental models of artificial intelligence series
    1. Gradient Handoff - when you can't write a rule, describe the goal and let the gradient find it.
    - matches with first AI intro note. Let the data find their way to output rather than we human paving the path.

    2. Build the hunger, not the skill -optimize a humble little task; the real understanding arrives as a side effect.
    - focus on core need rather than how to and where to. Rather than supplying fish or even teaching fishing, teach HUNGER. and let them come up with technics.

    3. Vectors - it showed importance of vectors. How they live under very complex models that feels almost magical. But under the hood it's just vectors. They are like alphabets. We can convert all media formats into vectors. It also demonstrated importance of tokenization and vectorization. They are pre-requisite parts of ML but often they are shadowed by more "cool" parts of ML.

    4. Compression of data and projecting it back to original size
    - This one feels strange as the usecase of this has not been encountered yet. But it says at many places (stable diffusion models, Deepseek, LoRA, etc) - they all do this where they compress data while not loosing the features to accomodate with GPU capacities and after the processing done they project it back.
    - It feels counter-inuitive but he showed that sometimes not all data are required. We can compress while not loosing features, we do processing at bottle-neck and then expand it back to real form.
    - I have yet to see this in action.