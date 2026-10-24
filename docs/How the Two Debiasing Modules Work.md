# How the Two Debiasing Modules Work

Oct 14, 2026 · 
A relational GNN builds each person's vector by collecting messages from what they are connected to: university, employer, spouse, occupation, country. Bias can enter through two doors, and each module closes one.
At a glance

Two bias doors · one module each
Module 1 closes the popularity door and Module 2 closes the direct door; neither alone removes the whole bias.
Module 1: degree-robust aggregation (popularity door)
This module makes how well the model knows someone depend less on how famous they are.
The problem. A standard GNN adds up messages from every neighbour. A famous scientist with 500 connections gets a rich, detailed vector, while a lesser-known scientist with 3 connections gets a thin, vague one. The model then predicts awards much better for the famous person simply because it knows more about them. Famous people in Wikidata are disproportionately men, so this turns into gender bias.
What the module changes:
	•	Neighbour budget: each person receives at most a fixed number of messages per relation, so famous people cannot hoard information.
	•	Average instead of sum: more neighbours no longer automatically means a "bigger" representation.
	•	Low-degree completion: for people with very few connections, a small learned component fills in the missing information, like an educated guess from similar people. It is trained by hiding most connections of famous people and teaching the component to reconstruct their full representation.
Module 2: two-channel representation with an adversary (direct door)
This module keeps gender, including hidden proxies, out of the part of the vector used to predict awards.
The problem. The model can learn "this person is a woman, so less likely to win this award", either from the gender edge itself or from proxies such as spouse, "actress" or certain names.
What the module changes. Each person's vector is split into two parts:
	•	Sensitive channel: the only place gender information is allowed to go.
	•	Content channel: everything else (field, achievements, institutions). Award predictions use only this part.
The adversary is a small extra classifier that tries to guess the person's gender from the content channel. The GNN is trained to make it fail: if the adversary can still guess gender, the GNN is penalised, so it learns to keep gender out of the content channel.
It works like blind peer review: the reviewer (the award predictor) sees only the anonymised part, and someone actively checks that the identity cannot be guessed from it.
Why both modules are needed
Each module closes only one door, so fixing one leaves the other open.
Module
Fixes
Leaves open if used alone
Degree-robust aggregation
"The model knows famous people better"
The model can still use gender itself
Two-channel + adversary
"The model uses gender itself"
Famous people still get better predictions
Using both is the main point of the project: it removes each source of bias separately, and ablations show which module removes which part.
