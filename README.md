Problem statement: Knowledge-graph link prediction models learn to infer missing relationships from existing facts, but their predictions 
may reflect social biases and unequal representation in the underlying data. This project investigates whether gender-related disparities 
in predicting awards received by people are associated with gender information encoded in entity representations, differences in graph 
connectivity, or both. The central research question is: To what extent are gender-related disparities in award link prediction associated 
with entity degree, and can these patterns be distinguished through controlled experiments?

Dataset: We will use Wikidata5M, a large knowledge graph constructed from a July 2019 Wikidata and Wikipedia snapshot, containing 
approximately 4.6 million entities and 20.6 million training triples. Because training on the complete dataset would be computationally 
demanding for this project, we will construct a reproducible, appropriately sized subsection focused on people and relevant relationships. 
The primary prediction task will use the Wikidata property P166 (award received); P21 (sex or gender) will identify groups for comparison, 
and graph degree will measure each entity’s connectivity in the training graph. P106 (occupation) may be included as contextual 
information if feasible. Any regional analysis using P27 (citizenship) will be optional and conducted only if group sizes support 
meaningful comparisons. The subset, filtering criteria, and train/validation/test split will be documented to support reproducibility.

Problem challenges and related work: Existing research, collected in the GitHub repository’s research folder, examines social bias in 
knowledge-graph embeddings, data imbalance, degree-related bias in graph learning, and fairness-aware or counterfactual methods. This 
research motivates our question: to what extent are gender-related disparities in knowledge-graph award prediction associated with 
differences in entity connectivity, and how can these patterns be distinguished from other sources of bias? The main challenges are 
incomplete award records, uneven representation across demographic groups, and the difficulty of distinguishing correlation from causation.
We will therefore audit a manageable dataset subset, establish a baseline model, and compare prediction performance and group-level 
disparities across degree groups. More complex counterfactual interventions will be considered only if time and computational resources 
permit.

Baselines and methodology: We will first audit the dataset and report the representation of gender groups, award triples, and degree 
distributions. We will then implement a knowledge-graph embedding baseline, such as TransE or DistMult, and, if computationally feasible, 
a relational graph neural network (R-GCN) with a DistMult decoder. Predictive performance will be evaluated using mean reciprocal rank (MRR)
and Hits@1/Hits@10. Group-level differences will be examined using clearly defined ranking-based evaluation procedures and degree-stratified
comparisons. If resources permit, a probe classifier will test whether gender can be predicted from frozen entity embeddings, and 
controlled ablations or a limited model-depth comparison will explore the contribution of structural information. A path-specific 
counterfactual fairness intervention will be treated as a stretch goal rather than a requirement for project completion. We will use 
fixed data splits, documented preprocessing, and a small number of random seeds where feasible.

Feasibility: The project is feasible within a 12–14-week period for a two-person team if its scope is staged. The complete Wikidata5M 
dataset, extensive hyperparameter searches, multiple complex architectures, and a custom counterfactual fairness objective would be too 
ambitious as simultaneous requirements, particularly when working with Google Colab and limited computing resources. The minimum viable 
project will therefore focus on a manageable dataset subsection, one primary baseline, a careful bias and degree analysis, and a 
reproducible evaluation. More complex models, additional demographic comparisons, and counterfactual interventions will be pursued only 
if the core experiments are completed successfully.

Declaration on Generative AI: During the preparation of this work, the authors used Claude (Anthropic) and ChatGPT (OpenAI) to assist 
with drafting, editing, and organizing research materials, and Grammarly to support language and grammar revision. After using these tools,
the authors reviewed and edited the content as needed and take full responsibility for the publication’s content.
