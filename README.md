# Code and Data corresponding to [this manuscript](https://doi.org/10.1101/2020.10.25.353375)

Statistical learning occurs during practice while high-order rule learning during rest period
================
Romain Quentin, Lison Fanuel, Mariann Kiss, Marine Vernet, Teodóra Vékony, Karolina Janacsek, Leonardo Cohen, Dezso Nemeth


Abstract
========
Knowing when the brain learns is crucial for both the comprehension of memory formation and consolidation, and for developing new training and neurorehabilitation strategies in healthy and patient populations. Recently, a rapid form of offline learning developing during short rest periods has been shown to account for most of procedural learning, leading to the hypothesis that the brain mainly learns during rest between practice periods. Nonetheless, procedural learning has several subcomponents not disentangled in previous studies investigating learning dynamics, such as acquiring the statistical regularities of the task, or else the high-order rules that regulate its organization. Here, we analyzed 506 behavioral sessions of implicit visuomotor deterministic and probabilistic sequence learning tasks, allowing the distinction between general skill learning, statistical learning and high-order rule learning. Our results show that the temporal dynamics of apparently simultaneous learning processes differ. While general skill and high-order rule learning are acquired offline, statistical learning is evidenced online. These findings open new avenues on the short-scale temporal dynamics of learning and memory consolidation and reveal a fundamental distinction between statistical and high-order rule learning, the former benefiting from online evidence accumulation and the latter requiring short rest periods for rapid consolidation.


Data
====

Data are publicly accessible in the data folder. All three datasets analyzed in the manuscript are available in .sav format.

Scripts
=======

Overall, the current scripts remain designed for research purposes, and could therefore be improved and clarified. If you judge that some codes would benefit from specific clarifications do not hesitate to contact us.

#### Config files
- 'base.py' # where all generic functions are defined

#### Analyses files
- 'SRTT_analyses.py'  # Analysing and plotting of the 1st dataset (SRTT)
- 'ASRT_analyses.py'  # Analysing and plotting of the 2nd dataset (ASRT)
- 'longASRT_analyses.py'  # Analysing and plotting of the 3rd dataset (long ASRT)


Dependencies
============
- Python 3.8.3
- Numpy: 1.18.5
- pingouin: 0.3.8
- pandas: 1.0.5
- seaborn: 0.10.1
- matplotlib: 3.2.2
- scipy: 1.5.0
