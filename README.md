# CS224N Project
Final project for [CS 224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/), Winter 2023.

PiGGyBacking off of PEGASUS: Pre-training with Gap-sentences for Government Bills. Alice Chen, Evelyn Choi, Karsen Wahal.

#### Abstract
Congressional bills are notoriously long and difficult to understand, limiting citizen engagement. As a result, summarizing legislation in a low-resource setting is a foundational step in making our democracy more inclusive. In this project, we propose a novel approach to bill summarization, PIGASUS, that is based off of Google's state-of-the-art summarizer, PEGASUS. PIGASUS makes government bills more accessible by creating quality summaries in low resource settings. Our model differs from PEGASUS by using a state-of-the-art extractive summarization technique, TextRank, to select masked sentences for pretraining. To experiment with this, we implement a second "pretraining" stage that builds off the pretrained PEGASUS model, mimicking the effect of pretraining a large summarization model from scratch. We find that in low-resource settings, PIGASUS is not only more efficient in learning, but also generates higher quality summaries that capture more important information than PEGASUS does in the same settings. This suggests that using TextRank to mask sentences during pretraining may improve the quality of summarization models.

The full report can be found [here](http://web.stanford.edu/class/cs224n/final-reports/final-report-169724371.pdf).
