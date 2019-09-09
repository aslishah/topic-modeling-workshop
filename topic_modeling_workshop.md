# A Workshop in Topic Modeling






## Overview

* [Introduction](#introduction)
* [What is topic modeling?](#what-is-topic-modeling)
* [Dataset preparation](#dataset-preparation)
* [DARIAH Topic Explorer](#dariah-topic-explorer)
* [R and the package Mallet](#r-and-the-package-mallet)
* [Online tutorials](#online-tutorials)
* [Bibliography](#bibliography)
* [References](#references)







## Introduction

This tutorial provides a very concise introduction to topic modeling. It is assumed that the user exhibits some familiarity with gneral concepts behind topic models. It might be profitable to consult the resources listed at the bottom of this document, in the section [Online tutorials](#online-tutorials). Also, the tutorial itself is supplemented by [this presentation](https://computationalstylistics.github.io/topic-modeling-workshop/presentation/intro_to_topic_models.html), in which possible applications of topic modeling are discussed, and a few introductory remarks about the method itself are provided.

The tutorial will cover basic functionalities of the [DARIAH topic explorer](https://dariah-de.github.io/TopicsExplorer/) software. As a dummy dataset, the Shakesperean canon will be used in the form of [raw text files](https://github.com/computationalstylistics/topic-modeling-workshop/tree/master/shakespeare_genre/corpus). In the second part, the same dataset will be used to train a topic model using the programming language R and the package [R mallet](https://cran.r-project.org/web/packages/mallet/index.html).






## What is topic modeling?

Topic modeling is a machine-learning technique aimed at discovering hidden thematic structures in large collections of texts. What makes this technique special, is that it belongs to distributional semantics paradigm – this means that the method doesn’t rely on any prior knowledge about word meanings, to discover semantic relations between groups of words referred to as _topics_.

Topic modeling takes advantage of the simple assumption that certain words tend to occur more frequently in a text covering a given topic than in other texts. Next, texts are usually about many topics. Consequently, a topic is a recurring pattern of co-occurring words. To quote David Blei (2012: 78), one of the authors of the topic modeling technique:

> We formally define a _topic_ to be a distribution over a fixed vocabulary. For example, the _genetics_ topic has words about genetics with high probability and the evolutionary biology topic has words about _evolutionary biology_ with high probability. 

More specifically, each topic in a corpus is a distribution over words, each text is a mixture of corpus-wide topics, and each word is drawn from one of those topics. In the real world, however, we only observe the texts and the words occurring in the texts; all the other characteristics of the texts are hidden variables, to be inferred by a generative model.

There are a few methods to perform topic modeling, but the most popular one is LDA, or latent Dirichlet allocation (Blei, 2003). The LDA method has several implementations; the following ones are relatively simple to use, and thus recommended for humanities scholars: 

* [Mallet](http://mallet.cs.umass.edu/) (Java)
* [Stanford Topic Modeling Toolbox](https://nlp.stanford.edu/software/tmt/tmt-0.4/) (Java)
* [gensim](https://radimrehurek.com/gensim/) (Python)
* [lda](https://github.com/lda-project/lda) (Python)
* [topicmodels](https://cran.r-project.org/web/packages/topicmodels/index.html) (R)
* [Mallet invoked from R](https://cran.r-project.org/web/packages/mallet/index.html) (R + Java)
* [**DARIAH Topic Explorer**](https://dariah-de.github.io/TopicsExplorer/) (standalone)






## Dataset preparation

Topic modeling is designed to analyze large collections of texts (documents). Leaving aside the question how large such a collection should be, we’ll focus on a dummmy corpus containing the works by Shakespeare. 

- in the GitHub repository

What is important to know before a topic modeling algorithm is applied, is that **the order of words** is not relevant for the method. A text sample becomes a “bag of words”, in which only word frequencies matter. Essentially, this means that the relation between any adjacent words in _Hamlet_ by Shakespeare 

is considered to be one text sample, 

* The order of documents is not relevant
* The number of topics is fixed and known in advance

---



- bag of words

- works automatically without any prior knowledge about word meanings or grammar.

- text chunking
- optimal number of topics



1. Przygotowanie tekstów:
   - usunięcie interpunkcji

   - usunięcie wielkich liter

   - wykluczenie wyrazów synsemantycznych

     Lepsze rezultaty osiąga się poprzez usunięcie wyrazów na podstawie wartości `idf`. Różnica ta wzrasta wraz ze wzrostem liczby tematów. (Schofield et al. 2017)

2.  Określenie parametrów modelu.

   - większa liczba tematów jest trudniej interpretowalna przez ludzi (Chang 2009)



   

## DARIAH Topic Explorer

- [DARIAH topic explorer](https://dariah-de.github.io/TopicsExplorer/). 
- 
- download the [executable file](https://github.com/DARIAH-DE/TopicsExplorer/releases/tag/v2.0) matching the operating system.

![DARIAH Topics Explorer](https://raw.githubusercontent.com/DARIAH-DE/TopicsExplorer/master/docs/img/application-screenshot.png)





![document-topic distribution](https://raw.githubusercontent.com/DARIAH-DE/TopicsExplorer/master/docs/img/document-topic-distributions.png)





- launch the program
- choose the text files to be scrutinized by the LDA algorithm
- in our case, it will be the texts by Shakespeare to be found in the subdirectory `corpus`
- choose 100 most frequent words



... then experiment with other stopword lists



An excerpt from _The Merchant of Venice_:



```
ACT 1.
SCENE I. Venice. A street
[Enter ANTONIO, SALARINO, and SALANIO]
ANTONIO.
In sooth, I know not why I am so sad;
It wearies me; you say it wearies you;
But how I caught it, found it, or came by it,
What stuff 'tis made of, whereof it is born,
I am to learn;
And such a want-wit sadness makes of me
That I have much ado to know myself.
SALARINO.
Your mind is tossing on the ocean;
There where your argosies, with portly sail--
Like signiors and rich burghers on the flood,
Or as it were the pageants of the sea--
Do overpeer the petty traffickers,
That curtsy to them, do them reverence,
As they fly by them with their woven wings.
```





`ANTONIO` and `SALARINO`



A fragment from _Hamlet_ shows a different way of indicating speakers:

```
SCENE. Elsinore.
ACT I.
Scene I. Elsinore. A platform before the Castle.
[Francisco at his post. Enter to him Bernardo.]
Ber.
Who's there?
Fran.
Nay, answer me: stand, and unfold yourself.
Ber.
Long live the king!
Fran.
Bernardo?
Ber.
He.
Fran.
You come most carefully upon your hour.
Ber.
'Tis now struck twelve. Get thee to bed, Francisco.
Fran.
For this relief much thanks: 'tis bitter cold,
And I am sick at heart.
​
```




Whatever is the case, the names should be excluded from the analysis



A very rough list of proper names here: ......

Same about words such as _enter_ or _exeunt_





(there exist more sophisticated ways of excluding stop words, such as tf/idf weighting)





## R and the package Mallet








``` R
# first, some variables should be set

# number of topics to be inferred
no.of.topics = 25
# number of iterations (epochs) in the training stage
train_iterations = 200
# a directory containing text files
directory = "corpus"
# slicing the texts into samples of N words
sample_size = 1000
# the file containing stopwords
stopwords = "combined_stopwords.txt"


# when the corpus is big, the default memory allocation might be not enough;
# the following option increases the Java memory to 4Gb
options(java.parameters = "-Xmx4g")

# invoking relevant R libraries
library(stylo)
library(mallet)
library(wordcloud)


# loading the corpus as is
raw.corpus = load.corpus(files = dir(), corpus.dir = directory)
# splitting the texts into words
parsed.corpus = parse.corpus(raw.corpus)
# splitting the texts into equal-sized samples
sliced.corpus = make.samples(parsed.corpus, sampling = "normal.sampling", sample.size = sample_size)
# since Mallet prefers to split the texts iteself, joining the words into space-delimited strings
deparsed.corpus = sapply(sliced.corpus, paste, collapse = " ")



# invoking Mallet: first importing the texts
mallet.instances = mallet.import(id.array = names(deparsed.corpus), 
         text.array = deparsed.corpus, stoplist.file = stopwords,
         token.regexp = "[A-Za-z]+")

# create a topic trainer object
topic.model = MalletLDA(num.topics = no.of.topics)

# load the texts/samples
topic.model$loadDocuments(mallet.instances)

# get the vocabulary, and some statistics about word frequencies
vocabulary = topic.model$getVocabulary()
word.freqs = mallet.word.freqs(topic.model)

# optimize hyperparameters every 20 iterations, after 50 burn-in iterations.
topic.model$setAlphaOptimization(20, 50)

# train a model, using the specified number of iterations
topic.model$train(train_iterations)

# run through a few iterations where we pick the best topic for each token,
# rather than sampling from the posterior distribution.
topic.model$maximize(10)

# Get the probability of topics in documents and the probability of words in topics.
# By default, these functions return raw word counts. Here we want probabilities,
# so we normalize, and add "smoothing" so that nothing has exactly 0 probability.
doc.topics = mallet.doc.topics(topic.model, smoothed = TRUE, normalized = TRUE)
topic.words = mallet.topic.words(topic.model, smoothed = TRUE, normalized = TRUE)

# now, add words' IDs and samples' IDs to both tables
colnames(topic.words) = vocabulary
# names of the samples
rownames(doc.topics) = names(deparsed.corpus)
# names of the topics: actually, simple numeric IDs
colnames(doc.topics) = 1:length(doc.topics[1,])
```










``` R

############## Exploration of the dataset

# to get N words from Xth topic
no.of.words = 50
topic.id = 1
current.topic = sort(topic.words[topic.id,], decreasing = T)[1:no.of.words]

# to make a wordcloud out of the most characteristic topics
wordcloud(names(current.topic), current.topic, random.order = FALSE, rot.per = 0)

```



Please keep in mind that in your case, the numbers assigned to the topics will probably be different. This is due to the fact that the LDA algorithm assigns the topics IDs randomly. Moreover, the word proportions in particular topics might differ as well, due to the random seeding of the word proportions at the first interation.



``` R
no.of.words = 50
for(i in 1 : no.of.topics) {
    topic.id = i
    current.topic = sort(topic.words[topic.id,], decreasing = T)[1:no.of.words]
    png(file = paste("topic_", topic.id, ".png", sep=""))
    wordcloud(names(current.topic), current.topic, random.order = FALSE, rot.per = 0)
    dev.off()
}


```


Caveat: the files will be saved in your current directory without checking what’s inside. It can clutter your hard-drive and/or overwrite some exsting files!






```R
topic.words = read.csv("https://raw.githubusercontent.com/computationalstylistics/diachronia/master/dane/abo_albo.csv", header = TRUE, row.names = 1)
```



`rownames(doc.topics)`



plot a sample from _Hamlet_ (sample #23, which is sample #750)

the ending of the King Lear (808)

the climax of the _Romeo and Juliet_ (880)


``` R

# to plot the proportions of topics in the Xth sample
no.of.sample = 880
plot(doc.topics[no.of.sample,], type = "h", xlab = "topic ID", ylab = "probability", ylim = c(0, 0.5), main = rownames(doc.topics)[no.of.sample], lwd = 5, col = "green")

```



What about the beginning of _The Tempest_?



```R
# to plot the proportions of topics in the Xth sample
no.of.sample = 626
plot(doc.topics[no.of.sample,], type = "h", xlab = "topic ID", ylab = "probability", ylim = c(0, 0.5), main = rownames(doc.topics)[no.of.sample], lwd = 5, col = "blue")

```





``` R

stylo(frequencies = doc.topics, gui = FALSE, dendrogram.layout.horizontal = FALSE)

stylo(frequencies = doc.topics, gui = FALSE, analysis.type = "PCR")

stylo(frequencies = doc.topics, gui = FALSE, analysis.type = "PCR", text.id.on.graphs = "points")
```








## Online tutorials

* [Oh A. (2010). Topic Models Applied to Online News and Reviews](https://www.youtube.com/watch?v=1wcX4fEdNUo) (YouTube)
* [Mimno (2012), Topic Modeling Workshop](https://vimeo.com/53080123) (Vimeo)
* [Jockers, Nelson (2012), Topic Modeling Workshop](https://vimeo.com/52959139) (Vimeo)
* [Guldi, Johnson-Roberson (2012), Topic Modeling Workshop](https://vimeo.com/53078693) (Vimeo)
* [Brett M. (2012), Topic Modeling: A Basic Introduction](http://journalofdigitalhumanities.org/2-1/topic-modeling-a-basic-introduction-by-megan-r-brett/)
* [Jockers, The LDA Buffet is Now Open](http://www.matthewjockers.net/2011/09/29/the-lda-buffet-is-now-open-or-latent-dirichlet-allocation-for-english-majors/) (LDA explained in a form of a literary fable)
* [Underwood, Topic modeling made just simple enough](https://tedunderwood.com/2012/04/07/topic-modeling-made-just-simple-enough/)
* [Topic Modeling in the Humanities](https://mith.umd.edu/topic-modeling-in-the-humanities-an-overview/)
* [Scott Weingart’s posts on Topic Modeling](http://scottbot.net/tag/topic-modeling/)
* [David Blei’s webpage](http://www.cs.columbia.edu/~blei/topicmodeling.html)
* [Text Mining with R](https://www.tidytextmining.com/)
* [Topic Models Learning and R Resources](https://github.com/trinker/topicmodels_learning)


## Bibliography

* [Topic Modeling Bibliography](https://mimno.infosci.cornell.edu/topics.html)



## References

**Blei, D.M.** (2012). [Probabilistic topic models](http://delivery.acm.org/10.1145/2140000/2133826/p77-blei.pdf). _Communications of the ACM_, **55**(4): 77–84.

**Blei, D.M., Ng, A.Y. and Jordan, M.I.** (2003). [Latent Dirichlet allocation](http://jmlr.csail.mit.edu/papers/v3/blei03a.html). _Journal of Machine Learning Research_. **3**: 993–1022.

