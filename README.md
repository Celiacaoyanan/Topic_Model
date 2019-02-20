# Topic_Model

Topic modeling is a type of statistical model to discover topics of documents.

- Doc 1: 100% Topic A

- Doc 2: 60% Topic A-->p1, 40% Topic B

- Topic A: 30% apple-->p2, 15% bananas, 10% breakfast  

- Topic B: 20% dog, 20% cat, 20% cute 

LDA assumes that a document is produced like this: 

1. choose a topic mixture for the document (60% Topic A, 40% Topic B). 

2. in order to generate each word in the document, we first picking a topic (60% Topic A)

3. use the topic to generate the word (30% apple)

4. the overall probability that we choose apple finally is 30%*60%


So, finding the topics is the reversed steps.Suppose you have a set of documents. You’ve chosen some K topics, and want to use LDA to learn the topic of every document and the words associated to every topic. 

1. Go through each document, and randomly assign each word to one of the K topics.

2. This random assignment gives both topic of all the documents and word associated to every topic.

3. Improve them

   - For each document d
     - Go through each word w in d
       - For each topic t, compute: 1) p(topic t | document d) = (# words that are currently assigned to t in d/ total # words in d)   2) p(word w | topic t) = ( # assignments to t that come from w in all documents / total # assignments to t in all documents). p1*p2, this is exactly the probability that topic t generated word w, so reassign w a new topic. In this step, we’re assuming that all topic assignments except for the current word in question are correct.

4. Repeat, reach a roughly steady state where assignments are pretty good. 



TM_lda.py is used to generate lda model, TM_vocabulary.py is used to process corpus

    python TM_lda.py -f directory_of_corpus -k number_of_topics -alpha hyperparameter -beta hyperparameter -i number_of_iteration

Get three models under different parameters

model1：k=10, alpha=0.5, beta=0.01, i=1000

model2：k=10, alpha=50/k=5, beta=0.01, i=1500

model3：k=10, alpha=0.2, beta=0.01, i=1500
