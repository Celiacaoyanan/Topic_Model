# Topic_Model
TM_lda.py is used to generate lda model, TM_vocabulary.py is used to process corpus

    python TM_lda.py -f directory_of_corpus -k number_of_topics -alpha hyperparameter -beta hyperparameter -i number_of_iteration

Get three models under different parameters

model1：k=10, alpha=0.5, beta=0.01, i=1000

model2：k=10, alpha=50/k=5, beta=0.01, i=1500

model3：k=10, alpha=0.2, beta=0.01, i=1500
