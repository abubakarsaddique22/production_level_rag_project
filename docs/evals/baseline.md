🎉 Baseline mil gaya! Ye kaafi acche numbers hain start ke liye:

Metric	Value	Matlab
hit@5	0.933 (93.3%)	28 mein se 30 sawal ka sahi chunk top-5 mein aaya
MRR	0.801	Jab mila, zyada tar 1st ya 2nd position pe mila (kaafi high rank)
recall@5	0.9 (90%)	Multi-chunk sawalon ke 90% relevant chunks mil gaye

Ye industry-acceptable baseline hai — blueprint ka target tha hit@5 ≥ 0.90, aap already usse upar hain bina hybrid search ya reranking ke.

hit_at_k()          # plain Python: kya relevant_id retrieved list mein hai?
reciprocal_rank()   # plain Python: kitni rank pe mila?
recall_at_k()       # plain Python: kitne % relevant chunks mile?