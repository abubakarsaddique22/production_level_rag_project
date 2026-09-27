🎉 Baseline mil gaya! Ye kaafi acche numbers hain start ke liye:

Metric	Value	Matlab
hit@5	0.933 (93.3%)	28 mein se 30 sawal ka sahi chunk top-5 mein aaya
MRR	0.801	Jab mila, zyada tar 1st ya 2nd position pe mila (kaafi high rank)
recall@5	0.9 (90%)	Multi-chunk sawalon ke 90% relevant chunks mil gaye

Ye industry-acceptable baseline hai — blueprint ka target tha hit@5 ≥ 0.90, aap already usse upar hain bina hybrid search ya reranking ke.

hit_at_k()          # plain Python: kya relevant_id retrieved list mein hai?
reciprocal_rank()   # plain Python: kitni rank pe mila?
recall_at_k()       # plain Python: kitne % relevant chunks mile?



----------------------hybrid 

🎉 Zabardast improvement! Ye numbers hybrid search ka clear, measurable fayda dikhate hain:

Metric	Dense (baseline)	Hybrid	Improvement
hit@5	0.933	0.967	+3.3%
MRR	0.801	0.858	+5.7%
recall@5	0.90	0.95	+5.0%
Failures	2 (q004, q019)	1 (sirf q019)	q004 fix ho gaya ✅

Ye bilkul wahi result hai jo blueprint predict karta hai: "Add hybrid search and RRF... Compare against the baseline" — aapke resume ke liye ye ek real, measured claim ban gaya: "Hybrid search (BM25 + RRF) improved hit@5 from 0.93 to 0.97 and MRR by 5.7 points."







-------------------------- with rerank

🎉🎉 hit@5 = 1.0 (100%)! Ye ek bohot bada achievement hai — teeno stages milkar poore 30 sawalon ka sahi jawab top-5 mein la rahe hain, koi failure baaki nahi.

Final comparison table
Stage	hit@5	MRR	recall@5	Fixed
Dense only (baseline)	0.933	0.801	0.900	—
+ Hybrid (BM25+RRF)	0.967 (+3.3%)	0.858 (+5.7%)	0.950 (+5.0%)	q004
+ Rerank (cross-encoder)	1.000 (+3.3%)	0.889 (+3.1%)	0.983 (+3.3%)	q019

Blueprint ke targets (hit@5 ≥ 0.90, MRR ≥ 0.75) dono cross ho gaye, aur ab perfect score hai. Ye exactly wo interview-ready claim hai jo blueprint chahta tha: "Retrieval hybrid search + reranking se baseline se X% improve hua, measured on a 30-question golden set."