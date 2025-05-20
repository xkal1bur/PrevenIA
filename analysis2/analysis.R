library(renv)
renv::load("~/Study/UTEC/2025-1/adv_bioinf")
library(seqinr)

aaseq<-read.fasta("aa17_explore.fasta" ,seqtype = "AA")
aaseq<-aaseq[[1]]
AAstat(aaseq)
