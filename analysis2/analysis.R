library(renv)
renv::load("~/Study/UTEC/2025-1/adv_bioinf") # se usa para cargar librerías, no es necesario si se instala seqinr
library(seqinr)

aaseq<-read.fasta("aa17_explore.fasta" ,seqtype = "AA")
aaseq<-aaseq[[1]]
AAstat(aaseq)
