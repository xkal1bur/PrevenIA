library(renv)
renv::load("~/Study/UTEC/2025-1/adv_bioinf") # se usa para cargar librerías, no es necesario si se instala seqinr
library(seqinr)

aaseq<-read.fasta("aa17_explore.fasta" ,seqtype = "AA")[[1]]
AAstat(aaseq)


aaseq2<-read.fasta("aa13_explore.fasta" ,seqtype = "AA")[[1]]
AAstat(aaseq2)
