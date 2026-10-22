# Extract only the already selected public cells. No filtering or reference fitting.
suppressPackageStartupMessages({library(SeuratObject);library(Matrix);library(rhdf5)})
args<-commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3, !file.exists(args[3]))
sel<-read.delim(args[2],check.names=FALSE,stringsAsFactors=FALSE)
x<-readRDS(args[1]);md<-slot(x,'meta.data');assays<-slot(x,'assays')
indices<-match(sel$cell_id,rownames(md))
stopifnot(!anyNA(indices),!anyDuplicated(indices),
 identical(as.character(md$donor[indices]),sel$donor_id),
 identical(paste(md$donor[indices],md$timepoint[indices],sep='__'),sel$sample_id))
h5createFile(args[3])
for(nm in c('RNA','ADT')){
 z<-slot(assays[[nm]],'counts')
 stopifnot(identical(colnames(z),rownames(md)))
 z<-z[,indices,drop=FALSE]
 stopifnot(all(is.finite(z@x)),all(z@x>=0),!anyDuplicated(rownames(z)))
 h5createGroup(args[3],nm)
 for(k in c('p','i','x','Dim')) h5write(slot(z,k),args[3],paste0(nm,'/',k))
 h5write(rownames(z),args[3],paste0(nm,'/feature_names'))
 h5write(colnames(z),args[3],paste0(nm,'/cell_ids'))
}
h5closeAll()
cat('Selected RNA and ADT assays exported without rounding.\n')
