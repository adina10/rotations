library(biomaRt)
library(Biostrings)

genes <- read.delim('/Users/adina/Desktop/genotype_phenotype_vector_human.txt')[-(1:2),-(2)]
proteins <- readAAStringSet("/Users/adina/Desktop/longest_per_gene_human.fa")

mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
mapping <- getBM(attributes = c("entrezgene_id",
                "ensembl_gene_id"),
                 filters = "entrezgene_id",
                 values = list(genes$Phenotype),
                 mart = mart)
mapping <- mapping[!mapping$entrezgene_id %in% mapping$entrezgene_id[
  duplicated(mapping$entrezgene_id) | duplicated(mapping$entrezgene_id, fromLast = TRUE)
], ]

genes_merged <- merge(genes, mapping, by.x = "Phenotype", by.y = "entrezgene_id", all.x = FALSE)
genes_merged <- genes_merged[, c("ensembl_gene_id", setdiff(names(genes_merged), "ensembl_gene_id"))]
colnames(genes_merged)[1:3] <- c("Ensembl_ID", "Entrez_ID", "Gene Symbol")

write.table(
  genes_merged,
  "/Users/adina/Desktop/matching_genes_phenotype_vector_human.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

header_ids <- strsplit(names(proteins), "\\|")
gene_id  <- sub("\\..*", "", sapply(header_ids, `[`, 3))
subset_to_write <- proteins[gene_id %in% genes_merged$Ensembl_ID]

writeXStringSet(subset_to_write, "/Users/adina/Desktop/longest_per_gene_human__matching_subset.fa")

subset <- readAAStringSet("/Users/adina/Desktop/longest_per_gene_human__matching_subset.fa")
