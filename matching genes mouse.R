library(biomaRt)
library(Biostrings)
library(homologene)

genes <- read.delim('/Users/adina/Desktop/genotype_phenotype_vector_mouse.txt')
proteins <- readAAStringSet("/Users/adina/Desktop/longest_per_gene_mouse.fa")
reference <- read.delim('/Users/adina/Desktop/mouse_reference.txt')[,-c(1)]

genes_add_human_entrez <- merge(genes, reference, by.x = "Gene", by.y = "Gene", all.x = FALSE)
genes_add_human_entrez <- genes_add_human_entrez[, c("Gene.ID", setdiff(names(genes_add_human_entrez), "Gene.ID"))]
colnames(genes_add_human_entrez)[1:2] <- c("Human_Entrez_ID", "Gene_Symbol")

data("homologeneData", package = "homologene")

human_HIDs <- subset(homologeneData, Taxonomy == 9606, select = c(HID, Gene.ID))
genes_add_HID <- merge(genes_add_human_entrez, human_HIDs, by.x = "Human_Entrez_ID", by.y = "Gene.ID", all.x = FALSE)
genes_add_HID <- genes_add_HID[, c("HID", setdiff(names(genes_add_HID), "HID"))]

mouse_HIDs <- subset(homologeneData, Taxonomy == 10090, select = c(HID, Gene.ID))
genes_add_mouse_entrez <- merge(genes_add_HID, mouse_HIDs, by.x = "HID", by.y = "HID", all.x = FALSE)
genes_add_mouse_entrez <- genes_add_mouse_entrez[, c("Gene.ID", setdiff(names(genes_add_HID), "Gene.ID"))]
colnames(genes_add_mouse_entrez)[1] <- "Mouse_Entrez_ID"

mart <- useMart("ensembl", dataset = "mmusculus_gene_ensembl")
mapping <- getBM(attributes = c("entrezgene_id",
                                "ensembl_gene_id"),
                 filters = "entrezgene_id",
                 values = list(genes_add_mouse_entrez$Mouse_Entrez_ID),
                 mart = mart)
mapping <- mapping[!mapping$entrezgene_id %in% mapping$entrezgene_id[
  duplicated(mapping$entrezgene_id) | duplicated(mapping$entrezgene_id, fromLast = TRUE)
], ]

genes_add_ensembl <- merge(genes_add_mouse_entrez, mapping, by.x = "Mouse_Entrez_ID", by.y = "entrezgene_id", all.x = FALSE)
genes_add_ensembl <- genes_add_ensembl[, c("ensembl_gene_id", setdiff(names(genes_add_ensembl), "ensembl_gene_id"))]
colnames(genes_add_ensembl)[1] <- "Ensembl_ID"

genes_add_ensembl <- genes_add_ensembl[!duplicated(genes_add_ensembl$Ensembl_ID) & !duplicated(genes_add_ensembl$Ensembl_ID, fromLast = TRUE), ]

write.table(
  genes_add_ensembl,
  "/Users/adina/Desktop/matching_genes_phenotype_vector_mouse.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

header_ids <- strsplit(names(proteins), "\\|")
gene_id  <- sub("\\..*", "", sapply(header_ids, `[`, 3))
subset_to_write <- proteins[gene_id %in% genes_add_ensembl$Ensembl_ID]

writeXStringSet(subset_to_write, "/Users/adina/Desktop/longest_per_gene_mouse_matching_subset.fa")

subset <- readAAStringSet("/Users/adina/Desktop/longest_per_gene_mouse_matching_subset.fa")