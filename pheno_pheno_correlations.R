library(gplots)

min_pheno_hits = 20
num_clusters = 100
species = "mouse"

if(species == "human"){
  gene_pheno_table <- as.matrix(read.delim(
    "/Users/adina/Desktop/matching_genes_phenotype_vector_human.txt",
    row.names = 1)[, -(1:2)])}

if(species == "mouse"){
  gene_pheno_table <- as.matrix(read.delim(
    "/Users/adina/Desktop/matching_genes_phenotype_vector_mouse.txt",
    row.names = 1)[, -(1:4)])}

gene_names <- rownames(gene_pheno_table)
gene_pheno_table <- apply(gene_pheno_table, 2, as.numeric)
rownames(gene_pheno_table) <- gene_names

gene_pheno_table <- gene_pheno_table[, colSums(gene_pheno_table, na.rm = TRUE) >= min_pheno_hits, drop = FALSE]

only_varying <- apply(gene_pheno_table, 2, function(x) var(x, na.rm = TRUE) > 0)
gene_pheno_table <- gene_pheno_table[, only_varying, drop = FALSE]

correlations  <- cor(gene_pheno_table, use = "pairwise.complete.obs", method = "pearson")
hc <- hclust(as.dist(1 - correlations), method = "ward.D2")

pdf(paste0("/Users/adina/Desktop/", species, "_heatmap.pdf"), width = 12, height = 12)

heatmap.2(correlations,
          Rowv = as.dendrogram(hc),
          Colv = as.dendrogram(hc),
          revC = TRUE,
          col = colorRampPalette(c("blue", "white", "red"))(100),
          trace = "none", density.info = "none",
          key = TRUE)
dev.off()

clusters <- cutree(hc, k = num_clusters)
cluster_means <- sapply(
  split(seq_len(ncol(gene_pheno_table)), clusters),
  function(idx) rowMeans(gene_pheno_table[, idx, drop = FALSE], na.rm = TRUE)
)

cluster_correlations  <- cor(cluster_means, use = "pairwise.complete.obs", method = "pearson")
cluster_hc <- hclust(as.dist(1 - cluster_correlations), method = "ward.D2")

pdf(paste0("/Users/adina/Desktop/", species, "_", num_clusters, "_clusters_heatmap.pdf"), width = 12, height = 12)

heatmap.2(cluster_correlations,
          Rowv = as.dendrogram(cluster_hc),
          Colv = as.dendrogram(cluster_hc),
          revC = TRUE,
          col = colorRampPalette(c("blue", "white", "red"))(100),
          trace = "none", density.info = "none",
          key = TRUE)
dev.off()




cut <- cutree(hc, k = 2)

cluster_sizes <- table(cut)
smaller_cluster <- which.min(cluster_sizes)
subset_pheno <- names(cut[cut == smaller_cluster])
subset_table <- gene_pheno_table[, subset_pheno, drop = FALSE]

larger_cluster <- which.max(cluster_sizes)
larger_subset_pheno <- names(cut[cut == larger_cluster])
larger_subset_table <- gene_pheno_table[, larger_subset_pheno, drop = FALSE]

out_txt <- paste0("/Users/adina/Desktop/first_subset_matching_genes_phenotype_vector_", species, ".txt")
write.table(subset_table, file = out_txt, sep = "\t", quote = FALSE, col.names = NA)

sub_cor <- cor(subset_table, use = "pairwise.complete.obs", method = "pearson")
sub_hc <- hclust(as.dist(1 - sub_cor), method = "ward.D2")

larger_sub_cor <- cor(larger_subset_table, use = "pairwise.complete.obs", method = "pearson")
larger_sub_hc <- hclust(as.dist(1 - larger_sub_cor), method = "ward.D2")

pdf(paste0("/Users/adina/Desktop/first_subset_matching_genes_phenotype_vector_", species, ".pdf"), width = 10, height = 10)
heatmap.2(sub_cor,
          Rowv = as.dendrogram(sub_hc),
          Colv = as.dendrogram(sub_hc),
          revC = TRUE,
          col = colorRampPalette(c("blue", "white", "red"))(100),
          trace = "none", density.info = "none",
          key = TRUE)
dev.off()




cut <- cutree(larger_sub_hc, k = 2)

cluster_sizes <- table(cut)
smaller_cluster <- which.min(cluster_sizes)
subset_pheno <- names(cut[cut == smaller_cluster])
subset_table <- gene_pheno_table[, subset_pheno, drop = FALSE]

out_txt <- paste0("/Users/adina/Desktop/second_subset_matching_genes_phenotype_vector_", species, ".txt")
write.table(subset_table, file = out_txt, sep = "\t", quote = FALSE, col.names = NA)

sub_cor <- cor(subset_table, use = "pairwise.complete.obs", method = "pearson")
sub_hc <- hclust(as.dist(1 - sub_cor), method = "ward.D2")

pdf(paste0("/Users/adina/Desktop/second_subset_matching_genes_phenotype_vector_", species, ".pdf"), width = 10, height = 10)
heatmap.2(sub_cor,
          Rowv = as.dendrogram(sub_hc),
          Colv = as.dendrogram(sub_hc),
          revC = TRUE,
          col = colorRampPalette(c("blue", "white", "red"))(100),
          trace = "none", density.info = "none",
          key = TRUE)
dev.off()
