# Set WD
setwd("C:/Users/thirt/Documents/Quon/Rotation_project")

#Load libraries
library(gplots)

# variables
min_pos_phenotypes = 20
num_clusters = 20
species = "mouse"

# Read in gene_attribute_matrices
if(species == "human"){
  pheno_data <- as.matrix(read.delim("output/human_gene_attribute_matrix.txt", row.names = 1)[-(1:2),-(1:2)])}

if(species == "mouse"){
  pheno_data <- as.matrix(read.delim("output/mouse_gene_attribute_matrix.txt", row.names =  1))}

# Mouse filter for some positives
pheno_data <- pheno_data[, colSums(pheno_data) >= min_pos_phenotypes]

# # Mouse clustering
# row_dend <- hclust(dist(pheno_data))
# col_dend <- hclust(dist(t(pheno_data)))
# Mouse clustering with correlation distance
row_cor <- cor(t(pheno_data), method = "pearson")
row_cor[is.na(row_cor)] <- 0
row_dend <- hclust(as.dist(1 - row_cor))

# Column correlations
col_cor <- cor(pheno_data, method = "pearson")
col_cor[is.na(col_cor)] <- 0
col_dend <- hclust(as.dist(1 - col_cor))

# --- collapse rows ---
row_clusters <- cutree(row_dend, k = num_clusters)
collapsed_rows <- rowsum(pheno_data, group = row_clusters) / as.vector(table(row_clusters))

# --- collapse columns on the row-collapsed matrix ---
col_clusters <- cutree(col_dend, k = num_clusters)
collapsed_matrix <- t(rowsum(t(collapsed_rows), group = col_clusters) / as.vector(table(col_clusters)))

# # --- cluster collapsed matrix ---
# row_dend <- hclust(dist(collapsed_matrix), method = "ward.D2")
# col_dend <- hclust(dist(t(collapsed_matrix)), method = "ward.D2")
# --- cluster collapsed matrix with correlation distance ---
row_dend <- hclust(as.dist(1 - cor(t(collapsed_matrix), method = "pearson")),
                   method = "ward.D2")

col_dend <- hclust(as.dist(1 - cor(collapsed_matrix, method = "pearson")),
                   method = "ward.D2")


# --- plot ---
pdf(paste0("collapsed_", species, "_", num_clusters, "_clusters_heatmap.pdf"), width = 12, height = 12)
heatmap.2(as.matrix(collapsed_matrix),
          Rowv = as.dendrogram(row_dend),
          Colv = as.dendrogram(col_dend),
          scale = "none",
          col = colorRampPalette(c("blue", "white", "pink", "red", "darkred"))(100),
          trace = "none", density.info = "none", key = TRUE,
          margins = c(10, 10),
          xlab = "Phenotype Clusters",
          ylab = "Gene Clusters")
dev.off()
