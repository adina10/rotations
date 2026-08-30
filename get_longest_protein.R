library(seqinr)

f <- "/Users/adina/Desktop/gencode.v49.pc_translations.fa"
seqs <- read.fasta(file = f, seqtype = "AA")

headers <- names(seqs)

extract_gene_id <- function(h) {
  parts <- strsplit(h, "\\|")[[1]]
  return(parts[3])
}

gene_id <- vapply(headers, extract_gene_id, character(1))

seq_len <- vapply(seqs, length, integer(1))

o <- order(gene_id, -seq_len)
keep_idx <- o[!duplicated(gene_id[o])]

seqs_longest <- seqs[keep_idx]

write.fasta(
  sequences = seqs_longest,
  names     = headers[keep_idx],
  file.out  = "/Users/adina/Desktop/longest_per_gene_human.fa",
)

o <- "/Users/adina/Desktop/longest_per_gene_human.fa"
seqs_updated_mouse <- read.fasta(file = o, seqtype = "AA")