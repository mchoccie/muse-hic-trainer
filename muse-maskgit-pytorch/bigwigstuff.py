import pyBigWig

bw = pyBigWig.open("/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/wgEncodeCrgMapabilityAlign36mer.bw")
with open("umap_k36_hg19.bedGraph", "w") as out:
    for chrom in bw.chroms().keys():
        intervals = bw.intervals(chrom)
        if intervals:
            for start, end, value in intervals:
                out.write(f"{chrom}\t{start}\t{end}\t{value}\n")
bw.close()