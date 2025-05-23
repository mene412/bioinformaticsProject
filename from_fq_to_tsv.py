from Bio import SeqIO
import sys


print(sys.argv[1])
print(sys.argv[2])
fq_file = sys.argv[1]
tsv_file = sys.argv[2]

with open(tsv_file, "w") as out:
    for record in SeqIO.parse(fq_file, "fastq"):
        qualities = sequence_str = str(record.seq)
        out.write(qualities + "\n")