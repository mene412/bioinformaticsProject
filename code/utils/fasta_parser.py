import os
from Bio import SeqIO # type: ignore
import gzip
import random

def read_fasta_samples(folder_path):
    """
    Reads all FASTA files from a folder and returns their sequences.
    
    Parameters:
    - folder_path: path to folder containing .fasta files
    
    Returns:
    - samples: dictionary with filename as key and set of sequences as value
    """
    samples = {}
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return samples
    
    # Get all .fasta files in the folder
    fasta_files = [f for f in os.listdir(folder_path) if f.endswith(".fasta")]
    
    if not fasta_files:
        print(f"No .fasta files found in {folder_path}")
        return samples
    
    print(f"Found {len(fasta_files)} FASTA files")
    
    for filename in fasta_files:
        file_path = os.path.join(folder_path, filename)
        print(f"Reading: {file_path}")
        
        # Use filename (without extension) as sample ID
        sample_id = os.path.splitext(filename)[0]
        
        try:
            with open(file_path, "rt") as handle:
                # Fixed: using "fasta" parser for .fasta files
                sequences = set(str(record.seq) for record in SeqIO.parse(handle, "fasta"))
                
                if sequences:
                    samples[sample_id] = sequences
                    print(f"  -> Found {len(sequences)} unique sequences for sample '{sample_id}'")
                else:
                    print(f"  -> Warning: No sequences found in {filename}")
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Successfully read {len(samples)} files")
    return samples

def read_fasta_samples_sampled(fasta_folder, sample_size_mb=10, avg_bytes_per_read=250):
    samples = {}
    reads_to_sample = int((sample_size_mb * 1024 * 1024) / avg_bytes_per_read)

    for subfolder in os.listdir(fasta_folder):
        subfolder_path = os.path.join(fasta_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        reads_path = os.path.join(subfolder_path, "reads", "anonymous_reads.fq.gz")
        if not os.path.exists(reads_path):
            print(f"File non trovato: {reads_path}")
            continue

        sample_id = subfolder
        reservoir = []

        try:
            with gzip.open(reads_path, "rt") as handle:
                print('aperto: ', handle)
                for i, record in enumerate(SeqIO.parse(handle, "fastq")):
                    if i < reads_to_sample:
                        reservoir.append(str(record.seq))
                    else:
                        j = random.randint(0, i)
                        if j < reads_to_sample:
                            reservoir[j] = str(record.seq)

        except EOFError:
            print(f"⚠️ Errore: il file {reads_path} è compresso male o troncato. Campionamento parziale salvato.")
        except Exception as e:
            print(f"⚠️ Errore inaspettato su {reads_path}: {e}")
        finally:
            # Salva i dati raccolti finora, anche se c'è stato un errore
            if reservoir:
                samples[sample_id] = set(reservoir)
                print(f"Sampled {len(reservoir)} reads from {reads_path} (~{sample_size_mb} MB)")
            else:
                print(f"Nessuna lettura campionata da {reads_path}.")

    print('Samples fatti tutti')
    return samples

def save_samples_to_fasta_files(samples, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for sample_id, seqs in samples.items():
        output_path = os.path.join(output_folder, f"{sample_id}.fasta")
        with open(output_path, "w") as f:
            for i, seq in enumerate(seqs):
                f.write(f">{sample_id}_read_{i}\n{seq}\n")
        print(f"Saved {len(seqs)} sequences to {output_path}")