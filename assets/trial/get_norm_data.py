import warnings
warnings.filterwarnings("ignore")
import anarci
from Bio import SeqIO
import sys 
import os
import pickle
import torch
import ablang2
import esm
import gzip
import numpy as np

class ProteinLanguageModel():
    def __init__( self, model_type: str ):
        self.model_type = model_type
        (
            self.model,
            self.alphabet,
            self.batch_converter
        ) = self.load()

    def load( self ):
        """
        Load the pre-trained ESM-1b model and its vocabulary
        Downloaded model saved at
            /Users/kartikmajila/.cache/torch/hub/checkpoints/
        """
        if self.model_type == "esm1b-650M":
            print( "Loading ESM1b-650M model. For the first time, the download will take some time..." )
            model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        elif self.model_type == "esm2-650M":
            print( "Loading ESM2-650M model. For the first time, the download will take some time..." )
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        elif self.model_type == "ablang2":
            print( "Loading AbLang2 model. For the first time, the download will take some time..." )
            model = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, ncpu=1)
            return model, None, None
        else:
            raise ValueError("Invalid model choice, available options : esm1b-650M, esm2-650M, ablang2")


        batch_converter = alphabet.get_batch_converter()
        model.eval()

        return model, alphabet, batch_converter


    def tokenize(
        self,
        label: str,
        seq: str ) -> np.ndarray:
        """
        Convert the raw sequence into tokens
            that the model understands.
        """
        data = [( label, seq )]
        batch_labels, batch_strs, batch_tokens = self.batch_converter( data )
        return batch_labels, batch_strs, batch_tokens

    def embed_seq(
        self,
        label: str,
        seq: str,
        layer = 33,
        ab_chain = "vh"
        ) -> np.ndarray:
        """
        Return the embedding for the given sequence.
        """

        if self.model_type == "ablang2":
            #prepare input
            if ab_chain == "vh":
                input_ = [[seq,'']]
            elif ab_chain == "vl":
                input_ = [['', seq]]
            else:
                raise ValueError("choose either 'vh' / 'vl' for ab_chain")

            with torch.no_grad():
                out = self.model(input_, mode='rescoding',stepwise_masking=False)

            #slicing special tokens
            if ab_chain == "vh":
                embedding = [e[1:-2] for e in out]
            elif ab_chain == "vl":
                embedding = [e[2:-1] for e in out]

            return embedding[0]


        elif self.model_type == "esm1b-650M" or self.model_name == "esm2-650M":
            batch_labels, batch_strs, batch_tokens = self.tokenize(
                label = label, seq = seq
            )
            # batch_tokens = batch_tokens.to( DEVICE )
            with torch.no_grad():
                out = self.model(
                    batch_tokens,
                    repr_layers = [layer],
                    return_contacts = False )

            # Get the representation from the specified layer
            # The tensor includes special start (<cls>) and end (<eos>) tokens.
            # [1, L+2, C]
            token_representations = out["representations"][layer]
            # We slice the tensor to extract only the residue embeddings [L x 1280]
            embedding = token_representations[0, 1 : len(seq) + 1].numpy()
            return embedding


    def get_logits(
        self,
        label: str,
        seq: str ) -> np.ndarray:
        """
        Return the predicted logits for the given sequence.
        """
        batch_labels, batch_strs, batch_tokens = self.tokenize(
            label = label, seq = seq
        )
        # batch_tokens = batch_tokens.to( DEVICE )
        with torch.no_grad():
            out = self.model(
                batch_tokens,
                repr_layers = [],
                return_contacts = False )
        # [L+2, 33]; L -> length of the input sequence.
        logits = out["logits"][0]
        return logits



def get_numbering(seq):
    """Get the numbering of a sequence using ANARCI."""
    try:
        numbering = anarci.number(seq, scheme='imgt')
        return numbering[0]
    except Exception as e:
        print(f"Error processing sequence: {e}")
        return None


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 numbering.py <input_vh_fasta> <input_vl_fasta>")
        sys.exit(1)

    input_fasta = sys.argv[1]
    if not os.path.isfile(input_fasta):
        print(f"File {input_fasta} does not exist.")
        sys.exit(1)


    esmmodel = ProteinLanguageModel(model_type="esm2-650M")
    ablang2model = ProteinLanguageModel(model_type="ablang2")

    ablang2_vh, esm2_vh = [], []
    ablang2_vl, esm2_vl = [], []

    VH_imgt_values, VL_imgt_values = [], []

    print("Processing VH sequences...")
    for record in SeqIO.parse(sys.argv[1], "fasta"):
        seq_id = str(record.id)
        seq = str(record.seq)
        numbering = get_numbering(seq)
        esm_embedding = esmmodel.embed_seq(label=seq_id, seq=seq, layer=33, ab_chain=sys.argv[2].lower())
        ablang2_embedding = ablang2model.embed_seq(label=seq_id, seq=seq, ab_chain=sys.argv[2].lower())

        ablang2_vh.append(ablang2_embedding)
        esm2_vh.append(esm_embedding)
        VH_imgt_values.append(numbering)

    print("Processing VL sequences...")
    for record in SeqIO.parse(sys.argv[2], "fasta"):
        seq_id = str(record.id)
        seq = str(record.seq)
        numbering = get_numbering(seq)
        esm_embedding = esmmodel.embed_seq(label=seq_id, seq=seq, layer=33, ab_chain=sys.argv[2].lower())
        ablang2_embedding = ablang2model.embed_seq(label=seq_id, seq=seq, ab_chain=sys.argv[2].lower())

        ablang2_vl.append(ablang2_embedding)
        esm2_vl.append(esm_embedding)
        VL_imgt_values.append(numbering)


    ablang2_vh = [t.to(torch.float16) for t in ablang2_vh]
    ablang2_vl = [t.to(torch.float16) for t in ablang2_vl]
    esm2_vh = [t.to(torch.float16) for t in esm2_vh]
    esm2_vl = [t.to(torch.float16) for t in esm2_vl]


    embedding_dict = {
        "AbLang2" : {
            "VH" : ablang2_vh,
            "VL" : ablang2_vl
        },

        "ESM2" : {
            "VH" : esm2_vh,
            "VL" : esm2_vl
        }
    }

    imgt_dict = {
        "VH" : VH_imgt_values,
        "VL" : VL_imgt_values
    }


    master_dict = {
        "embeddings" : embedding_dict,
        "imgt_numbering" : imgt_dict
    }


    with gzip.open("norm_data.pt.gz", "wb") as f:
        torch.save(master_dict, f)

if __name__ == "__main__":
    main()
