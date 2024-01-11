from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=4, experiment="nq_bsz32_dim256_lr1e-5")):

        config = ColBERTConfig(
            nbits=2,
            root="/path/to/experiments",
            dim=256,
        )
        indexer = Indexer(checkpoint="/home/shishijie/workspace/project/ColBERT/experiments/nq_bsz32_dim256_lr1e-5/none/2023-12/27/11.17.08/checkpoints/colbert-70000", config=config)
        indexer.index(name="nq_bsz32_dim256_lr1e-5.nbits=2_steps70k", collection="/home/shishijie/workspace/project/oag-qa/data/nq_colbert/collections.tsv")