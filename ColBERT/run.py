from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=4, experiment="nq_bsz32_dim256_lr1e-5")):

        config = ColBERTConfig(
            bsize=32,
            root="/home/shishijie/workspace/project/ColBERT/output",
            doc_maxlen=256,
            dim=256,
            lr=1e-5,
        )
        trainer = Trainer(
            triples="/home/shishijie/workspace/project/oag-qa/data/nq_colbert/triples.tsv",
            queries="/home/shishijie/workspace/project/oag-qa/data/nq_colbert/queries.tsv",
            collection="/home/shishijie/workspace/project/oag-qa/data/nq_colbert/collections.tsv",
            config=config,
        )

        checkpoint_path = trainer.train()

        print(f"Saved checkpoint to {checkpoint_path}...")