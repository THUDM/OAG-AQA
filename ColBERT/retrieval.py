from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="nq_bsz32_dim256_lr1e-5")):

        config = ColBERTConfig(
            root="/path/to/experiments",
        )
        searcher = Searcher(index="/home/shishijie/workspace/project/ColBERT/experiments/nq_bsz32_dim256_lr1e-5/indexes/nq_bsz32_dim256_lr1e-5.nbits=2_steps70k", config=config)
        queries = Queries("/home/shishijie/workspace/project/oag-qa/data/nq_colbert/test.tsv")
        ranking = searcher.search_all(queries, k=100)
        ranking.save("nq_bsz32_dim256_lr1e-5_step70k.tsv")