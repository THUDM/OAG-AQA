import json
import os
from tqdm import tqdm


def split_train_dev():
    with open("/home/shishijie/workspace/project/oag-qa/data/stackex_dpr/train_with_hn.json", "r") as f:
        data = json.load(f)
    print(len(data))

    n_len = len(data)
    n_train = int(n_len * 2 / 3)
    data_train = data[:n_train]
    data_dev = data[n_train:]

    data_dir = "data/kddcup/dpr/"
    os.makedirs(data_dir, exist_ok=True)
    with open(data_dir + "train.json", "w", encoding="utf-8") as f:
        json.dump(data_train, f, indent=4, ensure_ascii=False)

    with open(data_dir + "dev.json", "w", encoding="utf-8") as f:
        json.dump(data_dev, f, indent=4, ensure_ascii=False)


def gen_dpr_passages_tsv():
    with open("data/kddcup/pid_to_title_abs_new.json") as rf:
        paper_dict = json.load(rf)
    
    wf = open("data/kddcup/candidate_papers.tsv", "w")
    for i, pid in tqdm(enumerate(paper_dict)):
        cur_paper = paper_dict[pid]
        abstract = cur_paper.get("abstract")
        title = cur_paper["title"]
        content = ""
        if title is not None:
            content = title.replace("\n", " ").replace("\r", " ")
        if abstract is not None:
            content += " "
            content += abstract.replace("\n", " ").replace("\r", " ")
        content = " ".join(content.split()[:500])
        wf.write(str(i) + "\t" + content + "\t" + pid + "\n")
        wf.flush()
    wf.close()


def gen_dpr_valid_input():
    wf = open("data/kddcup/qa_valid_dpr.tsv", "w")
    with open("data/kddcup/qa_valid_wo_ans.txt") as rf:
        for line in tqdm(rf):
            items = json.loads(line)
            question = items["question"]
            wf.write(question + "\t")
            wf.write("[]\n")
            wf.flush()
    wf.close()
        

if __name__ == "__main__":
    # gen_dpr_passages_tsv()
    gen_dpr_valid_input()
