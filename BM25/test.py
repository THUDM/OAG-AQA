import json
import argparse

# 计算AP
def calculate_AP(Pq):
    Rq = sum(Pq)
    M = len(Pq)
    AP = 0
    for k in range(1, M + 1):
        if Pq[k - 1] == 1:
            AP += sum(Pq[:k]) / k
    if Rq == 0:
        return 0
    return AP / Rq

# 计算MAP
def calculate_MAP(questions):
    n = len(questions)
    total_AP = 0
    for q in questions:
        AP = calculate_AP(q)
        total_AP += AP
    MAP = total_AP / n
    return MAP

def format_check(file_path):
    """
    文件格式校对
    检查内容为：1、预测结果的每一行，是否包含20个论文pids；2、是否生成了2919个问题的答案
    :param file_path: 文件路径
    :return:
        flag: 通过格式检查为True, 反之为False
        info_json: 检查结果信息
    """
    flag = True
    info_json = {"err_code": 0, "err_msg": "[success] submission success"}
    
    
    with open(file_path, 'r') as fr:
        
        lines=fr.readlines()
        count=len(lines)
        
        if count<2919:
            flag=False
            info_json={"err_code": 1, "err_msg": "[file] file <2919 lines"}
            return flag, info_json
        
        if count>2919:
            flag=False
            info_json={"err_code": 1, "err_msg": "[file] file >2919 lines"}
            return flag, info_json
        
        for line in lines:
            line_list=line.strip().split(',')
            if len(line_list)<20:
                flag=False
                info_json={"err_code": 2, "err_msg": "[file] output <20 values"}
                break
            if len(line_list)>20:
                flag=False
                info_json={"err_code": 2, "err_msg": "[file] output >20 values"}
                break

            if not flag:
                break

    
    return flag, info_json

def stackex_QA(hp,rf,l):
    """
    计算MAP
    :param hp: 选手提交文件
    :param rf: golden集文件
    :param l: 结果文件,MAP
    :return:
    """
    flag_submission, sub_info_json = format_check(hp)

    if not flag_submission:
        with open(l, "w", encoding="utf-8") as f:
            f.writelines(str(sub_info_json))
        return 0
    
    preds=[]
    with open(hp,'r',encoding='utf8') as fr:
        for line in fr.readlines():
            line=line.strip()
            preds.append(line.split(','))

    labels=[]
    with open(rf,'r',encoding='utf8') as fr:
        for line in fr.readlines():
            line=json.loads(line.strip())
            labels.append(line['pids'])

    questions=[]
    for pred,label in zip(preds,labels):
        temp=[0]*20
        for idx,item in enumerate(pred):
            if item in label:
                temp[idx]=1
        questions.append(temp)
    
    map_score = calculate_MAP(questions)

    with open(l, "w", encoding="utf-8") as f:
        f.writelines(str(map_score) + f"###submision success")
    return 0

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('-hp', help='学生提交文件')
    parser.add_argument('-rf',  help='答案文件')
    parser.add_argument('-l',  help='结果文件')
    args = parser.parse_args()

    try:
        stackex_QA(args.hp, args.rf, args.l)
    except Exception as e:
        with open(args.l, "w", encoding="utf-8") as f:
            f.write(str(e))

