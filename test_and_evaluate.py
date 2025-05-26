import json
import requests
from tqdm import tqdm
def produce_data():
    with open('test_question.json','r',encoding='utf-8') as f:
        data = json.load(f)
        questions = [d['question'] for d in data]

    test_datas = []
    headers = {
        "Authorization": "Bearer {APP API}",
        "Content-Type": "application/json"
    }

    for idx,question in tqdm(enumerate(questions),total=len(questions),desc = "processing question"):
        jsondata = {
            "inputs": {"input": question},
            "response_mode": "blocking",
            "user": "user"
        }
        
        re = requests.post('http://ip/v1/workflows/run',headers=headers,json=jsondata)
        response_data = json.loads(re.text)
        status = response_data['data']['status']
        if status == 'succeeded':
            text = response_data['data']['outputs']['text']
            elapsed_time = response_data['data']['elapsed_time']
            total_tokens = response_data['data']['total_tokens']
        else:
            text = ''
            elapsed_time = 0
            total_tokens = 0
        test_data = {
            "question":question,
            "status": status,
            "answer": text, # 这里是解码后的文本
            "spending_time": elapsed_time,
            "tokens": total_tokens,
            "judgement":True
        }
        test_datas.append(test_data)
        with open("test_result.json", "w+", encoding="utf-8") as f:
            json.dump(test_datas, f, ensure_ascii=False, indent=2)
            print("数据已保存到 test_result.json")

def data_analysis():
     with open('test_result_all.json','r',encoding='utf-8') as f:
        data_list = json.load(f)
        avg_time = 0.0
        avg_token = 0
        for data in data_list:
            avg_time += data['spending_time']
            avg_token += data['tokens']
        avg_time = avg_time / len(data_list)
        avg_token = avg_token / len(data_list)
        print(avg_time)
        print(avg_token)


produce_data()
data_analysis()
