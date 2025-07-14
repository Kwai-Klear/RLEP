import os
import json
from collections import defaultdict
# import time
from datetime import datetime
from tqdm import tqdm

class RolloutRecorder:
    def __init__(self, local_folder, steps=1000):
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
        self.local_folder = f'{local_folder}/rollout_logs'
        if not os.path.exists(self.local_folder):
            os.makedirs(self.local_folder)
        self.folder_idx = 0
        self.prompt2folder = {}
        self.question_last_average_reward = {} # 最近一次Rollout的记录；
        self.question_last_rollout = {} # 最近一次Rollout的记录；
        self.step = 0
        self.resume(steps)
        
        # import ipdb
        # ipdb.set_trace()
    
    def resume(self, steps):
        return 
        """
        从已有训练记录中resume，稍晚添加；
        """
        the_steps = set()
        for folder in tqdm(os.listdir(self.local_folder)):
            if not folder.startswith('rollout_of_question_'):
                continue
            folder = os.path.join(self.local_folder, folder)
            files = sorted(os.listdir(folder))
            import ipdb
            # ipdb.set_trace() 
            while len(files):
                latest_file = files[-1]
                latest_file = os.path.join(folder, latest_file)
                try:
                # if 1:
                    record = json.loads(open(latest_file).readline())
                    if record['step'] // 512 > steps:
                        files = files[:-1]
                        # 按指定位置重启，不要加载Future的Rollout结果！
                        continue
                    the_steps.add(record['step'])
                    self.step = max(self.step, record['step'])
                    question = record['question']
                    self.question_last_rollout[question] = record
                    self.prompt2folder[question] = folder
                    break
                except:
                    print('exception  in reading ', latest_file)
                    pass
        # ipdb.set_trace()
        # print(the_steps, 'resumed from theses steps!')
        print(len(self.question_last_rollout), 'self.question_last_rollout')
    def do_record(self, inputs, outputs, scores, inputs_ids=None, output_ids=None, entropys=None):
        # import ipdb
        # ipdb.set_trace()
        if inputs_ids is None:
            inputs_ids = [None] * len(inputs)
            output_ids = [None] * len(inputs)
            entropys = [None] * len(inputs)

        question2answers = defaultdict(list)
        question2scores = defaultdict(list)
        question2others = defaultdict(list)
        for question, candidate, score, inid, outid, entropy in zip(inputs, outputs, scores, inputs_ids, output_ids, entropys):
            question2answers[question].append(candidate)
            question2scores[question].append(score)
            # entropy = [round(num, 3) for num in entropy] if entropy is not None else entropy
            question2others[question].append([inid, outid, entropy])

        for question, answers in question2answers.items():
            scores = question2scores[question]
            id_info = question2others[question]
            if question in self.prompt2folder:
                out_folder = self.prompt2folder[question]
            else:
                out_folder = os.path.join(self.local_folder, f'rollout_of_question_{self.folder_idx}')
                self.prompt2folder[question] = out_folder # os.path.join(self.local_folder, timestamp)
                self.folder_idx += 1
                if not os.path.exists(out_folder):
                    os.mkdir(out_folder)
            # 给当前的问题建一个临时的时间戳
            out_folder # 这个问题的记录文件夹
            timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S__%f")
            ofnm = os.path.join(out_folder, timestamp)
            record = {'question': question, 
                        'answers': answers, 
                        'rewards': scores,  
                        'ground_truth': '', 
                        'average_reward': '',
                        'step': self.step,
                        'id_info_and_entropy': id_info, 
                        }
            ln = json.dumps(record, ensure_ascii=False) + '\n'
            with open(ofnm, 'w') as of: # 每个step500个问题，不会超过5s？
                of.write(ln)
            self.step += 1
            self.question_last_rollout[question] = record

if __name__ == '__main__':
    # RolloutRecorder('/nlp_group/zhanghongzhi/verl/save/math-zero-dapo-hz/DAPO-2node_v0508_exp5_r1_and_rej_norm2debug2')
    path = '/nlp_group/zhanghongzhi/verl/save/math-zero-dapo-hz-v0509/DAPO-4node_v0501_rerunv0509'
    path = '/nlp_group/zhanghongzhi/verl/save/ForRelease/grpo_baseline_clip_high_ec'
    RolloutRecorder(path, 280)
