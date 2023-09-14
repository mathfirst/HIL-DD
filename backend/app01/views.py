import shutil
import time

import numpy as np
from django.shortcuts import render, HttpResponse, redirect
from django.http import JsonResponse
from django.middleware.csrf import get_token
import os, subprocess, json, sys
from subprocess import Popen

DETACHED_PROCESS = 0x00000008

if sys.version_info[0] < 3:
    from StringIO import BytesIO
else:
    from io import BytesIO

import pandas as pd


def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(BytesIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    logger.info('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
    # idx = np.argmax(gpu_df['memory.free'])
    indices = np.argpartition(gpu_df['memory.free'], -3)[-3:]  # get the indices of the GPUs with top 3 free memory
    for idx in indices:
        logger.info('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return indices


# Create your views here.
def get_logger(name, log_filename=None):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')
    # formatter = logging.Formatter('[%(asctime)s-%(name)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    #     logger.addHandler(stream_handler)
    #     https://alexandra-zaharia.github.io/posts/fix-python-logger-printing-same-entry-multiple-times/
    if not logger.hasHandlers():  # check whether any handlers are already attached before adding them to the logger
        logger.addHandler(stream_handler)
        # logger.addHandler(file_handler)
    if log_filename is not None:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = get_logger('ui_log', 'log_ui.txt')

def test(request):
    return HttpResponse('pref')


def getPDBList(request):
    import pandas as pd
    logger.info(f"getPDBList: {os.getcwd()}")
    df = pd.read_csv("../configs/PDB_ID_CrossDocked_testset.csv", encoding="utf-8")
    full_pdb_dir = './app01/static/full_pdb/'
    if not os.path.isdir(full_pdb_dir):
        logger.info(f'making dir {full_pdb_dir}')
        os.makedirs(full_pdb_dir)
    pdb_list = []
    for i, j in zip(df['Index'], df['PDB_ID']):
        shutil.copy(f'../configs/test_protein/protein_{i}.pdb', f'app01/static/full_pdb/protein_{j}.pdb')
        pdb_list.append({'id': i,
                         'name': j,
                         'url': f'full_pdb/protein_{j}.pdb'})

    return JsonResponse({'pdb_list': pdb_list, 'token': get_token(request)})


def getAnnotations(request):
    if request.method == 'POST':
        timestamp = str(request.POST.get('time_stamp'))
        like_ids = request.POST.get('liked_ids')
        dislike_ids = request.POST.get('disliked_ids')
        annotation_dict = {'liked_ids': like_ids, 'disliked_ids': dislike_ids}
        annotation_dir = os.path.join('./app01/static/', timestamp, 'annotations')
        os.makedirs(annotation_dir, exist_ok=True)
        annotation_path = os.path.join(annotation_dir, 'annotations.json')
        with open(annotation_path, "w") as outfile:
            json.dump(annotation_dict, outfile, indent=2)
        return HttpResponse('received')
    else:
        return HttpResponse('not received')


def getTimePDB(request):  # api/confirmpdb/
    timestamp = str(request.POST.get('timestamp'))
    pdb = str(request.POST.get('pdb_id'))
    logger.info(f"In getTimePDB, {timestamp}, {pdb}")
    logger.info('In getTimePDB, starting a human-in-the-loop drug design program which is driven by preference learning')
    output = os.getcwd()
    logger.info(f'current dir {output}')
    if not output.endswith('HIL-DD'):
        os.chdir('../')
        output = os.getcwd()
        logger.info(f'after changing dir {output}')
    # gpu_indices = get_free_gpu()
    gpu_indices = np.random.randint(0, 6, 3)
    logger.info(f"use GPU:{gpu_indices[0]} to propose")
    logger.info(f"use GPU:{gpu_indices[1]} to learn")
    logger.info(f"use GPU:{gpu_indices[2]} to evaluate")
    pyfile_propose = '/datapool/data2/home/pengxingang/zhaoyouming/HIL-DD/HIL_DD_ui_proposals.py'
    pycmd_propose = ' '.join(['python3', pyfile_propose, timestamp, pdb, '--device', 'cuda:' + str(gpu_indices[0])])
    logger.info(pycmd_propose)
    Popen(['bash', 'script.sh', '35', pycmd_propose], shell=False, close_fds=True)
    Popen(['bash', 'script.sh', '42', pycmd_propose], shell=False, close_fds=True)
    # Popen(['python3', propose_pyfile, timestamp, pdb, '--device', 'cuda:' + str(gpu_indices[0])], shell=False,
    #       close_fds=True)
    time.sleep(25)
    pyfile_learn = '/datapool/data2/home/pengxingang/zhaoyouming/HIL-DD/HIL_DD_ui_learning.py'
    pycmd_learn = ' '.join(['python3', pyfile_learn, timestamp, pdb, '--device', 'cuda:' + str(gpu_indices[1])])
    logger.info(pycmd_learn)
    Popen(['bash', 'script.sh', '36', pycmd_learn], shell=False, close_fds=True)
    # Popen(['python3', 'HIL_DD_ui_learning.py', timestamp, pdb, '--device', 'cuda:' + str(gpu_indices[1])], shell=False,
    #       close_fds=True)
    time.sleep(25)
    pyfile_eval = '/datapool/data2/home/pengxingang/zhaoyouming/HIL-DD/HIL_DD_ui_evaluation.py'
    pycmd_eval = ' '.join(['python3', pyfile_eval, timestamp, pdb, '--device', 'cuda:' + str(gpu_indices[2])])
    logger.info(pycmd_eval)
    Popen(['bash', 'script.sh', '38', pycmd_eval], shell=False, close_fds=True)
    Popen(['bash', 'script.sh', '43', pycmd_eval], shell=False, close_fds=True)
    # Popen(['python3', 'HIL_DD_ui_evaluation.py', timestamp, pdb, '--device', 'cuda:' + str(gpu_indices[2])], shell=False,
    #       close_fds=True)
    output = os.getcwd()
    if output.strip().endswith('HIL-DD'):
        os.chdir('./backend/')
        logger.info(f'changed back to {os.getcwd()}')

    # return redirect(f"/api/getMoleculeList/?timestamp={logdir}&pdb={pdb}")
    return sendMoleculeList(request, True)


def sendMoleculeList(request, start=False):
    timestamp = str(request.POST.get('timestamp'))
    if start:
        logger.info('sendMoleculeList, starting a task...')
    else:
        logger.info('sendMoleculeList, getting annotations...')
        # like_ids = request.POST.get('liked_ids')
        # dislike_ids = request.POST.get('disliked_ids')
        received_json_data = json.loads(request.body)
        logger.info(f"sendMoleculeList, received_json_data: {received_json_data}")
        timestamp = received_json_data['timestamp']
        # pdb_id = received_json_data.pdb_id
        like_ids = received_json_data['liked_ids']
        dislike_ids = received_json_data['disliked_ids']
        annotation_dict = {'liked_ids': like_ids, 'disliked_ids': dislike_ids}
        annotation_dir = os.path.join('./app01/static/', timestamp, 'annotations')
        os.makedirs(annotation_dir, exist_ok=True)
        annotation_path = os.path.join(annotation_dir, 'annotations.json')
        with open(annotation_path, "w") as outfile:
            json.dump(annotation_dict, outfile, indent=2)
    pdb = str(request.POST.get('pdb_id'))
    logger.info(f'sendMoleculeList: {timestamp}, {pdb}')
    proposal_dir = os.path.join('./app01/static/', timestamp, 'proposals')
    used_proposal_dir = os.path.join('./app01/static/', timestamp, 'used_proposals')
    os.makedirs(used_proposal_dir, exist_ok=True)
    # proposal_json = os.path.join(proposal_dir, 'proposals.json')
    # logger.info(f"sendMoleculeList: {os.path.abspath(proposal_json)}")
    while True:
        proposal_files = [f for f in os.listdir(proposal_dir) if f.endswith('.json')]
        if len(proposal_files):
            logger.info(f"before sorting proposal filenames: {proposal_files}")
            proposal_files.sort(key=lambda x: np.longlong(x.split('.')[0].split('_')[1]))
            logger.info(f"after sorting proposal filenames: {proposal_files}")
            proposal_json = os.path.join(proposal_dir, proposal_files[0])
        # if os.path.isfile(proposal_json):
            logger.info(f"loading {proposal_json}")
            while True:
                try:
                    with open(proposal_json, 'r') as f:
                        proposals = json.load(f)
                        # proposals = json.loads(f.read())
                    break
                except Exception as err:
                    logger.info(f"error when reading proposal file: {err}")
                    time.sleep(0.5)
            shutil.move(proposal_json, used_proposal_dir)
                        # os.path.join(proposal_dir, f"proposals_{len(os.listdir(proposal_dir))}.json"))
            logger.info(f"proposals: {proposals}")
            break
    # annotation_dir = os.path.join('./app01/static/', timestamp, 'annotations')
    # os.makedirs(annotation_dir, exist_ok=True)
    # annotation_json = os.path.join(annotation_dir, 'annotations.json')
    # if request.GET.get('sim'):
    #     next_mols = proposals['next_molecules']
    #     like, dislike = [], []
    #     for i in range(len(next_mols)):
    #         metrics = next_mols[i]['metrics']
    #         vina = metrics['Vina']
    #         if vina <= -8:
    #             like.append(i)
    #         elif vina > -7:
    #             dislike.append(i)
    #     annotations = {'time_stamp': timestamp,
    #                    'like_ids': like,
    #                    'dislike_ids': dislike}
    #     print('like list', like)
    #     print('dislike list', dislike)
    #     with open(annotation_json, "w") as outfile:
    #         json.dump(annotations, outfile, indent=2)
    # if os.path.isfile(proposal_json):
    #     print(f'removing {proposal_json}')
    #     os.remove(proposal_json)
    # # return HttpResponse(output)
    # return redirect("api/getMoleculeList/")
    return JsonResponse(proposals)


def evaluation(request):
    timestamp = str(request.GET.get('timestamp'))
    # pdb = str(request.GET.get('pdb'))
    logger.info(f"evaluation: {timestamp}")
    eval_dir = os.path.join('./app01/static/', timestamp, 'evaluation')
    evaluation_json = os.path.join(eval_dir, 'evaluation.json')
    while True:
        if os.path.isfile(evaluation_json):
            logger.info(f"loading {evaluation_json}")
            time.sleep(0.1)
            with open(evaluation_json, 'r') as f:
                evaluation_samples = json.load(f)
            os.rename(evaluation_json, os.path.join(eval_dir, f'evaluation_{len(os.listdir(eval_dir))}.json'))
            break

    return JsonResponse(evaluation_samples)


def feedback(request):
    timestamp = request.GET.get('timestamp')
    logger.info(f"Entering feedback, timestamp is {timestamp}.")
    received_json_data = json.loads(request.body)
    score = received_json_data['score']
    logger.info(f"The score is {score}.")
    return HttpResponse('success')


def login(request):
    print('entering login')
    return render(request, 'login.html')
