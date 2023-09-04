import shutil
import time

from django.shortcuts import render, HttpResponse, redirect
from django.http import JsonResponse
from django.middleware.csrf import get_token
import os, subprocess, json
from subprocess import Popen

DETACHED_PROCESS = 0x00000008
from sys import platform


# sys.path.append('../../../HIL-DD')
# Create your views here.


def test(request):
    return HttpResponse('pref')


def getPDBList(request):
    import pandas as pd
    df = pd.read_csv("../configs/PDB_ID_CrossDocked_testset.csv", encoding="utf-8")
    full_pdb_dir = 'app01/static/full_pdb/'
    if not os.path.isdir(full_pdb_dir):
        print(f'making dir {full_pdb_dir}')
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


def getTimePDB(request):
    timestamp = str(request.POST.get('timestamp'))
    pdb = str(request.POST.get('pdb_id'))
    print(timestamp)
    print(pdb)
    print('starting a human-in-the-loop drug design program which is driven by preference learning')
    output = os.getcwd()
    print('current dir', output)
    if not output.endswith('HIL-DD'):
        os.chdir('../')
        output = os.getcwd()
        print('after changing dir', output)
    Popen(['python3', 'HIL_DD_ui_proposals.py', timestamp, pdb, '--device cuda:1'], shell=False,
          close_fds=True)  # , creationflags=DETACHED_PROCESS)
    time.sleep(30)
    Popen(['python3', 'HIL_DD_ui_learning.py', timestamp, pdb, '--device cuda:2'], shell=False,
          close_fds=True)  # , creationflags=DETACHED_PROCESS)
    time.sleep(20)
    Popen(['python3', 'HIL_DD_ui_evaluation.py', timestamp, pdb, '--device cuda:3'], shell=False,
          close_fds=True)  # , creationflags=DETACHED_PROCESS)
    output = os.getcwd()
    if output.strip().endswith('HIL-DD'):
        os.chdir('./backend/')
        print('changed back to', os.getcwd())

    # return redirect(f"/api/getMoleculeList/?timestamp={logdir}&pdb={pdb}")
    return sendMoleculeList(request, True)


def sendMoleculeList(request, start=False):
    timestamp = str(request.POST.get('timestamp'))
    if start:
        print('starting a task...')
    else:
        print('getting annotations...')
        like_ids = request.POST.get('liked_ids')
        dislike_ids = request.POST.get('disliked_ids')
        received_json_data = json.loads(request.body)
        print(received_json_data)
        timestamp = received_json_data.timestamp
        pdb_id = received_json_data.pdb_id
        like_ids = received_json_data.like_ids
        dislike_ids = received_json_data.dislike_ids
        annotation_dict = {'liked_ids': like_ids, 'disliked_ids': dislike_ids}
        annotation_dir = os.path.join('./app01/static/', timestamp, 'annotations')
        os.makedirs(annotation_dir, exist_ok=True)
        annotation_path = os.path.join(annotation_dir, 'annotations.json')
        with open(annotation_path, "w") as outfile:
            json.dump(annotation_dict, outfile, indent=2)
    pdb = str(request.POST.get('pdb_id'))
    print('sendMoleculeList', timestamp)
    print(pdb)
    proposal_dir = os.path.join('./app01/static/', timestamp, 'proposals')
    proposal_json = os.path.join(proposal_dir, 'proposals.json')
    print(os.path.abspath(proposal_json))
    while True:
        if os.path.isfile(proposal_json):
            print(f"loading {proposal_json}")
            os.system('django-admin collectstatic')
            print('django-admin collectstatic is done.')
            time.sleep(0.1)
            with open(proposal_json, 'r') as f:
                proposals = json.load(f)
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
    pdb = str(request.GET.get('pdb'))
    print(timestamp)
    print(pdb)
    eval_dir = os.path.join('./app01/static/', timestamp, 'evaluation')
    evaluation_json = os.path.join(eval_dir, 'evaluation.json')
    while True:
        if os.path.isfile(evaluation_json):
            print(f"loading {evaluation_json}")
            time.sleep(0.1)
            with open(evaluation_json, 'r') as f:
                evaluation_samples = json.load(f)
            os.rename(evaluation_json, os.path.join(eval_dir, f'evaluation_{len(os.listdir(eval_dir))}.json'))
            break

    return JsonResponse(evaluation_samples)
    # return HttpResponse('success')


def login(request):
    print('entering login')
    return render(request, 'login.html')
