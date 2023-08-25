import shutil
import time

from django.shortcuts import render, HttpResponse, redirect
from django.http import JsonResponse
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

    return JsonResponse({'pdb_list': pdb_list})


def getAnnotations(request):
    if request.method == 'POST':
        time_stamp = request.POST.get('time_stamp')
        like_ids = request.POST.get('liked_ids')
        dislike_ids = request.POST.get('disliked_ids')
        annotation_dict = {'liked_ids': like_ids, 'disliked_ids': dislike_ids}
        annotation_path = os.path.join('../logs/annotations/', 'annotations.json')
        with open(annotation_path, "w") as outfile:
            json.dump(annotation_dict, outfile, indent=2)
        return HttpResponse('receieved')
    else:
        return HttpResponse('not receieved')


def getTimePDB(request):
    logdir = request.GET.get('timestamp')
    pdb = request.GET.get('pdb')
    print(logdir)
    print(pdb)
    proposal_dir = '../logs/' + logdir + '/proposals/'
    os.makedirs(proposal_dir, exist_ok=True)
    proposal_json = os.path.join(proposal_dir, 'proposals.json')
    print(os.path.isfile(proposal_json), proposal_json, proposal_dir)
    print('starting a human-in-the-loop drug design program which is driven by preference learning')
    output = os.getcwd()
    print('current dir', output)
    if not output.endswith('HIL-DD'):
        os.chdir('../')
        output = os.popen('cd').read()
        print('after changing dir', output)
    Popen(['python3', 'HIL_DD_ui_proposals.py', logdir, pdb], shell=False,
          close_fds=True)#, creationflags=DETACHED_PROCESS)
    time.sleep(30)
    Popen(['python3', 'HIL_DD_ui_learning.py', logdir, pdb], shell=False,
          close_fds=True)#, creationflags=DETACHED_PROCESS)
    time.sleep(20)
    Popen(['python3', 'HIL_DD_ui_evaluation.py', logdir, pdb], shell=False,
          close_fds=True)#, creationflags=DETACHED_PROCESS)
    output = os.getcwd()
    if output.strip().endswith('HIL-DD'):
        os.chdir('./backend/')
        print('changed back to', os.popen('cd').read())

    # return redirect(f"/api/getMoleculeList/?timestamp={logdir}&pdb={pdb}")
    return sendMoleculeList(request)


def sendMoleculeList(request):
    logdir = request.GET.get('timestamp')
    pdb = request.GET.get('pdb')
    print('sendMoleculeList', logdir)
    print(pdb)
    proposal_dir = './app01/static/' + logdir + '/proposals/'
    annotation_dir = './app01/static/' + logdir + '/annotations/'
    os.makedirs(annotation_dir, exist_ok=True)
    annotation_json = os.path.join(annotation_dir, 'annotations.json')
    proposal_json = os.path.join(proposal_dir, 'proposals.json')
    print(os.path.abspath(proposal_json))
    while True:
        if os.path.isfile(proposal_json):
            print(f"loading {proposal_json}")
            time.sleep(0.1)
            with open(proposal_json, 'r') as f:
                proposals = json.load(f)
            # dst = os.path.join(proposal_dir, f'proposals_{len(os.listdir(proposal_dir))}.json')
            # shutil.move(proposal_json, dst)
            break
    next_mols = proposals['next_molecules']
    like, dislike = [], []
    for i in range(len(next_mols)):
        metrics = next_mols[i]['metrics']
        vina = metrics['Vina']
        if vina <= -8:
            like.append(i)
        elif vina > -7:
            dislike.append(i)
    annotations = {'time_stamp': logdir,
         'like_ids': like,
         'dislike_ids': dislike}
    print('like list', like)
    print('dislike list', dislike)
    with open(annotation_json, "w") as outfile:
        json.dump(annotations, outfile, indent=2)
    # if os.path.isfile(proposal_json):
    #     print(f'removing {proposal_json}')
    #     os.remove(proposal_json)
    # # return HttpResponse(output)
    # return redirect("api/getMoleculeList/")
    return JsonResponse(proposals)


def evaluation(request):
    timestamp = request.GET.get('timestamp')
    pdb = request.GET.get('pdb')
    print(timestamp)
    print(pdb)
    eval_dir = '../logs/' + timestamp + '/evaluation/'
    evaluation_json = '../logs/' + timestamp + '/evaluation/' + 'evaluation.json'
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
