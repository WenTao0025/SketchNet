conf = {
    'WORK_PATH': None,
    "CUDA_VISIBLE_DEVICES" : "0",
    "data":{
        'dataset_path':None,
        'resolution':None,
        'dataset': 'CASIA-B',
        'pid_shuffle':False,
    },
    "model":{
        'hidden_dim':256,
        'lr':1e-4,
        'hard_or_full_trip':'full',
        'batch_size':(8,16),
        'restore_iter':0,
        'total_iter':80000,
        'margin':0.2,
        'num_workers':16,
        'frame_num':30,
        'model_name':'GaitSet',
    },
}