from omni_class import OmniglotNShot
#from omniglot import OmniglotNShot
import torch
import numpy as np
from meta import Meta
from tqdm import tqdm

def main():
    
    n_way = 5
    epochs = 4000
    k_shot = 1
    k_query = 1
    task_num = 32

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)


    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [n_way, 64])
    ]

    device = torch.device('cuda:0')
    maml = Meta(config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print("Model: \n", maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    
    # train_path = '/home/atik/Documents/UMAML_FSL/data/unsupervised/'
    # test_path = '/home/atik/Documents/UMAML_FSL/data/'
    # model_path = '/home/atik/Documents/Meta Augmentation/model_%sw_%ss_%sq.pth' %(n_way,k_shot,k_query)
    
    # mini_train = OmniglotNShot(train_path, mode='train', n_way=n_way, k_shot=k_shot,
    #                     k_query=k_query,
    #                     batchsz=10000, resize=84)
    # mini_test = OmniglotNShot(test_path, mode='test', n_way=n_way, k_shot=k_shot,
    #                          k_query=k_query,
    #                          batchsz=100, resize=84)
    
    path = '/home/atik/Documents/MAML/Summer_1/datasets/Omniglot/'
    db_train = OmniglotNShot(path,
                   batchsz=task_num,
                   n_way=n_way,
                   k_shot=k_shot,
                   k_query=k_query,
                   imgsz=28)
    
    for step in tqdm(range(epochs)):
        
        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        accs = maml(x_spt, y_spt, x_qry, y_qry)
        
        if step % 50 == 0:
            print('\n','step:', step, '\ttraining acc:', accs)

        if step % 500 == 0:  # evaluation

            accs = []
            for _ in range(1000//task_num):
            # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            
            print('\n','Test acc:', accs)

main()    
