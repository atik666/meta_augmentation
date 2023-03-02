import numpy as np
import pickle
import  torchvision.transforms as transforms
from    PIL import Image
import  os
import matplotlib.pyplot as plt

path = '/home/atik/Documents/MAML/Summer_1/datasets/Omniglot/'
data = np.load(path+'omniglot.npy')

train = pickle.load(open(path+"train.pickle", "rb"))
val = pickle.load(open(path+"val.pickle", "rb"))

a = Image.open('/home/atik/Documents/MAML/Summer_1/datasets/Omniglot/images_background/Alphabet_of_the_Magi/character01/0709_01.png').convert('L')

aa = np.asarray(a)
aa = np.expand_dims(aa, axis = 0)
aa = np.reshape(aa, (105, 105, 1))
aa = aa/255
img = Image.fromarray(aa)
img.show()
plt.imshow(aa)

a = a.resize((28, 28))
a = np.reshape(a, (28, 28, 1))
a = np.transpose(a, [2, 0, 1])

imgsz = 28
x = transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                lambda x: x.resize((imgsz, imgsz)),
                                                lambda x: np.reshape(x, (imgsz, imgsz, 1)),
                                                lambda x: np.transpose(x, [2, 0, 1]),
                                                lambda x: x/255.])

transform = x('/home/atik/Documents/MAML/Summer_1/datasets/Omniglot/images_background/Alphabet_of_the_Magi/character01/0709_01.png')

x_new = transforms.Compose([lambda x: x,
                                      transforms.ToTensor()])

transform_query = transforms.Compose([lambda x: x,
                                       # transforms.ToPILImage(),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.RandomResizedCrop(28,(0.8,1.0)),
                                      #transforms.RandomResizedCrop(84),
                                      transforms.ToTensor(),
                                      #transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11)),
                                      #transforms.RandomRotation(degrees=(60, 90)),
                                      transforms.RandomInvert(p=0.5),
                                      transforms.GaussianBlur(kernel_size=9)])

sample = data_pack[cur_class][selected_img[:k_shot]]
sample1 = np.squeeze(sample)

sample = sample.astype(np.uint8)

sample1 = np.squeeze(sample, axis=0)

pil_img = Image.fromarray(sample1)
pil_img.show()

transform1 = x_new(Image.fromarray(sample1).convert('L'))

# TODO: modify
transform1 = transform_query(Image.fromarray(sample1).convert('L'))

x_spt1 = np.squeeze(x_spt, axis=1)

for i in x_spt1:  
    transform2 = transform_query(Image.fromarray(i).convert('L'))


transform1 = transform_query(pil_img)

plt.imshow(sample1)



x1 = np.load(os.path.join(path, 'omniglot.npy'))
    
x_train, x_test = x1[:1200], x1[1200:]

batchsz = 1
n_cls = x1.shape[0]  # 1623
n_way = 5  # n way
k_shot = 1  # k shot
k_query = 1  # k query
assert (k_shot + k_query) <=20


indexes = {"train": 0, "test": 0}
datasets = {"train": x_train, "test": x_test}  # original data cached
print("DB: train", x_train.shape, "test", x_test.shape)



def load_data_cache(data_pack):
    """
    Collects several batches data for N-shot learning
    :param data_pack: [cls_num, 20, 84, 84, 1]
    :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
    """
    #  take 5 way 1 shot as example: 5 * 1
    resize = 28
    setsz = k_shot * n_way
    querysz = k_query * n_way
    data_cache = []

    # print('preload next 50 caches of batchsz of batch.')
    for sample in range(10):  # num of episodes

        x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
        for i in range(batchsz):  # one batch means one set

            x_spt, y_spt, x_qry, y_qry = [], [], [], []
            selected_cls = np.random.choice(data_pack.shape[0], n_way, False)

            for j, cur_class in enumerate(selected_cls):

                selected_img = np.random.choice(20, k_shot + k_query, False)

                # meta-training and meta-test
                x_spt.append(data_pack[cur_class][selected_img[:k_shot]])
                x_qry.append(data_pack[cur_class][selected_img[k_shot:]])
                y_spt.append([j for _ in range(k_shot)])
                y_qry.append([j for _ in range(k_query)])

            # shuffle inside a batch
            perm = np.random.permutation(n_way * k_shot)
            x_spt = np.array(x_spt).reshape(n_way * k_shot, 1, resize, resize)[perm]
            y_spt = np.array(y_spt).reshape(n_way * k_shot)[perm]
            perm = np.random.permutation(n_way * k_query)
            x_qry = np.array(x_qry).reshape(n_way * k_query, 1, resize, resize)[perm]
            y_qry = np.array(y_qry).reshape(n_way * k_query)[perm]

            # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
            x_spts.append(x_spt)
            y_spts.append(y_spt)
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)


        # [b, setsz, 1, 84, 84]
        x_spts = np.array(x_spts).astype(np.float32).reshape(batchsz, setsz, 1, resize, resize)
        y_spts = np.array(y_spts).astype(int).reshape(batchsz, setsz)
        # [b, qrysz, 1, 84, 84]
        x_qrys = np.array(x_qrys).astype(np.float32).reshape(batchsz, querysz, 1, resize, resize)
        y_qrys = np.array(y_qrys).astype(int).reshape(batchsz, querysz)

        data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

    return data_cache

datasets_cache = {"train": load_data_cache(datasets["train"]),  # current epoch data cached
                       "test": load_data_cache(datasets["test"])}



data_pack = datasets["train"] 
resize = 28
setsz = k_shot * n_way
querysz = k_query * n_way
data_cache = []

# print('preload next 50 caches of batchsz of batch.')
for sample in range(10):  # num of episodes

    x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
    for i in range(batchsz):  # one batch means one set

        x_spt, y_spt, x_qry, y_qry = [], [], [], []
        selected_cls = np.random.choice(data_pack.shape[0], n_way, False)

        for j, cur_class in enumerate(selected_cls):

            selected_img = np.random.choice(20, k_shot, False)

            # meta-training and meta-test
            # TODO: take data_pack
            x_spt.append(data_pack[cur_class][selected_img[:k_shot]])

            # TODO: modify query here
            a = data_pack[cur_class][selected_img[:k_shot]]
            aa = np.squeeze(np.squeeze(a, axis=0),axis=0)
            
            transform2 = transform_query(Image.fromarray(aa).convert('L')) 
            transform2 = transform2.cpu().detach().numpy()
            transform2 = np.expand_dims(transform2, axis=1)
            x_qry.append(transform2)
                
            y_spt.append([j for _ in range(k_shot)])
            y_qry.append([j for _ in range(k_query)])

        # shuffle inside a batch
        perm = np.random.permutation(n_way * k_shot)
        x_spt = np.array(x_spt).reshape(n_way * k_shot, 1, resize, resize)[perm]
        y_spt = np.array(y_spt).reshape(n_way * k_shot)[perm]
        perm = np.random.permutation(n_way * k_query)
        x_qry = np.array(x_qry).reshape(n_way * k_query, 1, resize, resize)[perm]
        y_qry = np.array(y_qry).reshape(n_way * k_query)[perm]

        # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
        x_spts.append(x_spt)
        y_spts.append(y_spt)
        x_qrys.append(x_qry)
        y_qrys.append(y_qry)


    # [b, setsz, 1, 84, 84]
    x_spts = np.array(x_spts).astype(np.float32).reshape(batchsz, setsz, 1, resize, resize)
    y_spts = np.array(y_spts).astype(int).reshape(batchsz, setsz)
    # [b, qrysz, 1, 84, 84]
    x_qrys = np.array(x_qrys).astype(np.float32).reshape(batchsz, querysz, 1, resize, resize)
    y_qrys = np.array(y_qrys).astype(int).reshape(batchsz, querysz)

    data_cache.append([x_spts, y_spts, x_qrys, y_qrys])