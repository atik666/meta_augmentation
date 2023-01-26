

data = []
img2label = {}
for i, (label, imgs) in enumerate(mini_train.items()):
    data.append(imgs)  # [[img1, img2, ...], [img111, ...]]
    img2label[label] = i  # {"img_name[:9]":label}
cls_num = len(data)

create_batch(batchsz, mode)

path = '/home/atik/Documents/UMAML_FSL/data/unsupervised/'

filenames = next(walk(path))[2]

img = []
for i in range(len(filenames)):  
    for images in glob.iglob(f'{path+filenames[i]}'):
        # check if the image ends with png
        if (images.endswith(".jpeg")) or (images.endswith(".jpg")):
            img_temp = images[len(path):]
            img.append(img_temp)

selected_cls = np.arange(5)

selected_imgs_idx = np.random.choice(len(img), 5, False)


support_x_batch = []  # support set batch
for _ in range(10):  # for each batch
    # 1.select n_way classes randomly

    selected_cls = np.arange(5)
    np.random.shuffle(selected_cls)
    
    selected_imgs_idx = np.random.choice(len(img), 5, False)
    support_x = [img[selected_imgs_idx[i]] for i in range(len(selected_imgs_idx))]

    support_x_batch.append(support_x) 



support_x = [img[selected_imgs_idx[i]] for i in range(len(selected_imgs_idx))]

flatten_support_x = [os.path.join(path, support_x_batch[0][item])
                             for item in range(len(support_x_batch[0]))]

flatten_support_y = np.repeat(flatten_support_x,5).tolist()

support_y = np.arange(5)
np.random.shuffle(support_y)

query_x = np.repeat(support_x,5).tolist()

query_y = np.repeat(support_y,5)

support_x = torch.FloatTensor(4, 3, 32, 32)



