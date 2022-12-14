
# π Annotation μ©λλ‘ μ°μΌ Meta File (.txt) μμ±νλ νμ΄μ¬ νμΌ 
# π Train & Val & Testλ‘ λΆν λΌμμ§ μμ λ°μ΄ν°(μ΄λ―Έμ§)μ νμΌλͺμ μΆμΆν΄μ νμΌλͺμ λ΄μ train & test txtνμΌ νμ±

# --> Split dataset into Train & Test   (Split dataset into Train & Val & Testλ ./extract_split_move.pyμ μμΉ)
# --> λ°μ΄ν°μ μμ± μ μ΄λ―Έμ§ νμΌλͺμ κ°μ Έμμ μ΄λ―Έμ§ νμΌμ λ§€μΉ­λλ json νμΌλ‘ label μμ


'''
< Structure Example >
βββ Root folder (name : airplane_custom)
β   βββ Dataset folder (name : 100_data)
β         βββ images (folder name : 100_images)
β         βββ labels (folder name : 100_labels)
β         βββ train.txt
β         βββ val.txt
β         βββ (text.txt)
'''



###################################### π Import Libraries π ##########################################

# #Install mmcv if not installed
# print(f"β¨[Info Msg] MMCV Install Start \n")
# os.system('pip install -qq mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html')
# print(f"β¨[Info Msg] MMCV Install Complete π οΈ \n\n")


import os
import mmcv # mmcv μ€μΉ
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm


###################################### π Hyperparameters π ##########################################
def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract_Split_Move')
    parser.add_argument('--root_dir', help='Path : root_dir/dataset_dir/images or labels.')
    parser.add_argument('--dataset_dir', help='Name of dataset folder.') # 100_data
    parser.add_argument('--test_ratio', type=float, default=0.2, 
                        help='Ratio of test data to generate. Train : 0.8 / Test : 0.2')
    parser.add_argument('--tr_anno_name', help='Name of Train Annotation file.')
    parser.add_argument('--val_anno_name', help='Name of Valid Annotation file.')
    parser.add_argument('--test_anno_name', help='Name of Test Annotation file.')   

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():

    args = parse_args()

    os.chdir(f'{args.root_dir}')
    target_path = f'{args.root_dir}/{args.dataset_dir}/'
    tgt_img_path = os.path.join(target_path, '100_images')
    tgt_label_path = os.path.join(target_path, '100_labels')


    print(f'\nβ¨[Info msg] Loading total images and labels')
    img_100_list = sorted(tqdm(glob(os.path.join(tgt_img_path,'*'))))
    label_100_list = sorted(tqdm(glob(os.path.join(tgt_label_path,'*'))))
    print(f'total imgs num : {len(img_100_list)}')
    print(f'total labels num : {len(label_100_list)}\n')


    ###################################### π Extract Filename to make Annotation file π##########################################
    print(f'\nβ¨[Info msg] Extract filenames out of imgs & labels list to make Annotation file')
    def name_parsing(file_list):

        name = []

        for i in tqdm(range(len(file_list))):
            file_list[i] = file_list[i].split('/')[-1].split('.')[0]
            name.append(file_list[i])

        return name

    imgs = name_parsing(img_100_list)
    labels = name_parsing(label_100_list)

    print(f'\nlength of imgs: {len(imgs)}')
    print(f'length of labels : {len(labels)}\n')



    ############################################### π Data Split π##########################################

    # split the 100_data into two groups
    # trian 0.8, test 0.2

    # sklearn ν¨ν€μ§μμ μ κ³΅νλ ShuffleSplitκ³Ό λ°μ΄ν°μμ λΆν 
    # shufflesplit ν¨μλ λ°μ΄ν°μ μΈλ±μ€λ₯Ό λ¬΄μμλ‘ μ¬μ μ μ€μ ν λΉμ¨λ‘ λΆν 
    # μ¦, 4:1 λ‘ λΆν νκ³  μΆμ κ²½μ°μ λ¬΄μμ μΈλ±μ€λ‘ 4:1 λΉμ¨λ‘ λΆν 

    print(f'\nβ¨[Info msg] Split data into Train & Test.. Ratio : {float(1-args.test_ratio)} & {args.test_ratio} ')

    from sklearn.model_selection import ShuffleSplit

    train_idx = []
    test_idx = []

    sss = ShuffleSplit(n_splits=1, test_size=args.test_ratio, random_state=100)
    indices = range(len(imgs)) # μ΄λ―Έμ§μ μ΄ κ°μλ₯Ό μΈλ±μ€λ‘ λ³ν

    # κ° μ΄λ―Έμ§λ€μ νμΌλͺμ΄ λ¦¬μ€νΈμ μΈλ±μ€λ‘ μ κ·Όν  μ μμΌλ―λ‘ μΈλ±μ€λ₯Ό νμ©ν΄μ train & test λΆν 
    for train_index, test_index in sss.split(indices):  
        train_idx.append(train_index)
        test_idx.append(test_index)
    
    '''
    12000
    ----------
    3000 

    train_idx
    -->
    [array([ 6859, 10995,  4306, ..., 14147,  6936,  5640])]
    '''

    ################### π Images Split #######################
    ## images

    # ShuffleSplitμ arrayλ‘ λ°νλλ―λ‘ listλ‘ λ³ν
    tr_idx = train_idx[0].tolist()
    test_idx = test_idx[0].tolist()

    print(f'length of tr_idx :  {len(tr_idx)}')
    print(f'length of test_idx :  {len(test_idx)}\n')


    tr_imgs_list = []
    test_imgs_list = []


    for i in tr_idx :
        tr_img = imgs[i]
        tr_imgs_list.append(tr_img)


    for i in test_idx:
        test_img = imgs[i]
        test_imgs_list.append(test_img)
        
    print(f'Quantity of Train data :  {len(tr_imgs_list)}')
    print(f'Quantity of Test data :  {len(test_imgs_list)}\n')



    ################### π Labels Split #######################
    ## labels

    tr_labels_list = []
    test_labels_list = []

    for i in tr_idx :
        tr_label = labels[i]
        tr_labels_list.append(tr_label)


    for i in test_idx :
        test_label = labels[i]
        test_labels_list.append(test_label)


    print(f'Quantity of Train label :  {len(tr_labels_list)}')
    print(f'Quantity of Test label :  {len(test_labels_list)}\n')


    print('\nβ¨[Info msg] Check whether they are matched')
    print(f'train : {tr_imgs_list == tr_labels_list}')
    print(f'test : {test_imgs_list == test_labels_list}\n')

    ############################################### π ann_file (meta_file) generation π##########################################
    ## Meta File : txt νμΌ μμ± (Annotationμ μΈ κ²)

    print('\nβ¨[Info msg] Start generating annotation file (Meta File) for Middle Format \n')
    # listμ λ΄κ²¨ μλ filenameμ νμ©νμ¬ pd.Dataframe μμ±
    tr_df = pd.DataFrame({'tr_filename' : tr_imgs_list})
    test_df = pd.DataFrame({'test_filename': test_imgs_list, 'test_labels_name' : test_labels_list})


    # # κΈ°μ‘΄ μΈλ±μ€(annotation νμΌ) μ­μ  
    # # Cross Validation μ­ν 
    # os.system('rm ./100_data/train.txt')
    # os.system('rm ./100_data/val.txt')


    # Annotation νμΌμ© txt νμΌ μμ±
    # args.root_dir / args.dataset_dir / args.tr or val or test
    tr_df['tr_filename'].to_csv(f'{args.root_dir}/{args.dataset_dir}/{args.tr_anno_name}', index=False, header=False)
    test_df['test_filename'].to_csv(f'{args.root_dir}/{args.dataset_dir}/{args.test_anno_name}', index=False, header=False)
    print('β¨[Info msg] Generating annotation file (Meta File) Complete')
    print(f'β¨[Info msg] Saving ... --> {args.root_dir}/{args.dataset_dir}/{args.tr_anno_name}')
    print(f'β¨[Info msg] Saving ... --> {args.root_dir}/{args.dataset_dir}/{args.test_anno_name} \n')


    # txt λ΄μ μλ μΈμλ€μ λ¦¬μ€νΈ ννλ‘ λΆλ¬μ΄
    print('β¨[Info msg] Check filenames in the annotation file\n')
    image_tlist = mmcv.list_from_file(f'{args.root_dir}/{args.dataset_dir}/{args.tr_anno_name}')
    image_test_list = mmcv.list_from_file(f'{args.root_dir}/{args.dataset_dir}/{args.test_anno_name}')


    print(f'Length of total Train_imgs names : {len(image_tlist[:])}')
    print(f'Train 5 imgs names : \n {image_tlist[:5]}\n')
    print(f'Length of total Test_imgs names : {len(image_test_list[:])}')
    print(f'Test 5 imgs names : \n {image_test_list[:5]}') 


if __name__ =='__main__':
    main()
