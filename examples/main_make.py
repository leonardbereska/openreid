# deepfashion without full body images

import os
import glob
import cv2
import os

path_to_root = '/Users/leonardbereska/myroot/'
path_to_data = '/Users/leonardbereska/PycharmProjects/master/reid/examples/data/'


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    else:
        return False




def get_img(path):
    return glob.glob(path+'/*')


def get_id(img_path):
    i = img_path.split('/')[-1]
    return i.split('_')[0]


def get_imgid(img_path):
    i = img_path.split('/')[-1]
    return i.split('.')[0]

def get_i(id, from_list, skip_how_many=0):
    for i in from_list:
        time = skip_how_many
        if get_id(i) == id:
            if time == 0:
                return i
            time -= 1


def get_i_list(id, from_list):
    imgs = []
    for i in from_list:
        if get_id(i) == id:
            imgs.append(i)
    return imgs


def save_imglist(img_list, save_dir, path=path_to_data):
    save_path = path + save_dir
    make_dir(save_path)
    for i in img_list:
        img_name = i.split('/')[-1]
        image = cv2.imread(i)
        img_name = img_name.split('.')[0] + '.png'
        cv2.imwrite(save_path+'/'+img_name, image)


def get_img_from_id(img_id, png='png', img_dir='df/'):
    return path_to_data + img_dir + img_id + '.' + png


def to_id(img_list):
    return [get_id(i) for i in img_list]


def to_img(id_list, png='png', img_dir='df/'):
    return [get_img_from_id(i, png, img_dir) for i in id_list]

def get_first_id(id_list):
    id_list_new = []
    current_id = ''
    for i in sorted(id_list):
        id_ = i.split('_')[0]
        if id_ != current_id:
            id_list_new.append(i)
        current_id = id_
    return id_list_new

def id(i):
    return i.split('_')[0]


df = path_to_data + 'df'
df_fb = path_to_data + 'df_fullbody'
df_no = path_to_data + 'df_nofull'
df_gen = path_to_data + 'gen_train'
# from collections import Counter
# c = Counter(id_fb)

# id_fb_only2 = [i for i in id_fb if c[i] >=2]
# id_fb_only2 = list(set(id_fb_only2))

# id_double = [i for i in img_fb if get_id(i) in id_fb_only2]  # 675

img = get_img(df)
img_gen = get_img(df_gen)
# img_fb = get_img(df_fb)

id_all = [get_imgid(i) for i in img]
id_gen = [get_imgid(i) for i in img_gen]
select_id = [id(i) for i in id_gen]
select_id = (set(select_id))
len(select_id)

id_orig = [id(i) for i in id_all if id(i) in select_id]
set2 = set(id_orig)
len(set2)
id_gen2 = [i for i in id_gen if id(i) in set2]
# set1 = set(id_gen)
# id_orig = [i for i in id_all if id(i) in set1]
# id_gen = [i for i in id_gen if id(i) in set1]
# len(set1)
img_save = to_img(id_gen2, png='jpg', img_dir='gen_train/')
save_imglist(img_save, 'subset')

# id_fb = [get_imgid(i) for i in img_fb]
# id_fb1 = get_first_id(id_fb)
# img_fb1 = to_img(id_fb1)
# save_imglist(img_fb1, 'df_fb2')
#
# i_fb = [id(i) for i in id_fb]
# from collections import Counter
# c_fb = Counter(i_fb)
# id_list = [id_ for id_ in id_fb if c_fb[id(id_)] >= 2]
# n_id = len(set([i for i in i_fb if c_fb[i]>=2]))
# img_list = to_img(id_list)
# save_imglist(img_list, 'df_fb_many')

# diff = list(sorted(set(id_).difference(set(id_fb1))))
# id_diff = get_first_id(diff)
# select_id = [id(i) for i in id_fb1]
# id_sel = [i for i in id_diff if id(i) in select_id]
# reverse_sel = [id(i) for i in id_sel]
# id_sel_rev = [i for i in id_fb1 if id(i) in reverse_sel]
# final_sel = [id(i) for i in id_sel_rev]
# id_sel = [i for i in id_diff if id(i) in final_sel]
#
# final_sel_test = [id(i) for i in id_sel]
# assert final_sel == final_sel_test
#
# img_list = to_img(id_sel)
# save_imglist(img_list, 'df_nofb')
# testy = [i for i in id_fb1 if id(i) in final_sel]
#
# img_list = to_img(testy)
# save_imglist(img_list, 'df_fb')


# onlyfb = [get_i(id, from_list=img_fb) for id in id_fb_only2]


# onlyfb = [get_i(id, from_list=img_fb) for id in id_allfb]
# save_imglist(onlyfb, 'df_fb1')
# all_imgid = [get_imgid(i) for i in img]
# all_fbid = [get_imgid(i) for i in img_fb]
#
# img_fb1 = get_img(path_to_root + 'df_fb1')
# all_fb1id = [get_imgid(i) for i in img_fb1]
#
# diff = list(sorted(set(all_imgid).difference(set(all_fb1id))))
#
# diff = [get_img_from_id(i) for i in diff]
#
#
# to_img(all_fb1id)
# [get_img_from_id(i) for i in all_fbid]
#
# id_diff = [get_imgid(i) for i in diff]
# # for i in
# id_diff1 = [get_imgid(i) for i in diff]
# all_fb_img = [i for id_ in id_fb_only2 for i in get_i_list(id_, from_list=img_fb)]
# save_imglist(diff, 'df_nofb1')



# all_fb_img = [get_i(id, from_list=img_fb) for id in id_allfb]
# save_imglist(all_fb_img, 'df_allfb')


# save_imglist(onlyfb, 'onlyfb')
