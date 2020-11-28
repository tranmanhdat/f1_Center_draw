import os
import shutil

fn = 'nuadau'

path_labeled = 'sanphai/images2/'+fn+'/'
path_raw = '/media/fitmta/DULIEU/MinhTu/CDS/Center/images/'
dst_dir = 'sanphai/images/'+fn+'/'

for filename in os.listdir(path_labeled):
    if 'lanetrai' in filename:
        continue
    
    name_id_only = filename.split('__')[-1]

    print(name_id_only)

    if os.path.exists(path_raw+name_id_only):
        shutil.move(path_raw+name_id_only, dst_dir+filename)


# folder = 'images_draw'
# fn = 'thang_lanetrai'
# path_labeled = '/media/fitmta/DULIEU/MinhTu/CDS/f1_Center_draw/data/'+folder+'/'
# dst_dir = '/media/fitmta/DULIEU/MinhTu/CDS/f1_Center_draw/sanphai/'+folder+'/'+fn+'/'

# for filename in os.listdir(path_labeled):

#     if fn == 'phai_lanephai':
#         if filename[0] == 'r':
#             shutil.move(path_labeled+filename, dst_dir+filename)

#     elif fn == 'thang_lanetrai':
#         if filename[0] == 's' and '__lanetrai' in filename:
#             shutil.move(path_labeled+filename, dst_dir+filename)

#     elif fn == 'thang_lanephai':
#         if filename[0] == 's' and '__lanetrai' not in filename:
#             shutil.move(path_labeled+filename, dst_dir+filename)

#     elif fn == 'nuadau':
#         if '__' not in filename:
#             shutil.move(path_labeled+filename, dst_dir+filename)
