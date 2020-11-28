import os
import numpy as np

batch_size = 256*1
epochs = 150
learning_rate = 0.001
train_size = 0.8

san = 'santrai' # santrai
part = 'trai_lanephai' # nuadau
                # thang_lanephai, phai_lanephai, thang_lanetrai (san phai)
                # thang_lanephai, trai_lanephai, trai_lanetrai (san trai)

lanepath = part.split('_')[-1] # '', 'lanephai', 'lanetrai'

HEIGHT, WIDTH = 135, 320
model_num_postfix = 'santrai_T6_1'
line_from_bottom = 70
line1 = 240-line_from_bottom #90=160-70

mixed_input = False

draw_guide_append = True



if mixed_input:
    model_name = 'mixed__{}x{}'.format(WIDTH, HEIGHT)
    data_path_X_img = 'data/x_mixed_img.npy'
    data_path_X_route = 'data/x_mixed_route.npy'
    data_path_X_lane = 'data/x_mixed_lane.npy'
    data_path_y = 'data/y_mixed.npy'
    INPUT_SHAPE = (HEIGHT, WIDTH, 3)
else:
    postfix_name = '320x{}__{}+40__{}'.format(HEIGHT+40, HEIGHT, model_num_postfix)
    model_name = 'draw_more__{}'.format(postfix_name)
    data_dir = 'data/{}'.format(postfix_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_path_x = '{}/x_more.npy'.format(data_dir)
    data_path_y = '{}/y_more.npy'.format(data_dir)
    center_save_path = '{}/center.json'.format(data_dir)


if draw_guide_append:
    INPUT_SHAPE = (HEIGHT+40, WIDTH, 3)

    org_thang = (160, 20)
    org_phai = (300, 20)
    org_trai = (20, 20)

    pts_lanephai = [(org_thang[0]+40, 0), (org_phai[0]-40, 40)]
    pts_lanetrai = [(org_trai[0]+40, 0), (org_thang[0]-40,40)]
else:
    INPUT_SHAPE = (HEIGHT, WIDTH, 3)

    # org_thang = (160, HEIGHT-20)
    # org_phai = (250, HEIGHT-20)
    # org_trai = (70, HEIGHT-20)
    org_thang = (160, HEIGHT-50)
    org_phai = (295, HEIGHT-50)
    org_trai = (25, HEIGHT-50)

    pts_lanephai = [(WIDTH//2+20, HEIGHT-30), (WIDTH,HEIGHT)]
    pts_lanetrai = [(0, HEIGHT-30), (WIDTH//2-20,HEIGHT)]



if san == 'sanphai':
    routes_mapping = {'nuadau':0, 'thang':1, 'phai':2}
elif san =='santrai':
    routes_mapping = {'nuadau':0, 'thang':1, 'trai':2}
lanes_mapping = {'nuadau': 0, 'lanephai':1, 'lanetrai':2}


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels.
        eg: nb_classes = 6
            data = [[2, 3, 4, 0]]
    """
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]



path_raw_origin = san+'/images/'+part+'/'
path_raw_resized = san+'/images_resized/'+part+'/'
if draw_guide_append:
    path_draw = san+'/images_draw_more/'+part+'/'
else:
    path_draw = san+'/images_draw/'+part+'/'

path_labeled = san+'/images2/'
# path_move_labeled_not_progressed = san+'/images2__notprogressed/'
# path_move_raw_not_progressed = san+'/images_draw__notprogressed/'

if not os.path.exists(path_draw):
    os.makedirs(path_draw)
if not os.path.exists(path_labeled):
    os.makedirs(path_labeled)
# if not os.path.exists(path_move_labeled_not_progressed):
#     os.makedirs(path_move_labeled_not_progressed)
# if not os.path.exists(path_move_raw_not_progressed):
#     os.makedirs(path_move_raw_not_progressed)


