import os

def read_dataset(directory_path):
    """
    VOC Dataset 파일을 적절한 폴더에 옮긴 뒤 python으로 import하는 함수입니다.
    Directory에 있는 파일들의 이름을 구한 뒤, read_image 함수를 이용해 numpy array로
    바꿔 주시고, 유일한 key로 구분할 수 있는 Dictionary 등의 자료구조에 넣어 주세요.

    :param directory_path: 사진이 들어있는 폴더의 경로
    :return: 사진 파일을 key로 구분할 수 있는 자료구조
    """
    ###############################################################
    #                              1번                            #
    ###############################################################
    file_list = os.listdir(directory_path)
    file_dict = {}
    for i in file_list[:10]:
        key = i[2:6]  # 파일명의 마지막 4자리수를 key로 사용
        file_dict[key] = read_image(directory_path + '/' + i)
    return file_dict
    ###############################################################


def match_by_key(image_files, directory_path):
    """
    image file과 annotation file은 같은 파일명, 다른 확장자를 가졌습니다.
    파일명을 key로 가지는 자료구조를 가진 image_file와, 같은 key를 가지는
    annotation file을 서로 matching해주는 함수를 구현해주세요.
    * image와 annotation을 각각 다른 자료구조에 저장하셔도 됩니다.

    :param image_files: 파일 이름을 key로 가지는 자료구조
    :param directory_path: 이미지와 대응되는 annotation file들이
            들어있는 폴더의 경로
    :return: 사진 파일과 annotation을 같은 key로 구분할 수 있는 자료구조
    """
    ###############################################################
    #                              2번                            #
    ###############################################################
    pass
    ###############################################################


from imageio import imread
import numpy as np
def read_image(filepath):
    """
    Input으로 들어온 image file을 numpy array로 바꿔줍니다.

    :param filepath: image파일의 경로
    :return: image의 RGB값을 가지는 numpy Array 객체
    """
    ###############################################################
    #                              3번                            #
    ###############################################################
    img = imread(filepath)
    img = np.array(img)
    return img
    ###############################################################


def train_val_split(data, validation_ratio):
    """
     VOC dataset은 train과 test로만 나누어져 있습니다. 학습의 성능을 위해
    dataset을 train set과 validation set으로 나누는 함수를 구현해주세요.
    *주의: 각각의 사진은 서로 다른 해상도를 가지고 있습니다.
    :param data: data의 자료형은 다른 연구원들의 구현에 따라 달라집니다.
     교류를 통해 data의 타입을 요청하거나 파악해주세요. 다만, 이 data는
     각각 유일한 key로 호출할 수 있고, numpy array 객체이거나 annotation
     파일입니다.
    :param validation_ratio: float, [0-1].
    전체 train data 중 validation에 사용할 data의 비율을 나타냅니다. 0-1
    사이의 값이며, 해당하는 비율 만큼의 data를 validation set이 포함하게
    해 주세요.
    :return: X_train, X_val, y_train, y_val.
    X는 image에 해당하는 numpy data, y는 ground truth입니다. X, y는 유일한
    키로 대응되는 사진과 annotation file을 구할 수 있어야 하며, train set과
    validation set은 의도하는 비율대로 나누어져 있어야 합니다.
    """
    ###############################################################
    #                              4번                            #
    ###############################################################
    pass
    ###############################################################


def make_batch(X, y, batch_size):
    """
    실제로 Train을 수행할 땐 모든 사진을 메모리에 올리지 않고 적절한
    크기의 batch로 나눠서 연산을 수행합니다. batch size가 주어졌을 때
    X, y를 batch size 크기로 나눈 뒤, 적절한 자료구조에 넣어 주세요.
    List도 좋고, Numpy Array에 하나의 axis를 추가해도 좋으나, 구현 이후
    주석을 통해 output format을 명시해 주시면 좋을 것 같습니다.
    :param X: Array-like, (N, H, W, C).
    :param y: ground-truth. annotation 파일의 정보를 가지고 있습니다.
    :param batch_size: 하나의 batch에 들어갈 사진의 숫자입니다.
    :return: X_batch, y_batch 가 여러개 있는 자료구조.
    """
    ###############################################################
    #                              5번                            #
    ###############################################################
    pass
    ###############################################################


def read_annotation(path):
    """
    annotation file은 .xml 파일로서, bounding box, class, owner, source 등 다양한
    정보를 담고 있습니다. 이를 분석하고, 필요한 정보를 추출하는 함수를 구현하세요.
    output은 유일한 key로 구분하되, read_dataset 함수가 return한 자료구조와 같은 key를
    공유해야 합니다.
    추출할 정보의 범위는 연구자의 재량이나, bounding box/Class 는 반드시 포함해야 합니다.
    :param path: annotation file의 경로. .xml 형식
    :return: 유일한 key로 구분되는 annotation file 정보
    """
    ###############################################################
    #                              6번                            #
    ###############################################################
    pass
    ###############################################################

def import_in_one_function(directory_path, batch_size=128, validation_ratio=0.2):
    """
    모든 함수의 구현이 끝난 뒤, 위 함수들을 적절히 이용하여 모델의 training에
    사용할 수 있는 data type으로 만들어 주세요.
    자료구조는 무엇이든 좋으나, 각각의 X_batch는 batch_size만큼의 사진에 해당하는
    numpy array를 포함하고 있어야 합니다.
    :param directory_path: 데이터를 가져올 폴더의 path
    :return: X_train_batch, X_val_batch, y_train_batch, y_val_batch
    """
    ###############################################################
    #                              7번                            #
    ###############################################################
    pass
    ###############################################################

