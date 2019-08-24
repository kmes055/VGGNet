
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
    for i in file_list:
        key = i[2:6]  #파일명의 마지막 4자리수를 key로 사용
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
