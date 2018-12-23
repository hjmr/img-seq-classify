# -*- coding: utf-8 -*-


class Config:
    # -------------------------------------
    #  クラス数
    # -------------------------------------
    NUM_CLASSES = 4

    # -------------------------------------
    #  入力画像の大きさ
    # -------------------------------------
    # 画像の1辺の画素数
    IMAGE_SIZE = 128
    # モノクロ？
    IMAGE_MONO = False

    # -------------------------------------
    #  畳み込み層の情報
    # -------------------------------------
    # フィルタの大きさ
    CONV_SIZE = 5
    # 第1層目のマップ数
    CONV1_OUT_CHANNELS = 16
    # 第2層目のマップ数
    CONV2_OUT_CHANNELS = 32

    # -------------------------------------
    #  全結合層の情報
    # -------------------------------------
    # 第１層目のニューロン数は畳み込み層の出力から自動的に決まるので未指定
    # 第２層目のニューロン数
    NUM_HIDDEN_NEURONS2 = 2048
