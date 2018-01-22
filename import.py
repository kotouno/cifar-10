# -*- coding: utf-8 -*-
# モジュールのインポート
# ローカルファイルやディレクトリにアクセスするために必要
import os
# sys.argvで使用
import sys
# 色々な計算に便利
import numpy as np
# numpy配列から画像に変換してくれる便利なやつ
from scipy.misc import toimage
# 結果をグラフに起こしてくれるよ
import matplotlib.pyplot as plt

# kerasモジュールのインポート（pythonのdeepLearningライブラリTensorflowを更に簡単にまとめたライブラリ。なのでTensorflowでできないことは基本的にできない。バックエンドを変えればCNTK，Theanoにも対応。）
# 10のクラスにラベル付けされた，50000枚の32x32訓練用カラー画像，10000枚のテスト用画像のデータセット．中身（飛行機、車、鳥、猫、鹿、犬、カエル、馬、船、トラック）
from keras.datasets import cifar10
# 学習モデル（学習部分）を簡単に実装できるようになる！
from keras.models import Sequential
# レイヤーモジュールを使ってモデルを作成していく。
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
# np_utils.to_categoricalを用いてone-hotエンコーディング形式に変換。３種類なら(0, 0, 1)みたいに勝手にやってくれる。
from keras.utils import np_utils
# 作成したモデルを可視化する。
from keras.utils import plot_model
# 計算を早く行うためのバッチ処理など、画像の前処理を行ってくれる。画像の水増しに用いる！！！
from keras.preprocessing.image import ImageDataGenerator
# モデルをjson配列で表現してくれる。
from keras.models import model_from_json

# 学習済みモデルからの読み込み
result_dir = sys.argv[1]
model_file = os.path.join(result_dir, 'model.json')
weight_file = os.path.join(result_dir, 'model.h5')
with open(model_file, 'r') as fp:
    model = model_from_json(fp.read())
model.load_weights(weight_file)
model.summary()
