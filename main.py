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

# cifar10を画像データに変換、resultディレクトリに保存。
def plot_cifar10(X, y, result_dir):
    plt.figure()

    # 画像を描画
    nclasses = 10
    pos = 1
    for targetClass in range(nclasses):
        targetIdx = []
        # クラスclassIDの画像のインデックスリストを取得
        for i in range(len(y)):
            if y[i][0] == targetClass:
                targetIdx.append(i)

        # 各クラスからランダムに選んだ最初の10個の画像を描画
        np.random.shuffle(targetIdx)
        for idx in targetIdx[:10]:
            img = toimage(X[idx])
            plt.subplot(10, 10, pos)
            plt.imshow(img)
            plt.axis('off')
            pos += 1

    plt.savefig(os.path.join(result_dir, 'plot.png'))

# テストセットの損失関数、テストセットの認識精度、バリデーションセットの損失関数、バリデーションセットの認識精度を記録
# 画像データセットはトレーニングセット、バリデーションセット、テストセットの3つ。
def save_history(history, result_file):
    # それぞれを結果から記録
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    # 書き込み用にファイルを開き、書き込んでいく。
    with open(result_file, "w") as fp:
        # \tは水平タブ、\nは改行。
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))

# 実行！
if __name__ == '__main__':
    # ターミナルからpythonファイルを実行した時、引数の数を確認。
    # 引数が4つでない場合の処理。
    if len(sys.argv) != 4:
        print("usage: python cifar10.py [nb_epoch] [use_data_augmentation (True or False)] [result_dir]")
        exit(1)

    # 引数が4つの時の処理。
    # これから使う変数の設定。
    # エポック数（何回訓練を繰り返すか）設定
    nb_epoch = int(sys.argv[1])
    # データ拡張（画像の水増し）するかどうか
    data_augmentation = True if sys.argv[2] == "True" else False
    # 結果を保存するディレクトリ
    result_dir = sys.argv[3]
    # もしこのファイルが実行された時、result_dirが無ければ作成。
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # 各種設定をプリント
    print("nb_epoch:", nb_epoch)
    print("data_augmentation:", data_augmentation)
    print("result_dir:", result_dir)

    # バッチサイズ
    batch_size = 128
    # 分類する画像の種類の数。nbってナンバー？
    nb_classes = 10

    # 入力画像の次元（32 * 32）
    img_rows, img_cols = 32, 32

    # チャネル数（RGBなので3）
    img_channels = 3

    # CIFAR-10データをロード
    # (nb_samples, nb_rows, nb_cols, nb_channel) = tf
    # tfファイル？？
    # オリジナルにしたい時はここを変える
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # ランダムに画像をプロット（resultディレクトリに保存）
    plot_cifar10(X_train, y_train, result_dir)

    # 画素値を0-1に変換（32 * 32 * 3）rgbを0-1に
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    # クラスラベル（0-9）をone-hotエンコーディング形式に変換。今回は10種類の画像だから0-9
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # CNN（Convolution Neural Network）を構築。モデル
    model = Sequential()

    # 今回は画像データなので2次元入力（使用するカーネルの数、畳み込みカーネルの幅と高さ、padding?、32*32*3）
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
    # ずーっと0で途中からクン！ってやつ。
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    # ダウンスケールする係数を決める。2つの整数のタプル（垂直，水平）。(2, 2) は画像をそれぞれの次元で半分にする。
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 0と1の間の浮動小数点数．入力ユニットをドロップする割合。
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 入力を平滑化する．バッチサイズに影響されない
    model.add(Flatten())
    # 正の整数，出力空間の次元数
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    # ソフトマックス関数はそれぞれの可能性を合計１になるようにしてくれるやつ。
    model.add(Activation('softmax'))

    # 訓練課程の設定。categorical_crossentropyは悪さを表す指標。0が一番いい。optimizer（最適化）よくわからん。metrics（評価関数）モデルの性能を評価するために使う。
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # モデルのサマリ（概要）を表示
    model.summary()
    # show_shapes:グラフ中に出力のshapeを出力するかどうか。
    plot_model(model, show_shapes=True, to_file=os.path.join(result_dir, 'model.png'))

    # # 学習済みモデルからの読み込み
    # model_file = os.path.join(result_dir, 'model.json')
    # weight_file = os.path.join(result_dir, 'model.h5')
    # with open(model_file, 'r') as fp:
    #     model = model_from_json(fp.read())
    # model.load_weights(weight_file)
    # model.summary()

    # 訓練
    # データ拡張がFalseの場合実行
    if not data_augmentation:
        print('Not using data augmentation')
        # 固定のエポック数でモデルを訓練する。
        # x:入力データ（numpy配列）
        # y:ラベル（numpy配列）
        # バッチサイズ
        # verbose:1の場合はログをプログレスバーで標準出力
        # バリデーションデータ
        # 各エポックにおいてサンプルをシャッフルするかどうか
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            verbose=1,
                            validation_data=(X_test, Y_test),
                            shuffle=True)
    else:
        print('Using real-time data augmentation')

        # 訓練データを生成するジェネレータ
        # ZCA白色化（グーグルのあの怖いやつみたくなる）を適用。ランダムに水平シフトする範囲。ランダムに垂直シフトする範囲。
        train_datagen = ImageDataGenerator(zca_whitening=True, width_shift_range=0.1, height_shift_range=0.1)
        # モデルを訓練。
        train_datagen.fit(X_train)
        # numpyデータとラベルのarrayを受け取り，拡張/正規化したデータのバッチを生成。無限ループ内で，無限にバッチを生成。
        train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)

        # テストデータを生成するジェネレータ
        # 画像のランダムシフトは必要ない？
        test_datagen = ImageDataGenerator(zca_whitening=True)
        test_datagen.fit(X_test)
        test_generator = test_datagen.flow(X_test, Y_test)

        # ジェネレータから生成される画像を使って学習
        # 本来は好ましくないがテストデータをバリデーションデータとして使う
        # validation_dataにジェネレータを使うときはnb_val_samplesを指定する必要あり
        # TODO: 毎エポックで生成するのは無駄か？
        # Python のジェネレータにより，バッチごとに生成されるデータでモデルを学習。ジェネレータは効率化のために，モデルを並列に実行。 たとえば，これを使えば CPU 上でリアルタイムに画像データを拡大しながら，それと並行してGPU上でモデルを学習できる。
        history = model.fit_generator(train_generator,
                                      samples_per_epoch=X_train.shape[0],
                                      nb_epoch=nb_epoch,
                                      validation_data=test_generator,
                                      nb_val_samples=X_test.shape[0])

    # 学習したモデルと重みと履歴の保存
    # このjson使えばいろいろできそう。
    model_json = model.to_json()
    # jsonファイルの書き込み
    with open(os.path.join(result_dir, 'model.json'), 'w') as json_file:
        json_file.write(model_json)
    # モデルの重みをHDF5形式のファイルに保存。
    model.save_weights(os.path.join(result_dir, 'model.h5'))
    # あらかじめ準備していたsave_history関数で損失関数や正確性を記録。
    save_history(history, os.path.join(result_dir, 'history.txt'))

    # モデルの評価
    # 水増しなし
    if not data_augmentation:
        # バッチごとにある入力データにおける損失値を計算。出力なし。
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    # 水増しあり
    else:
        # 学習は白色化した画像を使ったので評価でも白色化したデータで評価する
        # ジェネレータのデータによってモデルを評価。
        loss, acc = model.evaluate_generator(test_generator, val_samples=X_test.shape[0])

    print('Test loss:', loss)
    print('Test acc:', acc)
