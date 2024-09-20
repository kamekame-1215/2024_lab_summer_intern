# 2024年 夏のインターンシップ 深層学習を用いたオブジェクト検知

## 目次
- [はじめに](#はじめに)　
- [機械学習のセットアップ](#機械学習のセットアップ)
- [Dockerコンテナの起動](#dockerコンテナの起動)
- [OpenPCDetのインストール](#openpcdetのインストール)
- [データセットの準備](#データセットの準備)
- [OpenPCDetのデモプログラムの実行](#データセットの準備)
- [OpenPCDetを用いた3次元点群データからの学習](#openpcdetを用いた3次元点群データからの学習)
- [エラーが発生し、解決しない時](#エラーが発生し解決しない場合)
- [付録](#付録)


## はじめに

本ドキュメントは、2024年の夏インターンシップにて実施する、「深層学習を用いたオブジェクト検知」のための資料となります。<br>
ご不明点などありましたら何でも質問お願いいたします。

## 機械学習のセットアップ

本パッケージで扱う機械学習手法の学習用、およびテスト用プログラムは全て[OpenPCDet](https://github.com/open-mmlab/OpenPCDet/)というオープンソースプロジェクトに含まれている。

使用するサーバに機械学習環境を構築する大まかな手順は以下の通りとなっている。

1. Dockerコンテナの起動
2. OpenPCDetのイントール
3. OpenPCDetのdemoプログラムの実行

## Dockerコンテナの起動

本研究室では、開発環境を管理するために[Docker](https://www.docker.com/)を用いている。<br>
Dockerとは、インフラ関係やDevOps界隈で注目されている技術の1つであり、Docker社が開発しているコンテナ型の仮想環境の作成、配布、実行するためのプラットフォームである(詳細は[こちら](https://www.docker.com/resources/what-container/))。

今回は既に必要なものがインストールされてあるDockerイメージを用いる。<br>
使用するイメージは`summer-intern`である。

まず始めに、Dockerイメージからコンテナを作成し、入るために以下のコマンドを実行する。<br>
```
$ xhost local:
$ docker run -it --rm --gpus all -e DISPLAY=$DISPLAY -v /mnt/hdd_8tb_01/intern/[ユーザID]:/mnt/workspace -v /tmp/.X11-unix/:/tmp/.X11-unix summer-intern
```

次に、以下のコマンドを実行し、カーソルを追う目が表示されることを確認する。
```text
# xeyes
```

## OpenPCDetのインストール

次に、作業ディレクトリにOpenPCDetをインストールする。
今回は既にインストール済みであり、インストールは以下のコマンドで行なった。
```
$ git clone https://github.com/open-mmlab/OpenPCDet.git
```
この後、Dockerコンテナに入り、以下のコマンドで必要なライブラリなどをインストールする。
```
$ docker run -it --rm --gpus all -e DISPLAY=$DISPLAY -v /mnt/hdd_8tb_01/intern/[ユーザID]:/mnt/workspace -v /tmp/.X11-unix/:/tmp/.X11-unix summer-intern
# cd /mnt/workspace/OpenPCDet
# pip install -r requirements.txt
# python setup.py develop
```

## データセットの準備

本インターンでは、[KITTI](https://www.cvlibs.net/datasets/kitti/)と呼ばれるデータセットを使用する。<br>
データは既にダウンロードし、`OpenPCDet/data/kitti`に配置してある。<br>
ディレクトリ構造などについては、[Getting Started](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md)を参照のこと。

## OpenPCDetのデモプログラムの実行

`OpenPCDet/tools/`ディレクトリにある`demo.py`を実行することで簡単な動作確認(検出結果の可視化)を行うことが可能である。詳細は[Quick Demo](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/DEMO.md)を参照のこと。<br>
以下にKITTIデータセットを用いた一例を示す。なお、自分で取得したデータを用いる際はQuick Demoにある通り、座標系をKITTIデータセットのフォーマットに揃え、Intensityを正規化する必要があることに注意する。<br>

### 実行例

```
# cd /mnt/workspace/OpenPCDet/tools
# python demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt /mnt/workspace/OpenPCDet/data/ckpt/pv_rcnn_8369.pth --data_path /mnt/workspace/OpenPCDet/data/kitti/training/velodyne/000000.bin
```
以上のように、使用するモデルの設定ファイル(yamlファイル)、学習済みモデル(pthファイル)、3次元点群データ(binファイル、またはbinファイルが格納されているフォルダ)を指定して、`OpenPCDet/tools`ディレクトリ上で実行する。<br>
このプログラムを使用することで、各モデルの検出結果を可視化、比較することができる。また実行例では`pv_rcnn`というモデルを用いたが、その他の深層学習モデルもあるため、OpenPCDetのGitHubを参考に、様々なモデルを試してみることをおすすめする。

## OpenPCDetを用いた3次元点群データからの学習
### 学習パラメータの変更
OpenPCDetの学習モデルの設定ファイルは、`OpenPCDet/tools/cfgs/kitti_models`格納されている。
今回特に使用するパラメータは、
```
NUM EPOCHS: 80
```
であり、このパラメータは何回訓練を行うかを決めるものある。数が大きいほど精度が上がるとされているが、その分時間がかかる上、多すぎると過学習を引き起こす可能性がある。<br>
OpenPCDetでは、後から追加で学習させることも可能なので、最初は低い値から試すことをおすすめする。<br>
また、Epoch数は学習実行時のコマンドのオプションでも指定することができる(詳細は下記で説明)。

### 学習
学習に使用するデータを`OpenPCDet/data/kitti/training/`ディレクトリに配置する。今回は既に配置済みであり、データは以下のディレクトリに配置した。

学習は以下の手順で実行する。

1. データ情報の生成
```
# cd /mnt/workspace/OpenPCDet
# python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

2. 生成したデータ情報を用いて学習
```
# cd tools
# python train.py --cfg_file [使用するモデルの設定ファイル(yamlファイル)]
```

また、Epoch数は設定ファイル内を変更せず、コマンドのオプションとしても設定できる。
```
# python train.py --cfg_file [使用するモデルの設定ファイル]　--epochs [Epoch数]
```

学習後、`OpenPCDet/output/kitti_models/[設定ファイル名]/.../ckpt`ディレクトリに各エポックの学習済みモデル(pthファイル)が格納される。

### オプションについて
OpenPCDetには先述したEpoch数のオプションの他にも多数のオプションが用意されているため、時間があれば色々試してみることをおすすめする。<br>
オプションについては`OpenPCDet/tools/train.py`のコードに簡易的な説明が記載されている。

## エラーが発生し、解決しない場合
### GitHubで調べる
GitHubのOpenPCDetのリポジトリには様々なエラーがIssueとして挙げられているため、エラー文で検索を行えば、似たような問題に直面した際の情報が見れる可能性がある。

### ChatGPTに聞いてみる
ChatGPTでエラー文をコピー&ペーストし、説明を求めれば解説をしてくれる。これを元にエラーが解決される可能性がある。

## 付録
以下に我々が環境構築、動作確認の際に出力されたエラーとその対策を記載しておきます。

### Setupでのエラー

#### 実行コマンド
```
python setup.py develop
```
#### 出力エラー
```
packaging.version.InvalidVersion: Invalid version: '0.6.0+'
```
#### 対策
`OpenPCDet/setup.py`の以下の箇所を変更。<br><br>
修正前
```
if __name__ == '__main__':
    version = '0.6.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'pcdet/version.py')
```
修正後
```
if __name__ == '__main__':
    version = '0.6.0'
    write_version_to_file(version, 'pcdet/version.py')
```

### データ情報生成時エラー
#### 実行コマンド
```
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
#### 出力エラー
```
AssertionError
```
#### 対策
`data/kitti/ImageSets`に格納されているtxtファイル(train, test, val)内に記載されている0埋め6桁の数字が、`data/kitti/training`に格納されている点群のファイル名などと対応していないのが原因。<br>
各txtファイルの中身は以下のようにする必要がある。

- train, val.txt <br>
データセットをそれぞれモデルの学習用、テスト用に分割するために用いる。<br>
分割の比率は8:2、7:3にするのが主流で、`sklearn`の`train_test_split()`関数を使用するのがおすすめ。<br>

- test.txt <br>
モデルを評価するために用いる。<br>
OpenPCDetでは、`data/kitti/testing`に格納されているファイル名が全て格納されている必要がある。

※ OpenPCDetのtxtファイル名が目的と異なっていることに注意する。

### 学習時のエラー

#### 実行コマンド
```
python train.py --cfg_file {モデルの設定ファイル}
```
#### エラー文
```
KeyError: 'road_plane'
```

#### 対策
`OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml`の以下の箇所を変更。<br><br>

変更前
```
USE_ROAD_PLANE: True
```
変更後
```
USE_ROAD_PLANE: False
```
