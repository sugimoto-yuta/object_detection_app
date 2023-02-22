from torchvision import transforms
from PIL import Image
import numpy as np
from PIL import ImageDraw, ImageFont
import collections
from flask import Flask, request, render_template, redirect
import io
import base64
import pandas as pd
import torch


# 学習済みモデルを元に推論する
def predict(img):
    transform = transforms.ToTensor()
    x = transform(img)

    # 学習済みモデルを用意
    # model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    model = torch.load('./odfaster.pt')

    # 推論モードへ
    model.eval()

    # 推論
    y = model(x.unsqueeze(0))[0]
    return x, y

# クラスラベルのリストを用意
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', '人', '自転車', '自動車', 'バイク', '飛行機', 'バス',
    '電車', 'トラック', 'ボート', '信号機', '消火栓', 'N/A', '一時停止標識',
    'パーキングメーター', 'ベンチ', '鳥', '猫', '犬', '馬', '羊', '牛',
    '象', '熊', 'シマウマ', 'キリン', 'N/A', 'バックパック', '傘', 'N/A', 'N/A',
    'ハンドバック', 'ネクタイ', 'スーツケース', 'フリスビー', 'スキー板', 'スノーボード', 'ボール',
    '凧', 'バット', 'グローブ', 'スケートボード', 'サーフボード', 'テニスラケット',
    'ボトル', 'N/A', 'ワイングラス', 'カップ', 'フォーク', 'ナイフ', 'スプーン', 'ボウル',
    'バナナ', 'りんご', 'サンドウィッチ', 'オレンジ', 'ブロッコリー', 'にんじん', 'ホットドッグ', 'ピザ',
    'ドーナツ', 'ケーキ', 'イス', 'ソファ', '鉢植え', 'ベッド', 'N/A', 'テーブル',
    'N/A', 'N/A', 'トイレ', 'N/A', 'テレビ', 'ノートパソコン', 'マウス', 'リモコン', 'キーボード', '携帯電話',
    '電子レンジ', 'オーブン', 'トースター', 'シンク', '冷蔵庫', 'N/A', '本',
    '時計', '花瓶', 'はさみ', 'テディベア', 'ドライヤー', '歯ブラシ'
]

# 可視化をする関数
def visualize_results(input, output, threshold):
    image= input.permute(1, 2, 0).numpy()
    image = Image.fromarray((image*255).astype(np.uint8))

    boxes = output['boxes'].cpu().detach().numpy()
    labels = output['labels'].cpu().detach().numpy()

    if 'scores' in output.keys():
        scores = output['scores'].cpu().detach().numpy()
        boxes = boxes[scores > threshold]
        labels = labels[scores > threshold]

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('./NotoSansCJKjp-Bold.otf', 16)
    for box, label in zip(boxes, labels):
        # box
        draw.rectangle(box, outline='red')
        # label
        text = COCO_INSTANCE_CATEGORY_NAMES[label]
        w, h = font.getsize(text)
        draw.rectangle([box[0], box[1], box[0]+w, box[1]+h], fill='red')
        draw.text((box[0], box[1]), text, font=font, fill='white')

    return image, labels

# 一意なラベルリスト、出現回数の辞書、一意なラベルテキストリストを返す関数
def return_info(detection_list):
    label_list = list(set(detection_list))
    count_dict = collections.Counter(detection_list)
    dict = {}
    for key, _ in count_dict.items():
        dict[COCO_INSTANCE_CATEGORY_NAMES[key]] = count_dict[key]
    texts = [COCO_INSTANCE_CATEGORY_NAMES[label] for label in detection_list]
    return label_list, dict, list(dict.fromkeys(texts))

# 設定したクラスのみ検出して可視化する関数
def exclusive_visualize_results(input, output, threshold, applicable_label):
    image= input.permute(1, 2, 0).numpy()
    image = Image.fromarray((image*255).astype(np.uint8))

    boxes = output['boxes'].cpu().detach().numpy()
    labels = output['labels'].cpu().detach().numpy()

    if 'scores' in output.keys():
        scores = output['scores'].cpu().detach().numpy()
        boxes = boxes[scores > threshold]
        labels = labels[scores > threshold]

        applicable_boxes = []
        for box, label in zip(boxes, labels):
            if label == applicable_label:
                applicable_boxes.append(box)


    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('./NotoSansCJKjp-Bold.otf', 16)
    for box in applicable_boxes:
        # box
        draw.rectangle(box, outline='red')
        # label
        text = COCO_INSTANCE_CATEGORY_NAMES[applicable_label]
        w, h = font.getsize(text)
        draw.rectangle([box[0], box[1], box[0]+w, box[1]+h], fill='red')
        draw.text((box[0], box[1]), text, font=font, fill='white')

    return image

app = Flask(__name__)

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

# 拡張子が適切かどうかをチェック
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# result.htmlからexclusive.htmlの動作
@app.route('/exclusive', methods = ['POST'])
def exclusive_predicts():
    if request.method == 'POST':
        labelname = request.form.get('names')
        label_index = COCO_INSTANCE_CATEGORY_NAMES.index(labelname)
        exclusive_image = exclusive_visualize_results(x, y, 0.5, label_index)

        buf = io.BytesIO()
        exclusive_image.save(buf, 'png')
        # バイナリーデータをbase64でエンコードしてutf-8でデコード
        base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        # HTML側のの記述に合わせるために付帯情報を付与
        base64_data = f'data:image/png;base64,{base64_str}'

        # 変数展開(messageとnameの値がHTMLに渡される)
        return render_template('exclusive.html', image=base64_data, name=labelname)


# URLにアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'filename' not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        if file and allowed_file(file.filename):
            
            # 画像ファイルに対する処理
            # 画像書き込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')
            # 画像データをバッファに読み込む
            image.save(buf, 'png')

            # 変数をグローバル化
            global x, y, label_list, texts

            # 入力された画像に対して推論
            x, y = predict(image)

            # 検出後画像と検出ラベルを取得
            od_image, detection_list = visualize_results(x, y, 0.5)

            # 検出情報を取得
            label_list, dict, texts = return_info(detection_list)

            # 検出したもののデータフレームを作成
            df = pd.DataFrame.from_dict(dict, orient='index', columns=['検出数'])

            # 検出したもののリストの先頭名を取得
            detectionName_ = texts[0]

            # 元画像データをいったん削除
            del buf

            # 画像書き込み用バッファを確保
            buf2 = io.BytesIO()
            # 画像データをバッファに読み込む
            od_image.save(buf2, 'png')

            # バイナリーデータをbase64でエンコードしてutf-8でデコード
            base64_str = base64.b64encode(buf2.getvalue()).decode('utf-8')
            # HTML側のの記述に合わせるために付帯情報を付与
            base64_data = f'data:image/png;base64,{base64_str}'
            return render_template('result.html', detectionName=detectionName_, table=df.to_html(header='true'), image=base64_data, names=texts)

        return redirect(request.url)

    elif request.method == 'GET':
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)



