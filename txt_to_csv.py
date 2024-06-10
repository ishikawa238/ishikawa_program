# テキストファイルをcsvに変換する
import csv
import os
import re

input_file_path = os.path.join('test_csv.txt')
output_file_path = os.path.join('result_0919_gvem.csv')

# テキストファイルで使用される区切り記号を定義
delimiter = ','

# 指定された区切り記号でテキストファイルを開く
with open('input.txt', 'r') as f:
    reader = csv.reader(f, delimiter=delimiter)

# テキストファイルを読み取る
with open('input_file.txt', 'r') as infile:
# 指定された区切り文字でCSVリーダーオブジェクトを作成する
    reader = csv.reader(infile, delimiter='|')

# データの各行をループして出力する
for row in reader:
    print(row)

# 新しいファイルオブジェクトを作成する
with open('output.csv', 'w', newline='') as csvfile:
# ライターオブジェクトを作成する
    writer = csv.writer(csvfile)

#各データ行をループしてCSVファイルに書き込む
for row in reader:
    writer.writerow(row)

# ファイルとライターオブジェクトの両方を閉じる
csvfile.close()
writer.close()