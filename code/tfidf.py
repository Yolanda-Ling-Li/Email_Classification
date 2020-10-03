import os

# 把data讀進來
# Y_class = os.listdir('/Users/bibi/Documents/CSC522/project/maildir')

k = os.walk('/Users/bibi/Documents/CSC522/project/maildir/presto-k/junk_e_mail')
for root, dirs, files in allList:
#   列出目前讀取到的路徑
  print("path：", root)
#   列出在這個路徑下讀取到的資料夾(第一層讀完才會讀第二層)
  print("directory：", dirs)
#   列出在這個路徑下讀取到的所有檔案
  print("file：", files)
print(k)

# f = open('/Users/bibi/Documents/CSC522/project/maildir/presto-k/junk_e_mail/1.', 'r')
# s = f.read()
# f.close()


# Y_class = 人名list  => 轉成數字索引
# X = 文章list
# X 去除特殊符號  扔到tfidf  去除停止詞

