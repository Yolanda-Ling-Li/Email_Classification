# import os
# from chardet.universaldetector import UniversalDetector
#
#
# # def get_encode_info(file):
# #     with open(file, 'rb') as f:
# #         detector = UniversalDetector()
# #         for line in f.readlines():
# #             detector.feed(line)
# #             if detector.done:
# #                 break
# #         detector.close()
# #         return detector.result['encoding']
# #
# #
# # def convert_encode2utf8(file):
# #     f = open(file, 'rb')
# #     file_content = f.read()
# #     encode_info = get_encode_info(file)
# #     if encode_info == 'utf-8':
# #         return
# #     file_decode = file_content.decode(encode_info, 'ignore')
# #     file_encode = file_decode.encode('utf-8')
# #     f.close()
# #     f = open(file, 'wb')
# #     f.write(file_encode)
# #     f.close()
#
# # 把data讀進來
# def read_file():
#     names = os.listdir('/Users/bibi/Documents/CSC522/project/maildir')
#     X = []
#     Y = []
#
#     for idx, name in enumerate(names):
#         allList = os.walk('/Users/bibi/Documents/CSC522/project/maildir/' + name)
#         for root, dirs, files in allList:
#             for file in files:
#                 file_name = root + '/' + file
#
#                 # transform encoding to utf-8
#                 # convert_encode2utf8(file_name)
#
#                 f = open(file_name, 'rb')
#                 X.append(f.read())
#                 f.close()
#                 Y.append(idx)
#     return X, Y
#
#
# def split_space(X):
#     for idx, x in enumerate(X):
#         X[idx] = x.split(" ")
#     return X
#
# def run_tfidf():
#