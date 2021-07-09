# 轻量级文字识别技术创新大赛

## 数据处理
    1.根据所有的训练数据统计字库
    2.统计数据中各个字符的占比（分析数据是否均衡），存在大量只出现一次的字符
    3.离线数据拓增（仿射、滤波、模糊、锐化）（采用离线数据托增一方面是为了缓解数据不均衡）
    4.裁剪训练数据，得到单个字符的训练数据，结合所有的单个字符数据进行90度旋转，扩增训练数据
    5.待模型收敛后，使用模型清洗训练数据


## 模型优化
    1.分别使用PaddleOCR提供的几个常见模型进行迭代，最终确定crnn网络，模型较小且收敛较快
    2.根据PP-OCR: A Practical Ultra Lightweight OCR System 论文的实验部分确定crnn的网络结构以及通道数
    3.在两个bilstm的输出加入一个add操作
    4.crnn网络加入attention分支作为辅助loss迭代模型,固定backbone和neck优化head
    5.固定backbone和neck优化crnn的head

## 代码实现
    1.修改neck部分，实现bilstm输出特征的融合
    2.加入MTLHead这个选择，以实现ctc+att联合训练

##  快速开始
    1.训练
    
    sh train.sh
   
   2.测试

    python3 tools/infer_rec.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml -o Global.pretrained_model=./checkpoint/best_accuracy Global.load_static_weights=false Global.infer_img=./B榜测试数据集/TestBImages



    

