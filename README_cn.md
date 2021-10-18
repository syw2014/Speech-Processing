# Speech Processing
# 项目介绍

语音信号处理，该仓库主要用来收录在学习过程中，已落地的算法模型，仓库不同于其他已开源项目，本项目不仅关注算法模型，同时也更加注重模型的实际落地，打通了部署的最后”一公里“，才能真正发挥算法模型的价值，而不仅停留在paper层面。

# 内容说明

目录结构

```bash
.
├── README_cn.md
├── README.md
├── deploy
│   ├── KeywordSpotting
│   ├── cc
│   ├── data
│   ├── release
│   └── release.zip
├── development.md
├── keyword-spotting
└── tf23
```

目录内容说明

- tf23， 利用tf23训练唤醒模型，**建议参考此仓库进行研发**
- keyword-spotting,利用tf2.0训练唤醒模型
- deploy
  - KeywordSpotting,利用vs2015 开发唤醒lib和应用
  - 其他文件夹，可不做参考

# 项目依赖

1. Tensorflow 2.3.2，模型开发和训练
2. TFlite ，用于模型量化及推理
3. librosa ，用于数据增强
4. wandb，用于训练过程记录
5. visual studio 2015,开发win下唤醒lib

# 使用方式

1. 数据，可以利用出门问问已开放的唤醒数据集，[mobvoi_hotwords_dataset](https://www.openslr.org/87/) ，在这里感谢*出门问问*为语音信号处理领域作出的贡献。

2. 模型训练

   ```bash
   cd tf23
   python train.py # 注意修改数据路径，音频时长等参数
   ```

3. 模型量化

   ```bash
   cd tf23
   python quantize.py # 必须保证参数与训练阶段完全一致，注意修改checkpoint路径
   ```

4. 量化后模型测试

   ```
   cd tf23
   python test_tflite.py # 注意修改量化后模型的名字和测试数据的路径
   ```

5. win下唤醒库开发

   ```
   cd ./deploy/KeywordSpotting/
   # 利用vs2015 开发唤醒lib，
   # 注意需要win下的tensorflowlite_c.dll 和tensorflowlite_c.dll.if.lib，总大小2.5xM
   # 如果没有，需要根据自己的环境编译tflite的dll和lib
   ```

   备注：KeywordSpotting，是一个解决方案同时包含三个project，kws-win-demo是利用量化模型和tflite库进行测试模型，kws-lib是将唤醒模型制作为win下的库，供其他客户端集成，kws-interface是对唤醒lib的接口测试。



# 其他

开发过程中参数说明可参考[开发文档](https://github.com/syw2014/Speech-Processing/blob/master/development.md)

