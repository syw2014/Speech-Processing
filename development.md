1. 数据集已划分为train/valid/test
2. 唤醒词为“hi 小问”和“你好问问”，中文唤醒词，keyword_0：“hi,小问”， keyword_id1：“你好问问”

## Speech Command源码说明

### 1.FLAGS参数说明

1. data_url，数据下载链接，官方是从远端下载speech command dataset,如果是自己的数据集这里可以为空

2. data_dir，数据集存放路径，逻辑是会先检查该路径下数据集是否存在，不存在则从data_url这个地址下载

3. background_volume,背景声音的大小，范围为0-1，默认为0.1，用来设置噪声大小

4. background_frequency，有多少比例的训练样本中加入背景声音，默认是0.8

5. silence_percentage,有多少训练数据为背景声音，默认10.0

6. unknown_percentage，有多少比例的数据为unknown word，默认10.0

7. time_shift_ms,训练音频数据的移动范围，默认100.0

8. testing_percentage，多少比例的数据用做测试集，默认10

9. validation_percentage，多少比例的数据用于验证集，默认10

10. sample_rate，采用频率，默认16000

11. clip_duration_ms，音频数据时长，默认1000ms

12. window_size_ms，每帧时长，默认30ms

13. window_stride_ms，帧移动位移，默认10ms

14. featue_bin_count，MFCC特征的维度数，默认40【待确认】

15. how_many_training_steps，训练迭代部署，默认是"15000,3000",这里为什么是两个，考虑到不同训练阶段采用不同的learning rate配置【待确认】

16. eval_step_interval，多个少步做测试，默认400

17. learning_rate，学习率，默认“0.001,0.0001”，训练学习率

18. batch_size，默认100

19. summaries_dir, tensorboard 日志位置

20. wanted_words，目标词，默认:"nihaoxiaowen,chumenwenwen"

21. train_dir，存放eventlog和checkpoint的位置

22. save_step_interval，经过多少步存储一下模型，默认100

23. start_checkpoint，从哪里restore模型，默认空

24. model_architecture，模型结构，默认‘conv’ 【待补充】

25. check_nans，在处理过程中是否要做invalid 数值检查，默认false

26. preprocess，频谱处理方法，默认'mfcc'，可供选择包括"mfcc,average,micro"

27. verbosity，日志等级，默认为INFO

28. optimizer，优化器，默认gradient_descent， 还可以选择momentum

    

### 2.input_data说明

#### 2.1.class AudioProcessor

音频数据处理类，处理主要包括数据载入、数据集划分、训练数据准备

**init**

该函数是数据处理主要流程，包括三个环节:

**maybe_download_and_extract_dataset**

判断data_dir路径下是否有数据，若无则直接从地址下载

**prepare_data_index**

准备数据列表包括wav 文件名列表和label，整理出train/valid/test 音频文件列表和对应label，在我们的场景下，因已切分train/dev/test数据集，基于词进行处理即可，不需要再进行数据的划分，原样例中是根据hash来区分放到哪个数据集。具体过程如下

1. 准备正样本数据集，其中data_index是正样本即包含准确唤醒词的音频数据，unknown为不含唤醒词的音频数据

2. ```python
   self.data_index = {'validation': [], 'testing': [], 'training': []}
   unknown_index = {'validation': [], 'testing': [], 'training': []}
   ```

3. 由于音频文件和类别文件是单独分开存放的，在准备数据集时，需要将wav和label组对

4. 在每个数据集中加入silence_percentage比例的背景数据，背景数据选择是self.data_index['training'][0]['file']【这里可以替换其他数据，因为这些后续处理会和zero tensor相乘，不影响后续计算】

5. 将负样本对应的train、dev、valid加入到对应数据列表中



**prepare_background_data**

背景数据处理，载入内存，这里选用的speech_command_dataset中的背景声音，所有的背景声音全部都放在了self.background_data tensor中

**prepare_processing_graph**

对音频数据具体处理过程，主要包括创建音频WAV数据处理计算图Graph，同时进行decodes it, scales the volume,
        shifts it in time, adds in background noise, calculates a spectrogram, and
        then builds an MFCC fingerprint from that.

整个模块有6个placeholder和一个输出

1. wav_filename_placeholder_，wav文件输入占位
2. foreground_volume_placeholder_, 剪辑音频片段的音量大小占位
3. time_shift_padding_placeholder_,在哪个位置对音频片段进行padding
4. time_shift_offset_placeholder_，帧移
5. background_data_placeholder_，PCM格式的背景声音
6. background_volume_placeholder_，混入背景音的音量
7. output_，已处理的2-D音频特征

整体流程，

构造wav_loader->创建wav_decoder->构造volume适应器->wav_padding ->background声音加入->value clip保证在-1.0~1.0之间->利用gen_audio_ops生成spectrogram

这一步实质是将音频数据转化为模型可利用的数字化数据，即fingerprint,即音频文件的某种表示，类似于将文本表示成文本向量一样，之后将这个向量作为模型输入训练模块。

### 2.2 models说明

**create_model 函数**

模型工具函数，可以设计任意的模型结构，输入是fingerprint特征（即某个固定维度的向量表示），输出是[batch_size, feature_dim]，输出某个固定维度的特征表示。所需参数如下：

fingerprint_input， 音频特征

model_settings,模式设置参数

model_architecture，模型结构，可以通过该参数选择不同模型

is_training,模型运行模式

runtime_settings,runtime信息

**create_conv_model函数**

这里只介绍CONV模型，其他模型暂时未用到

模型整体结构为：

fingerprint_input->Conv2D(weights+bias)->Relu->MaxPool->Conv2D(weights+bias)->Relu->MaxPool->Matmul(weights+bias)-<logits

如果是在训练模式下，每个conv后面会经过dropout

整个模型即是一个机遇CNN的分类模型，重点是怎么将

## 使用方式

### 训练模型

生成checkpoint模型和summary

```bash
python  train.py #注意修改FLAGS的参数
```



### 模型quantize 和freeze

将checkpoint模型转换为pb模型

```
python freeze.py
```



### 模型inference

利用固化好的pb模型对wav文件进行预测

```
python label_wav.py 
```

C++方式的预测和推理

```bash
bash build.sh 1
```

C++方式，需要先编译生成libtensorflow2.4.1.so

## 参考文献

1. [PCM音频格式介绍]: https://blog.csdn.net/ljxt523/article/details/52068241

   

## 测试

### 测试方法

计划弄成PC端可执行文件形式，以测试SDK是否可集成在助手客户端内，进行中。

## 参考文献
1.https://blog.csdn.net/Berylxuan/article/details/80826533

