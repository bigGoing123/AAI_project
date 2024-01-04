# AAI_project
1.运行train_for_first，是用官方mnist数据集得到一个准确率高的模型，模型参数保存在cnn2.pkl中；

2.运行train_testSecond,首先删除噪声较大的数据，然后用余下的train数据集来训练一个新的模型，并用新的模型来测试test数据集，直接得到相应的csv文件。

