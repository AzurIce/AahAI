## map-operator-deploy

### 1.描述

该目录为基于YOLOv8模型的地图部署干员检测数据处理区

实现：在一个关卡中，识别**一个干员**，部署在**一个地块**

### 2.环境

关卡：1-4

干员：

★Lancet

★★yedao

★★heijiao

★★xunlinzhe

★★dulin

★★★meilansha

★★★lingyu

★★★shiduhuade

★★★ansaier

★★★keluosi

地图： 9 x 5

![MuMu12-20240701-180919](README.assets/MuMu12-20240701-180919.png)



### 3.数据集

```yaml
names:
  0: Lancet
  1: yedao
  2: heijiao
  3: xunlinzhe
  4: dulin
  5: meilansha
  6: lingyu
  7: shiduhuade
  8: ansaier
  9: keluosi
  10: 12f
  11: fen
  12: paopuka
  13: yanrong
  14: zilan
  15: xiangcao
  16: yuejianye
  17: bandian
  18: kati
  19: migelu
  20: andeqieer
  21: kongbao
  22: furong
  23: taojinniang
```

图片：

![image-20240703150545813](README.assets/image-20240703150545813.png)

![image-20240703150628220](README.assets/image-20240703150628220.png)





image/label 命名表达式

```python
# direction表示朝向，从右开始顺时针以此为0 1 2 3
image = f"{index}_{name}_{x}_{y}_{direction}.png"
label = f"{index}_{name}_{x}_{y}_{direction}.txt"
```

6-3是1-3跑出来的



# 田佳文请认真阅读以下内容

### 地图数据参考

![image-20240702163102818](README.assets/image-20240702163102818.png)

工作内容：

1.录制一段自己打1-4的视频，干员为视频中的这10个，3星要求全部精一，医疗车是黄色的

2.对于视频中的任何一帧，截图（1920 x 1080 大小**严格要求**），命名如下：

`id_name_x_y_direction`

```
# direction表示朝向，从右开始顺时针以此为0 1 2 3
# id name对照表：
names:
  0:Lancet
  1:yedao
  2:heijiao
  3:xunlinzhe
  4:dulin
  5:meilansha
  6:lingyu
  7:shiduhuade
  8:ansaier
  9:keluosi
```

举例：

![6_lingyu_4_3_1](README.assets/6_lingyu_4_3_1.png)

6_lingyu_4_3_1.png

6号lingyu位于x=4，y=3的格子，朝向为1

## 问题来了：如果同时部署了多个干员怎么办

比如：

![2](README.assets/2.png)

那就搞两张这个图片，分别命名为：

`1_yedao_4_2_0.png` 和 `0_Lancet_5_5_3.png`

![4](README.assets/4.png)

小练习：上面的这张图片需要___张截图，除了`1_yedao_4_2_0.png` 和 `0_Lancet_5_5_3.png`还需要哪些命名？（填空，每空半条命）

