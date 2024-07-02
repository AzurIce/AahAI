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

image/label 命名表达式

```python
# direction表示朝向，从右开始顺时针以此为0 1 2 3
image = f"{index}_{name}_{x}_{y}_{direction}.png"
label = f"{index}_{name}_{x}_{y}_{direction}.txt"
```

6-3是1-3跑出来的
