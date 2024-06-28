# AahAI

## 模型描述

- **deploy_direction_cls.onnx**：用于识别战斗干员的方向（四分类模型）。
- **operators_det.onnx**：用于检测战斗干员。
- **skill_ready_cls.onnx**：用于识别战斗技能是否准备好（二分类模型）。

## 文件说明

### skill_ready_cls_test.py

该脚本用于测试 `skill_ready_cls.onnx` 模型，判断战斗技能是否准备好。

