# AahAI

## 模型描述

- **deploy_direction_cls.onnx**：用于识别战斗干员的方向（四分类模型）。
- **operators_det.onnx**：用于检测战斗干员。
- **skill_ready_cls.onnx**：用于识别战斗技能是否准备好（二分类模型）。
- **ocr模型**:调用paddlepaddle包预测。

## 模型基本信息

### Model: resources/models/skill_ready_cls.onnx

输入名称: input
`输入形状: [1, 3, 64, 64]`
输入类型: tensor(float)

原始图像形状: (64, 64, 3)
调整后输入图像的形状: (1, 3, 64, 64)
模型的输出: [array([[-3757.0815,  3828.271 ]], dtype=float32)]
预测类别索引: 1

```mermaid
graph TD;
    A[输入层 1x3x64x64] --> B[卷积层1 5x5x10]
    B --> C[MaxPool 2x2]
    C --> D[ReLU]
    D --> E[Inception模块1]
    E --> F[卷积层2 5x5x20]
    F --> G[MaxPool 2x2]
    G --> H[ReLU]
    H --> I[Inception模块2]
    I --> J[Reshape]
    J --> K[全连接层]
    K --> L[输出层]
    
    subgraph Inception模块1
        direction TB
        E1[1x1卷积]
        E2[1x1卷积 5x5卷积]
        E3[1x1卷积 3x3卷积 3x3卷积]
        E4[平均池化 1x1卷积]
        E1 --> E
        E2 --> E
        E3 --> E
        E4 --> E
    end
    
    subgraph Inception模块2
        direction TB
        I1[1x1卷积]
        I2[1x1卷积 5x5卷积]
        I3[1x1卷积 3x3卷积 3x3卷积]
        I4[平均池化 1x1卷积]
        I1 --> I
        I2 --> I
        I3 --> I
        I4 --> I
    end

```



### Model: resources/models/deploy_direction_cls.onnx

输入名称: input
`输入形状: [1, 3, 96, 96]`
输入类型: tensor(float)

原始图像形状: (96, 96, 3)
调整后输入图像的形状: (1, 3, 96, 96)
模型的输出: [array([[-303.65408, -643.6919 , -568.03253,    0.     ]], dtype=float32)]
预测干员方向类别索引: 3

```mermaid
graph TD;
    A[输入层 1x3x96x96] --> B[减法常量层]
    B --> C[乘法常量层]
    C --> D[卷积层 1 5x5x32]
    D --> E[ReLU 1]
    E --> F[卷积层 2 5x5x64]
    F --> G[ReLU 2]
    G --> H[卷积层 3 5x5x96]
    H --> I[ReLU 3]
    I --> J[卷积层 4 5x5x128]
    J --> K[ReLU 4]
    K --> L[卷积层 5 5x5x160]
    L --> M[ReLU 5]
    M --> N[转置层 NHWC]
    N --> O[Flatten层]
    O --> P[全连接层]
    P --> Q[BatchNormalization层]
    Q --> R[LogSoftmax层]
    R --> S[输出]
```



###Model: resources/models/operators_det.onnx

Model inputs:
Name: images
Type: FLOAT
`Shape: [1, 3, 640, 640]`
Model outputs:
Name: output0
Type: FLOAT
`Shape: [1, 5, 8400]`







```
use std::io::Cursor;
use image::DynamicImage;
use tract_onnx::prelude::*;

#[derive(Debug)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
}



// 嵌入模型文件
const SKILL_READY_CLS_MODEL: &[u8] = include_bytes!("../resources/models/skill_ready_cls.onnx");
const DEPLOY_DIRECTION_CLS_MODEL: &[u8] = include_bytes!("../resources/models/deploy_direction_cls.onnx");
const OPERATORS_DET_MODEL: &[u8] = include_bytes!("../resources/models/operators_det.onnx");

/// 将图像转换为模型输入的张量
fn image_to_tensor(image: &DynamicImage, size: u32) -> Tensor {
    // draw_box(image, x, y, width, height, rgba_u8)
    let resized_img = image::imageops::resize(image, size, size, image::imageops::FilterType::Triangle);
    let input: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, size as usize, size as usize), |(_, c, h, w)| {
        resized_img[(w as u32, h as u32)][c] as f32 / 255.0
    }).into();
    input
}

/// 加载ONNX模型并进行推理（通用）
fn run_onnx_model_generic(model_bytes: &[u8], image_path: &str, input_size: u32) -> TractResult<usize> {
    // 加载ONNX模型
    let mut cursor = Cursor::new(model_bytes);
    let model = tract_onnx::onnx().model_for_read(&mut cursor)?;

    // 设置输入形状
    let input_shape: [usize; 4] = [1, 3, input_size as usize, input_size as usize];
    let model = model.with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))?;
    let model = model.into_optimized()?.into_runnable()?;

    // 读取并预处理输入图像
    let img = image::open(image_path)?.to_rgb8();
    let img = DynamicImage::ImageRgb8(img);  // 将 ImageBuffer 转换为 DynamicImage
    let input_data = image_to_tensor(&img, input_size);

    // 运行模型进行推理
    let result = model.run(tvec!(input_data.into()))?;
    let output = result[0].to_array_view::<f32>()?;
    
    // 获取预测结果
    let predicted_class = output.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(idx, _)| idx).unwrap_or(0);

    Ok(predicted_class)
}

/// 加载ONNX模型并进行推理（二分类模型）
pub fn run_onnx_model_two_class(image_path: &str) -> TractResult<usize> {
    run_onnx_model_generic(SKILL_READY_CLS_MODEL, image_path, 64)
}

/// 加载ONNX模型并进行推理（四分类模型）
pub fn run_onnx_model_four_class(image_path: &str) -> TractResult<usize> {
    run_onnx_model_generic(DEPLOY_DIRECTION_CLS_MODEL, image_path, 96)
}

/// 加载ONNX模型并进行目标检测（YOLOv8 模型）
pub fn run_yolov8_model(image_path: &str) -> TractResult<Vec<Detection>> {
    // 加载ONNX模型
    let mut cursor = Cursor::new(OPERATORS_DET_MODEL);
    let model = tract_onnx::onnx().model_for_read(&mut cursor)?;

    // 设置输入形状
    let input_shape: [usize; 4] = [1, 3, 640, 640];
    let model = model.with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))?;
    let model = model.into_optimized()?.into_runnable()?;

    // 读取并预处理输入图像
    let img = image::open(image_path)?.to_rgb8();
    let img = DynamicImage::ImageRgb8(img);  // 将 ImageBuffer 转换为 DynamicImage
    let input_data = image_to_tensor(&img, 640);

    // 运行模型进行推理
    let result = model.run(tvec!(input_data.into()))?;
    let output = result[0].to_array_view::<f32>()?;

    // 解析模型输出
    let mut results = Vec::new();
    let output_shape = output.shape();
    for i in 0..output_shape[1] {
        let score = output[[0, i, 4]];
        if score > 0.3  {  // 过滤置信度较低和不合理的检测框
            // 获取检测框的中心坐标和尺寸
            let center_x = output[[0, i, 0]] * img.width() as f32 / 640.0;  // 中心x坐标
            let center_y = output[[0, i, 1]] * img.height() as f32 / 640.0;  // 中心y坐标
            let w = output[[0, i, 2]] * img.width() as f32 / 640.0;  // 宽度
            let h = output[[0, i, 3]] * img.height() as f32 / 640.0;  // 高度

            // 将中心坐标转换为左上角坐标
            let x = center_x - w / 2.0;  // 左上角x坐标
            let y = center_y - h / 2.0;  // 左上角y坐标

            // 存储检测结果
            results.push(Detection { x, y, w, h, score });
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    

    use super::*;

    #[test]
    fn test_run_onnx_model_two_class() {
        
        let image_path = "resources/input/skill_ready/unready/unready1.png";
        match run_onnx_model_two_class(image_path) {
            Ok(predicted_class) => println!("预测二分类模型类别索引: {}", predicted_class),
            Err(e) => eprintln!("运行二分类模型时出错: {:?}", e),
        }
    }

    #[test]
    fn test_run_onnx_model_four_class() {
        let screen_cap_cnt = 2;
        let tests = [("right", 4), ("down", 0), ("left", 3), ("up", 1)];
        for i in 0..screen_cap_cnt {
            for (_class, &(dir, cnt)) in tests.iter().enumerate() {
                for j in 0..cnt {
                    let image_path = format!("resources/input/direction/LS-6_{i}_{dir}{j}.png");
                    match run_onnx_model_four_class( &image_path) {
                        Ok(predicted_class) => {
                            println!("预测四分类模型类别索引: {}", predicted_class);
                            // assert_eq!(i, predicted_class);
                        }
                        Err(e) => eprintln!("运行四分类模型时出错: {:?}", e),
                    }
                }
            }
        }
    }

    #[test]
    fn test_run_yolov8_model() {
      
        let image_path = "resources/input/operators_det/2.png";
        match run_yolov8_model( image_path) {
            Ok(detections) => {
                for detection in detections {
                    println!(
                        "检测到的目标: 位置 ({}, {}), 宽度 {}, 高度 {}, 置信度 {:.2}",
                        detection.x, detection.y, detection.w, detection.h, detection.score
                    );
                }
            }
            Err(e) => eprintln!("运行模型时出错: {:?}", e),
        }
    }


}

```

```
use regex::Regex;
use std::process::Command;
use std::path::Path;
use encoding::{Encoding, DecoderTrap};
use encoding::all::GBK;

#[derive(Debug)]
pub struct OcrResult {
    pub bbox: Vec<[f64; 2]>,
    pub text: String,
    pub confidence: f64,
}

pub fn ocr(image_path: &str) -> Result<Vec<OcrResult>, Box<dyn std::error::Error>> {
    // Check if the image file exists
    if !Path::new(image_path).exists() {
        return Err(format!("Image file not found: {}", image_path).into());
    }

    // Construct the command
    let output = Command::new("paddleocr")
        .arg("--image_dir")
        .arg(image_path)
        .arg("--use_angle_cls")
        .arg("true")
        .arg("--use_gpu")
        .arg("false")
        .output()?;

    if !output.status.success() {
        return Err(format!("Failed to execute PaddleOCR: {}", String::from_utf8_lossy(&output.stderr)).into());
    }

    // Convert output to string, handle possible GBK encoding
    let stdout_gbk = GBK.decode(&output.stdout, DecoderTrap::Replace)?;
    let stdout = String::from_utf8_lossy(stdout_gbk.as_bytes()).to_string();

    // Define the regex pattern to match OCR results
    let re = Regex::new(r": \[\[\[(.*?), (.*?)\], \[(.*?), (.*?)\], \[(.*?), (.*?)\], \[(.*?), (.*?)\]\], \('(.*)', (.*)\)\]").unwrap();
    let mut results = Vec::new();

    // Parse the OCR results
    for cap in re.captures_iter(&stdout) {
        let bbox = vec![
            [cap[1].parse::<f64>()?, cap[2].parse::<f64>()?],
            [cap[3].parse::<f64>()?, cap[4].parse::<f64>()?],
            [cap[5].parse::<f64>()?, cap[6].parse::<f64>()?],
            [cap[7].parse::<f64>()?, cap[8].parse::<f64>()?],
        ];
        let text = cap[9].to_string();
        let confidence = cap[10].parse::<f64>()?;

        results.push(OcrResult { bbox, text, confidence });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocr() {
        let image_path = "resources/input/ocr/11.jpg";
        match ocr(image_path) {
            Ok(results) => {
                for result in results {
                    println!("{:#?}", result);
                }
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }
}

```

