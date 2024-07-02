use regex::Regex;
use std::process::Command;
use encoding::{Encoding, DecoderTrap};
use encoding::all::GBK;
use image::{DynamicImage};
use uuid::Uuid;
use std::io::Cursor;
use tract_onnx::prelude::*;

const SKILL_READY_CLS_MODEL: &[u8] = include_bytes!("../resources/models/skill_ready_cls.onnx");
const DEPLOY_DIRECTION_CLS_MODEL: &[u8] = include_bytes!("../resources/models/deploy_direction_cls.onnx");
const OPERATORS_DET_MODEL: &[u8] = include_bytes!("../resources/models/operators_det.onnx");

#[derive(Debug)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
}

#[derive(Debug)]
pub struct OcrResult {
    pub bbox: Vec<[f64; 2]>,
    pub text: String,
    pub confidence: f64,
}

/// 将图像转换为模型输入的张量,用户模型输入
fn image_to_tensor(image: &DynamicImage, size: u32) -> Tensor {
    
    let resized_img = image::imageops::resize(image, size, size, image::imageops::FilterType::Triangle);
    let input: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, size as usize, size as usize), |(_, c, h, w)| {
        resized_img[(w as u32, h as u32)][c] as f32 / 255.0
    }).into();
    input
}

/// 调用onnx模型进行预测
fn run_onnx_model(model_bytes: &[u8], image: &DynamicImage, input_size: u32) -> TractResult<usize> {
    // 加载ONNX模型
    let mut cursor = Cursor::new(model_bytes);
    let model = tract_onnx::onnx().model_for_read(&mut cursor)?;

    // 设置输入形状
    let input_shape: [usize; 4] = [1, 3, input_size as usize, input_size as usize];
    let model = model.with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))?;
    let model = model.into_optimized()?.into_runnable()?;

    // 将图像转换为模型输入的张量
    let input_data = image_to_tensor(&image, input_size);

    // 运行模型进行推理
    let result = model.run(tvec!(input_data.into()))?;
    let output = result[0].to_array_view::<f32>()?;
    
    // 获取预测结果
    let predicted_class = output.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(idx, _)| idx).unwrap_or(0);
    Ok(predicted_class)
}

/// 技能技能
pub fn get_skill_ready(image: &DynamicImage) -> TractResult<usize> {
    run_onnx_model(SKILL_READY_CLS_MODEL, image, 64)
}

/// 检测方位
pub fn get_direction(image: &DynamicImage) -> TractResult<usize> {
    run_onnx_model(DEPLOY_DIRECTION_CLS_MODEL, image, 96)
}

/// 检测血条
pub fn get_blood(image: &DynamicImage) -> TractResult<Vec<Detection>> {
    // 加载ONNX模型
    let mut cursor = Cursor::new(OPERATORS_DET_MODEL);
    let model = tract_onnx::onnx().model_for_read(&mut cursor)?;

    // 设置输入形状
    let input_shape: [usize; 4] = [1, 3, 640, 640];
    let model = model.with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))?;
    let model = model.into_optimized()?.into_runnable()?;

    // 将图像转换为模型输入的张量
    let input_data = image_to_tensor(&image, 640);

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
            // let center_x = output[[0, i, 0]] * img.width() as f32 / 640.0;  // 中心x坐标
            // let center_y = output[[0, i, 1]] * img.height() as f32 / 640.0;  // 中心y坐标
            // let w = output[[0, i, 2]] * img.width() as f32 / 640.0;  // 宽度
            // let h = output[[0, i, 3]] * img.height() as f32 / 640.0;  // 高度
            let center_x = output[[0, i, 0]] ;  // 中心x坐标
            let center_y = output[[0, i, 1]] ;  // 中心y坐标
            let w = output[[0, i, 2]] ;  // 宽度
            let h = output[[0, i, 3]] ;  // 高度

            // 将中心坐标转换为左上角坐标
            let x = center_x - w / 2.0;  // 左上角x坐标
            let y = center_y - h / 2.0;  // 左上角y坐标

            // 存储检测结果
            results.push(Detection { x, y, w, h, score });
        }
    }

    Ok(results)
}

/// ocr预测
pub fn ocr(image: &DynamicImage) -> Result<Vec<OcrResult>, Box<dyn std::error::Error>> {
    // 生成一个唯一的文件名
    let image_path = format!("{}.png", Uuid::new_v4());

    // 将DynamicImage保存到文件
    image.save(&image_path)?;

    // 构建命令
    let output = Command::new("paddleocr")
        .arg("--image_dir")
        .arg(&image_path)
        .arg("--use_angle_cls")
        .arg("true")
        .arg("--use_gpu")
        .arg("false")
        .output()?;

    // 删除临时文件
    std::fs::remove_file(image_path)?;

    if !output.status.success() {
        return Err(format!("Failed to execute PaddleOCR: {}", String::from_utf8_lossy(&output.stderr)).into());
    }

    // 将输出转换为字符串，处理可能的GBK编码
    let stdout_gbk = GBK.decode(&output.stdout, DecoderTrap::Replace)?;
    let stdout = String::from_utf8_lossy(stdout_gbk.as_bytes()).to_string();

    // 定义正则表达式模式来匹配OCR结果
    let re = Regex::new(r": \[\[\[(.*?), (.*?)\], \[(.*?), (.*?)\], \[(.*?), (.*?)\], \[(.*?), (.*?)\]\], \('(.*)', (.*)\)\]").unwrap();
    let mut results = Vec::new();

    // 解析OCR结果
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
    use image::open;

    #[test]//测试技能
    fn test_get_skill_ready() {
        let image_path = "resources/input/skill_ready/ready/ready1.png";
        println!("加载图片: {}", image_path);

        match open(image_path) {
            Ok(img) => {
                let img = img.to_rgb8();
                let img = DynamicImage::ImageRgb8(img);
                match get_skill_ready(&img) {
                    Ok(predicted_class) => println!("预测二分类模型类别索引: {}", predicted_class),
                    Err(e) => eprintln!("运行二分类模型时出错: {:?}", e),
                }
            }
            Err(e) => eprintln!("无法加载图片: {}", e),
        }
        
    }

    #[test]//测试方位 "right" "down" "left" "up" -> 0 1 2 3
    fn test_get_direction() {
        let image_path = "resources/input/direction/down1.png";
        println!("加载图片: {}", image_path);

        match open(image_path) {
            Ok(img) => {
                let img = img.to_rgb8();
                let img = DynamicImage::ImageRgb8(img);
                match get_direction(&img) {
                    Ok(predicted_class) => println!("预测四分类模型类别索引: {}", predicted_class),
                    Err(e) => eprintln!("运行四分类模型时出错: {:?}", e),
                }
            }
            Err(e) => eprintln!("无法加载图片: {}", e),
        }
        
    }

    #[test]
    fn test_get_blood() {
        let image_path = "resources/input/operators_det/2.png";
        println!("加载图片: {}", image_path);
        
        match open(image_path) {
            Ok(img) => {
                let img = img.to_rgb8();
                let img = DynamicImage::ImageRgb8(img);
                match get_blood(&img) {
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
            Err(e) => eprintln!("无法加载图片: {}", e),
        }

        
    }


    #[test]//测试ocr
    fn test_ocr() {
        let image_path = "resources/input/ocr/11.jpg";
        println!("加载图片: {}", image_path);

        match open(image_path) {
            Ok(img) => {
                let img = img.to_rgb8();
                let img = DynamicImage::ImageRgb8(img);
                match ocr(&img) {
                    Ok(results) => {
                        for result in results {
                            println!("{:#?}", result);
                        }
                    }
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            Err(e) => eprintln!("无法加载图片: {}", e),
        }
    }
}
