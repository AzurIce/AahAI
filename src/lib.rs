//! 模型列表：
//! - [`get_skill_ready`]: 模型名 `skill_ready_cls.onnx`
//! - [`get_direction`]: 模型名 `deploy_direction_cls.onnx`
//! - [`ocr`] :  掉包!!!
//! - [`get_blood`]: 模型名 `operators_det.rten`
//! 结构体列表
//! - [`Detection`] : 血条检测结果
//!                 label: 检测到的标签
//!                 score: 置信度
//!                 x1: 左上角x坐标
//!                 y1: 左上角y坐标
//!                 x2: 右下角x坐标
//!                 y2: 右下角y坐标
//! - [`OcrResult`] : ocr结果
//!                 bbox: 文本框坐标
//!                 text: 文本内容
//!                 confidence: 置信度

use encoding::all::GBK;
use encoding::{DecoderTrap, Encoding};
use image:: DynamicImage;
use regex::Regex;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use tract_onnx::prelude::*;
use uuid::Uuid;
use std::error::Error;
use rten::{
    ops::{non_max_suppression, BoxOrder},
    Dimension, FloatOperators, Model, TensorPool,
};
use rten_imageio::read_image;
use rten_imageproc::Rect;
use rten_tensor::{prelude::*, NdTensor, Storage};


pub struct Detection {
    pub label: String,
    pub score: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}
#[derive(Debug)]
pub struct OcrResult {
    pub bbox: Vec<[f64; 2]>,
    pub text: String,
    pub confidence: f64,
}

/// 获取模型路径
fn get_model_path<P: AsRef<Path>>(res_dir: P, model_filename: &str) -> PathBuf {
    let res_dir = res_dir.as_ref();
    res_dir.join("models").join(model_filename)
}

/// 将图像转换为模型输入的张量,用户模型输入
fn image_to_tensor(image: &DynamicImage, size: u32) -> Tensor {
    let resized_img =
        image::imageops::resize(image, size, size, image::imageops::FilterType::Triangle);
    let input: Tensor = tract_ndarray::Array4::from_shape_fn(
        (1, 3, size as usize, size as usize),
        |(_, c, h, w)| resized_img[(w as u32, h as u32)][c] as f32 / 255.0,
    )
    .into();
    input
}

/// 调用onnx模型进行预测
fn run_onnx_model<P: AsRef<Path>>(model_path: P, image: &DynamicImage, input_size: u32) -> TractResult<usize> {
    // 加载ONNX模型
    let model = tract_onnx::onnx().model_for_path(model_path)?;

    // 设置输入形状
    let input_shape: [usize; 4] = [1, 3, input_size as usize, input_size as usize];
    let model =
        model.with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))?;
    let model = model.into_optimized()?.into_runnable()?;

    // 将图像转换为模型输入的张量
    let input_data = image_to_tensor(&image, input_size);

    // 运行模型进行推理
    
    let result = model.run(tvec!(input_data.into()))?;
    let output = result[0].to_array_view::<f32>()?;
    

    // 获取预测结果
    let predicted_class = output
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    Ok(predicted_class)
}

/// 技能技能
pub fn get_skill_ready<P: AsRef<Path>>(image: &DynamicImage, res_dir: P) -> TractResult<usize> {
    let model_path = get_model_path(res_dir, "skill_ready_cls.onnx");
    
    run_onnx_model(model_path, image, 64)
}

/// 检测方位
pub fn get_direction<P: AsRef<Path>>(image: &DynamicImage, res_dir: P) -> TractResult<usize> {
    let model_path = get_model_path(res_dir, "deploy_direction_cls.onnx");
    run_onnx_model(model_path, image, 96)
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
        return Err(format!(
            "Failed to execute PaddleOCR: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
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

        results.push(OcrResult {
            bbox,
            text,
            confidence,
        });
    }

    Ok(results)
}

/// get_blood
pub fn get_blood<P: AsRef<Path>>(image: &DynamicImage, res_dir: P) -> Result<Vec<Detection>, Box<dyn Error>> {
    // 生成一个唯一的文件名
    let image_path = format!("{}.png", Uuid::new_v4());

    // 将DynamicImage保存到文件
    image.save(&image_path)?;

    // 获取模型路径
    let model_path = get_model_path(res_dir, "operators_det.rten");

    // 加载目标检测模型
    let model = Model::load_file(model_path)?;

    // 读取图像
    let image = read_image(&image_path)?;

    // 定义标签内容
    let labels = vec!["blood".to_string()];

    // 获取图像高度和宽度
    let [_, image_height, image_width] = image.shape();

    // 准备输入张量
    let mut image = image.as_dyn().to_tensor();
    image.insert_axis(0); // 增加批量维度

    // 获取模型输入形状
    let input_shape = model
        .input_shape(0)
        .ok_or("model does not specify expected input shape")?;
    let (input_h, input_w) = match &input_shape[..] {
        &[_, _, Dimension::Fixed(h), Dimension::Fixed(w)] => (h, w),
        _ => (640, 640), // 如果维度未固定，使用默认值
    };
    let image = image.resize_image([input_h, input_w])?;

    // 获取模型输入和输出节点ID
    let input_id = model.node_id("images")?;
    let output_id = model.node_id("output0")?;

    // 运行模型并获取输出
    let [output] = model.run_n(
        vec![(input_id, image.view().into())].as_slice(),
        [output_id],
        None,
    )?;

    // 处理模型输出
    let output: NdTensor<f32, 3> = output.try_into()?;
    let [_batch, box_attrs, _n_boxes] = output.shape();
    println!("{:?}", box_attrs);

    let model_in_h = image.size(2);
    let model_in_w = image.size(3);
    let scale_y = image_height as f32 / model_in_h as f32;
    let scale_x = image_width as f32 / model_in_w as f32;

    // 提取边框和分数
    let boxes = output.slice::<3, _>((.., ..4, ..)).permuted([0, 2, 1]);
    let scores = output.slice::<3, _>((.., 4.., ..));

    let iou_threshold = 0.3;
    let score_threshold = 0.25;

    // 非极大值抑制
    let nms_boxes = non_max_suppression(
        &TensorPool::new(),
        boxes.view(),
        scores,
        BoxOrder::CenterWidthHeight,
        None, /* max_output_boxes_per_class */
        iou_threshold,
        score_threshold,
    )?;
    let [n_selected_boxes, _] = nms_boxes.shape();

    println!("Found {n_selected_boxes} objects in image.");

    // 收集检测结果
    let mut detections = Vec::new();

    for b in 0..n_selected_boxes {
        let [batch_idx, cls, box_idx] = nms_boxes.slice(b).to_array();
        let [cx, cy, box_w, box_h] = boxes.slice([batch_idx, box_idx]).to_array();
        let score = scores[[batch_idx as usize, cls as usize, box_idx as usize]];

        let rect = Rect::from_tlhw(
            (cy - 0.5 * box_h) * scale_y,
            (cx - 0.5 * box_w) * scale_x,
            box_h * scale_y as f32,
            box_w * scale_x as f32,
        );

        let int_rect = rect.integral_bounding_rect().clamp(Rect::from_tlhw(
            2 as i32,
            2 as i32,
            image_height as i32 - 4 as i32,
            image_width as i32 - 4 as i32,
        ));

        let label = unsafe {
            labels
                .get(cls as usize)
                .map(|s| s.as_str())
                .unwrap_or("unknown")
        };

        detections.push(Detection {
            label: label.to_string(),
            score,
            x1: int_rect.left() as f32,
            y1: int_rect.top() as f32,
            x2: int_rect.right() as f32,
            y2: int_rect.bottom() as f32,
        });


    }

    // 删除临时文件
    std::fs::remove_file(image_path)?;

    Ok(detections)
}



#[cfg(test)]
mod tests {
    const res_dir: &str = "../azur-arknights-helper/resources";
    // use std::time::Instant;
    use super::*;
    use image::open;

    #[test] //测试技能
    fn test_get_skill_ready() {
        // let image_path = "resources/input/skill_ready/ready/ready1.png";
        // println!("加载图片: {}", image_path);

        // match open(image_path) {
        //     Ok(img) => {
        //         let img = img.to_rgb8();
        //         let img = DynamicImage::ImageRgb8(img);
        //         match get_skill_ready(&img) {
        //             Ok(predicted_class) => println!("预测二分类模型类别索引: {}", predicted_class),
        //             Err(e) => eprintln!("运行二分类模型时出错: {:?}", e),
        //         }
        //     }
        //     Err(e) => eprintln!("无法加载图片: {}", e),
        // }
    }

    #[test] //测试方位 "right" "down" "left" "up" -> 0 1 2 3
    fn test_get_direction() {
        // let image_path = "resources/input/direction/down1.png";
        // println!("加载图片: {}", image_path);

        // match open(image_path) {
        //     Ok(img) => {
        //         let img = img.to_rgb8();
        //         let img = DynamicImage::ImageRgb8(img);
        //         match get_direction(&img) {
        //             Ok(predicted_class) => println!("预测四分类模型类别索引: {}", predicted_class),
        //             Err(e) => eprintln!("运行四分类模型时出错: {:?}", e),
        //         }
        //     }
        //     Err(e) => eprintln!("无法加载图片: {}", e),
        // }
    }

    #[test] //测试ocr
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
    
    #[test] //测试foo
    fn test_get_blood() {
        let image_path = "model/resources/input/operators_det/1.png";
        println!("加载图片: {}", image_path);

        match open(image_path) {
            Ok(img) => {
                let img = img.to_rgb8();
                let img = DynamicImage::ImageRgb8(img);
                match get_blood(&img, res_dir) {
                    Ok(detections) => {
                        // 输出检测结果
                        for detection in detections {
                            println!(
                                "Label: {}, Score: {:.3}, Left: {}, Top: {}, Right: {}, Bottom: {}",
                                detection.label, detection.score, detection.x1, detection.y1, detection.x2, detection.y2
                            );
                        }
                    },
                    Err(e) => eprintln!("foo 测试失败: {:?}", e),
                }
            }
            Err(e) => eprintln!("无法加载图片: {}", e),
        }
    }
}
