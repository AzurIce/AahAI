use regex::Regex;
use std::process::Command;
use encoding::{Encoding, DecoderTrap};
use encoding::all::GBK;
use image::{DynamicImage};
use uuid::Uuid;

#[derive(Debug)]
pub struct OcrResult {
    pub bbox: Vec<[f64; 2]>,
    pub text: String,
    pub confidence: f64,
}

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
    std::fs::remove_file(&image_path)?;

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

    #[test]
    fn test_ocr() {
        // 加载测试图片
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
