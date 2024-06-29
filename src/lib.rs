use tract_onnx::prelude::*;
use image::DynamicImage;

/// 将图像转换为模型输入的张量
fn image_to_tensor(image: DynamicImage) -> Tensor {
    let resized_img = image::imageops::resize(&image, 64, 64, image::imageops::FilterType::Triangle);
    let input: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 64, 64), |(_, c, h, w)| {
        resized_img[(w as u32, h as u32)][c] as f32 / 255.0
    }).into();
    input
}

/// 加载ONNX模型并进行推理
pub fn run_onnx_model(model_path: &str, image_path: &str) -> TractResult<usize> {
    // 加载ONNX模型
    let model = tract_onnx::onnx().model_for_path(model_path)?;

    // 设置输入形状
    let input_shape: [usize; 4] = [1, 3, 64, 64];
    let model = model.with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))?;
    let model = model.into_optimized()?.into_runnable()?;

    // 读取并预处理输入图像
    let img = image::open(image_path)?.to_rgb8();
    let img = DynamicImage::ImageRgb8(img);  // 将ImageBuffer转换为DynamicImage
    let input_data = image_to_tensor(img);

    // 运行模型进行推理
    let result = model.run(tvec!(input_data.into()))?;
    let output = result[0].to_array_view::<f32>()?;
    
    // 获取预测结果
    let predicted_class = output.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(idx, _)| idx).unwrap_or(0);

    Ok(predicted_class)
}



#[test]
fn test_run_onnx_model() {
    let model_path = "resources/models/skill_ready_cls.onnx";
    let image_path = "resources/input/skill_ready/unready/1.png";
    match run_onnx_model(model_path, image_path) {
        Ok(predicted_class) => println!("预测类别索引: {}", predicted_class),
        Err(e) => eprintln!("运行模型时出错: {:?}", e),
    }
}

