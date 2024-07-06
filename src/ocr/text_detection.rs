use image::{DynamicImage, GenericImageView, ImageBuffer, RgbImage};
use tract_onnx::prelude::*;
use nalgebra::{Matrix3, Vector3};
use std::path::Path;

struct TextDetector{
    binary_threshold: f32,
    polygon_threshold: f32,
    unclip_ratio: f32,
    max_candidates: i32,
    long_side_thresh: i32,
    short_size: i32,
    mean_values: [f32; 3],
    norm_values: [f32; 3],
    input_image: Vec<f32>,
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>
}

impl TextDetector {
    fn new<P: AsRef<Path>>(model_path: P) -> Self {
        let model = tract_onnx::onnx()
            .model_for_path(model_path).unwrap()
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 736, 736))).unwrap()
            .into_optimized().unwrap()
            .into_runnable().unwrap();

        TextDetector {
            binary_threshold: 0.3,
            polygon_threshold: 0.5,
            unclip_ratio: 1.6,
            max_candidates: 1000,
            long_side_thresh: 3,
            short_size: 736,
            mean_values: [0.485, 0.456, 0.406],
            norm_values: [0.229, 0.224, 0.225],
            input_image: vec![],
            model,
        }
    }

    fn preprocess(&self, srcimg: &DynamicImage) -> DynamicImage {
        let mut img = srcimg.to_rgb8();
        let (h, w) = (img.height(), img.width());
        let (mut scale_h, mut scale_w) = (1.0, 1.0);

        if h < w {
            scale_h = self.short_size as f32 / h as f32;
            let mut tar_w = w as f32 * scale_h;
            tar_w = tar_w - (tar_w as u32 % 32) as f32;
            tar_w = tar_w.max(32.0);
            scale_w = tar_w / w as f32;
        } else {
            scale_w = self.short_size as f32 / w as f32;
            let mut tar_h = h as f32 * scale_w;
            tar_h = tar_h - (tar_h as u32 % 32) as f32;
            tar_h = tar_h.max(32.0);
            scale_h = tar_h / h as f32;
        }

        img = image::imageops::resize(&img, (scale_w * w as f32) as u32, (scale_h * h as f32) as u32, image::imageops::FilterType::Lanczos3);
        DynamicImage::ImageRgb8(img)
    }

    fn normalize(&mut self, img: &DynamicImage) {
        let (width, height) = img.dimensions();
        self.input_image.clear();
        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let pixel = img.get_pixel(x, y).0[c];
                    self.input_image.push(pixel as f32 / 255.0);
                }
            }
        }
    }

    fn get_perspective_transform(src: &[(f32, f32); 4], dst: &[(f32, f32); 4]) -> Matrix3<f32> {
        let a = Matrix3::from_columns(&[
            Vector3::new(src[0].0, src[0].1, 1.0),
            Vector3::new(src[1].0, src[1].1, 1.0),
            Vector3::new(src[2].0, src[2].1, 1.0),
        ]);
        let b = Vector3::new(src[3].0, src[3].1, 1.0);

        let a_inv = a.try_inverse().unwrap();
        let h = a_inv * b;

        let mut h_matrix = Matrix3::identity();
        h_matrix[(0, 0)] = h[0];
        h_matrix[(1, 1)] = h[1];
        h_matrix[(2, 2)] = h[2];

        let mut src_mat = Matrix3::identity();
        for i in 0..3 {
            src_mat[(0, i)] = src[i].0;
            src_mat[(1, i)] = src[i].1;
            src_mat[(2, i)] = 1.0;
        }
        src_mat = src_mat * h_matrix;

        let mut dst_mat = Matrix3::identity();
        for i in 0..3 {
            dst_mat[(0, i)] = dst[i].0;
            dst_mat[(1, i)] = dst[i].1;
            dst_mat[(2, i)] = 1.0;
        }

        dst_mat * src_mat.try_inverse().unwrap()
    }

    fn warp_perspective(img: &DynamicImage, transform: &Matrix3<f32>, width: u32, height: u32) -> DynamicImage {
        let img = img.to_rgb8();
        let mut out_img = RgbImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let src_pos = transform * Vector3::new(x as f32, y as f32, 1.0);
                let src_x = (src_pos[0] / src_pos[2]).round() as u32;
                let src_y = (src_pos[1] / src_pos[2]).round() as u32;

                if src_x < img.width() && src_y < img.height() {
                    let pixel = img.get_pixel(src_x, src_y);
                    out_img.put_pixel(x, y, *pixel);
                }
            }
        }

        DynamicImage::ImageRgb8(out_img)
    }

    fn get_rotate_crop_image(&self, frame: &DynamicImage, vertices: Vec<(f32, f32)>) -> DynamicImage {
        let rect = {
            let (min_x, min_y, max_x, max_y) = vertices.iter().fold((f32::MAX, f32::MAX, f32::MIN, f32::MIN), |(min_x, min_y, max_x, max_y), &(x, y)| {
                (min_x.min(x), min_y.min(y), max_x.max(x), max_y.max(y))
            });
            ((min_x as u32, min_y as u32), (max_x as u32, max_y as u32))
        };

        let crop_img = frame.crop_imm(rect.0 .0, rect.0 .1, rect.1 .0 - rect.0 .0, rect.1 .1 - rect.0 .1);
        let output_size = (rect.1 .0 - rect.0 .0, rect.1 .1 - rect.0 .1);

        let target_vertices = [
            (0.0, output_size.1 as f32),
            (0.0, 0.0),
            (output_size.0 as f32, 0.0),
            (output_size.0 as f32, output_size.1 as f32),
        ];

        let mut vertices_shifted = vertices.clone();
        for v in &mut vertices_shifted {
            v.0 -= rect.0 .0 as f32;
            v.1 -= rect.0 .1 as f32;
        }

        let transform = TextDetector::get_perspective_transform(&vertices_shifted.try_into().unwrap(), &target_vertices);
        TextDetector::warp_perspective(&crop_img, &transform, output_size.0, output_size.1)
    }
}

#[cfg(test)]
mod test {
    use tract_data::internal::tract_smallvec::SmallVec;

    use crate::get_model_path;
    use super::*;

    #[test]
    pub fn test_text_detector() {
        // Initialize the text detector with the path to the ONNX model
        let model_path = get_model_path("../azur-arknights-helper/resources", "ch_PP-OCRv3_det_infer.onnx");
        let mut text_detector = TextDetector::new(model_path);

        // Load the image from a file
        let img_path = "./assets/imgs/ocr1.png";
        let img = image::open(img_path).unwrap();

        // Preprocess the image
        let preprocessed_img = text_detector.preprocess(&img);

        // Normalize the image
        text_detector.normalize(&preprocessed_img);

        // Here you can add your code to run the model and process the output
        // For example:
        // Convert normalized image data to tract_ndarray Array4
        let input_tensor = tract_ndarray::Array4::from_shape_vec(
            (1, 3, preprocessed_img.height() as usize, preprocessed_img.width() as usize),
            text_detector.input_image.clone(),
        ).unwrap();

        // Convert to TValue
        let input_tensor: Tensor = input_tensor.into();
        let input_tensor: TValue = input_tensor.into();
        
        let result = text_detector.model.run(tvec!(input_tensor)).unwrap();

        // Process the result as needed
        println!("{:?}", result);
    }
}