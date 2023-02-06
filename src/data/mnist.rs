use std::{path::PathBuf, fs};
use crate::network::Mat;
// use crate::matrix::{
//     MatBase,
//     Mat
// };

/// MNIST Data Formatting Information
///
/// LABEL FORMAT:
/// 
/// [offset]   [value]          [description]
/// 0000       2049             magic number
/// 0004       ???              # labels
/// 0008       ???              label
/// 0009       ???              label
/// ...
/// xxxx       ???              label
/// 
/// IMAGE FORMAT:
/// 
/// [offset]   [value]          [description]
/// 0000       2051             magic number
/// 0004       ???              number of images
/// 0008       28               # rows
/// 0012       28               # columns
/// 0016       ??               pixel
/// 0017       ??               pixel
/// ...
/// xxxx       ??               pixel

const TRAIN_LABELS_PATH: &str = "src/res/train-labels";
const TRAIN_IMAGES_PATH: &str = "src/res/train-images";
const TEST_LABELS_PATH: &str =  "src/res/test_labels";
const TEST_IMAGES_PATH: &str =  "src/res/test-images";
const LABEL_MAGIC_NUMBER: u32 = 2049;
const IMAGE_MAGIC_NUMBER: u32 = 2051;
const TRAIN_SAMPLES: usize  = 60_000;
const TEST_SAMPLES:  usize  = 10_000;
const LABEL_DATA_OFFSET: usize =   8;
const IMAGE_DATA_OFFSET: usize =  16;
const BYTES_PER_IMAGE: usize =   784;
const BYTES_PER_AXIS: usize =     28;

fn one_hot(hot: usize) -> Mat {
    let mut mat = Mat::zeros(10, 1);
    mat[(hot, 0)] = 1.0;
    mat
}

pub enum DataType {
    Train,
    Test
}

/// MNIST dataset reader
#[derive(Clone)]
pub struct Reader {
    train_images: Vec<Mat>,
    train_labels: Vec<Mat>,   
    test_images:  Vec<Mat>,
    test_labels:  Vec<Mat>
}

impl Reader {
    pub fn new() -> Self {
        let work_dir = std::env::current_dir().unwrap();

        Self { 
            train_images: Self::read_images(DataType::Train, &work_dir),
            train_labels: Self::read_labels(DataType::Train, &work_dir),
            test_images:  Self::read_images(DataType::Test, &work_dir),
            test_labels:  Self::read_labels(DataType::Test, &work_dir)
        }
    }

    pub fn train_images(&self) -> &Vec<Mat> {
        &self.train_images
    }

    pub fn train_labels(&self) -> &Vec<Mat> {
        &self.train_labels
    }

    pub fn test_images(&self) -> &Vec<Mat> {
        &self.test_images
    }

    pub fn test_labels(&self) -> &Vec<Mat> {
        &self.test_labels
    }

    fn read_labels(data_type: DataType, work_dir: &PathBuf) -> Vec<Mat> {
        let path = match data_type {
            DataType::Train => TRAIN_LABELS_PATH,
            DataType::Test =>  TEST_LABELS_PATH
        };

        let mut label_bytes = fs::read(work_dir.join(path)).expect("couldnt read labels!");
        assert_eq!(Self::read_to_u32(&label_bytes[0..4]), LABEL_MAGIC_NUMBER);
        label_bytes.drain(0..LABEL_DATA_OFFSET);
        
        match data_type {
            DataType::Train => assert_eq!(label_bytes.len(), TRAIN_SAMPLES),
            DataType::Test  => assert_eq!(label_bytes.len(), TEST_SAMPLES)
        }
        
        label_bytes
            .iter()
            .map(|label| one_hot(*label as usize))
            .collect()
    }

    fn read_images(data_type: DataType, work_dir: &PathBuf) -> Vec<Mat> {
        let path = match data_type {
            DataType::Train => TRAIN_IMAGES_PATH,
            DataType::Test =>  TEST_IMAGES_PATH
        };
 
        let mut image_bytes = fs::read(work_dir.join(path)).expect("couldn't read images!");
        assert_eq!(Self::read_to_u32(&image_bytes[0..4]), IMAGE_MAGIC_NUMBER);
        image_bytes.drain(0..IMAGE_DATA_OFFSET);
        
        match data_type {
           DataType::Train => assert_eq!(image_bytes.len(), BYTES_PER_IMAGE*TRAIN_SAMPLES),
           DataType::Test  => assert_eq!(image_bytes.len(), BYTES_PER_IMAGE*TEST_SAMPLES)
        }

        image_bytes
            .chunks(BYTES_PER_IMAGE)
            .map(|image| {
                let buf: Vec<f32> = image
                    .iter()
                    .map(|n| *n as f32 / 255.0)
                    .collect();

                Mat::from_vec(BYTES_PER_IMAGE, 1, buf)
            })
            .collect() 
    }

    pub fn print_image(&self, data_type: DataType, index: usize) {
        let images;
        
        match data_type {
            DataType::Train => images = &self.train_images,
            DataType::Test  => images = &self.test_images
        }

        for (i, byte) in images[index].iter().enumerate() {
            if *byte > 0.8 {
                print!("■ ")
            } else if *byte > 0.4 {
                print!("▧ ")
            } else if *byte > 0.0 {
                print!("□ ")
            } else {
                print!("- ")
            };

            if i % BYTES_PER_AXIS == 0 {
                println!()
            } 
        }
    }

    /// Reads a `u32` from 4 bytes
    fn read_to_u32(buf: &[u8]) -> u32 {
        let buf = [buf[0], buf[1], buf[2], buf[3]];
        u32::from_be_bytes(buf)
    }
}