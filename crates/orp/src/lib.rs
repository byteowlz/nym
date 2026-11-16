//! `orp` is a lightweight framework designed to simplify the creation and execution of **O**NNX **R**untime **P**ipelines. 
//! 
//! Built on top of [`ort`](https://ort.pyke.io), it provides a simple way to handle data pre- and post-processing, chain 
//! multiple ONNX models together, while encouraging code reuse and clarity.

pub mod params;
pub mod model;
pub mod pipeline;
pub mod error;
pub mod env;

pub type Result<T> = core::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
