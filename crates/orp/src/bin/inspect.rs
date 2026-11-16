//! Inspects an onnx file and prints info about the model and input/output tensors

pub fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: inspect <path to onnx model>");
    println!("LOADING FROM: {path}");
    let model = orp::model::Model::new(path, orp::params::RuntimeParameters::default())?;
    model.inspect(std::io::stdout())    
}