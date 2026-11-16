use std::{collections::HashSet, error, fmt::Display};

#[derive(Debug, Clone)]
/// Defines an error caused by a mismatch between pipeline's expected input 
/// or outputs (if any), and the ones of the provided model.
pub struct UnexpectedModelSchemaError {
    message: String,
}

impl UnexpectedModelSchemaError {
    pub fn new_for_input(pipeline: &HashSet<&str>, model: &HashSet<&str>) -> Self {
        Self {
            message: format!("input tensors mismatch: pipeline provides {pipeline:?} but model expects {model:?}"),
        }
    }

    pub fn new_for_output(pipeline: &HashSet<&str>, model: &HashSet<&str>) -> Self {
        Self {
            message: format!("output tensors mismatch: pipeline expects {pipeline:?} but model provides {model:?}"),
        }
    }

    pub fn with(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }

    pub fn into_err<T>(self) -> super::Result<T> {
        Err(Box::new(self))
    }
}

impl error::Error for UnexpectedModelSchemaError { }

impl Display for UnexpectedModelSchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}