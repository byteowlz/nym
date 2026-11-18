//! Encapsulation of raw tensor outputs

use orp::pipeline::OwnedOutputs;
use composable::Composable;
use crate::util::result::Result;
use crate::model::pipeline::context::EntityContext;

/// Represents the raw tensor output of the inference step
pub struct TensorOutput {
    pub context: EntityContext,
    pub tensors: OwnedOutputs,
}


impl TensorOutput {
    pub fn from(tensors: OwnedOutputs, context: EntityContext) -> Self {
        Self { 
            context,
            tensors 
        }
    }
}


/// Composable: (SessionOutput, TensorMeta) => TensorOutput
#[derive(Default)]
pub struct SessionOutputToTensors { }


impl Composable<(OwnedOutputs, EntityContext), TensorOutput> for SessionOutputToTensors {
    fn apply(&self, input: (OwnedOutputs, EntityContext)) -> Result<TensorOutput> {
        Ok(TensorOutput::from(input.0, input.1))
    }
}
