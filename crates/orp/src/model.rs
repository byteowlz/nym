use std::collections::HashSet;
use std::path::Path;
use composable::Composable;
use ort::session::{Session, SessionInputs, SessionOutputs, builder::GraphOptimizationLevel};

use crate::env::global_environment;
use crate::error::UnexpectedModelSchemaError;

use super::Result;
use super::params::RuntimeParameters;
use super::pipeline::Pipeline;


/// A `Model` can load an ONNX model, and run it using the provided pipeline.
pub struct Model {
    /// Interior mutability exists only to satisfy ort rc.15's `&mut self` on
    /// `Session::run`, which ties output lifetimes to the session borrow.
    /// ONNX Runtime documents sessions as thread-safe for concurrent Run, and
    /// outputs never outlive a `run` call, so sharing `&Model` across threads
    /// is sound (mirrors nym's token-backend `SharedSession`).
    session: std::cell::UnsafeCell<Session>,
}


impl Model {    
    pub fn new<P: AsRef<Path>>(model_path: P, params: RuntimeParameters) -> Result<Self> {
        let environment = global_environment()?;
        let session = Session::builder(&environment)?
            // ONNX Runtime 1.31 auto-selects providers (MaxEfficiency
            // registers CoreML on macOS); preserve the historical
            // manual-EP-only behavior.
            .with_auto_device(ort::session::builder::AutoDevicePolicy::PreferCPU)
            .map_err(|error| error.to_string())?
            .with_intra_threads(params.threads())
            .map_err(|error| error.to_string())?
            .with_execution_providers(params.into_execution_providers())
            .map_err(|error| error.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| error.to_string())?
            .commit_from_file(model_path)?;

        Ok(Self {
            session: std::cell::UnsafeCell::new(session),
        })
    }

    pub fn new_from_bytes(model_bytes: &[u8], params: RuntimeParameters) -> Result<Self> {
        let environment = global_environment()?;
        let session = Session::builder(&environment)?
            // ONNX Runtime 1.31 auto-selects providers (MaxEfficiency
            // registers CoreML on macOS); preserve the historical
            // manual-EP-only behavior.
            .with_auto_device(ort::session::builder::AutoDevicePolicy::PreferCPU)
            .map_err(|error| error.to_string())?
            .with_intra_threads(params.threads())
            .map_err(|error| error.to_string())?
            .with_execution_providers(params.into_execution_providers())
            .map_err(|error| error.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| error.to_string())?
            .commit_from_memory(model_bytes)?;

        Ok(Self {
            session: std::cell::UnsafeCell::new(session),
        })
    }

    /// Perform inferences using the provided pipeline and parameters
    pub fn inference<'a, P: Pipeline<'a>>(&'a self, input: P::Input, pipeline: &P, params: &P::Parameters) -> Result<P::Output> {
        // check schema
        self.check_schema(pipeline, params)?;
        // pre-process
        let (input, context) = pipeline.pre_processor(params).apply(input)?;
        // inference
        let output = self.run(input)?;                
        // post-process
        let output = pipeline.post_processor(params).apply((output, context))?;        
        // ok
        Ok(output)
    }

    pub fn to_composable<'a, P: Pipeline<'a>>(&'a self, pipeline: &'a P, params: &'a P::Parameters) -> impl Composable<P::Input, P::Output> {
        ComposableModel::new(self, pipeline, params)
    }

    /// Writes various model properties from metadata and input/output tensors
    pub fn inspect<W: std::io::Write>(&self, mut writer: W) -> Result<()> {
        let metadata = self.session().metadata()?;
        // ort rc.15 metadata accessors return Option (custom metadata may be
        // absent); orp's Result error type is `Box<dyn Error + Send + Sync>`,
        // so a plain message keeps the original behavior for present values.
        let missing = |what: &str| -> Box<dyn std::error::Error + Send + Sync> {
            format!("model metadata {what} is missing").into()
        };
        writeln!(writer, "NAME: {}", metadata.name().ok_or_else(|| missing("name"))?)?;
        writeln!(writer, "PRODUCER: {}", metadata.producer().ok_or_else(|| missing("producer"))?)?;
        writeln!(writer, "VERSION: {}", metadata.version().ok_or_else(|| missing("version"))?)?;
        writeln!(writer, "INPUTS:")?;
        for input in self.session().inputs() {
            writeln!(writer, "\t{}: {:?}", input.name(), input.dtype())?;
        }
        writeln!(writer, "OUTPUTS:")?;
        for input in self.session().outputs() {
            writeln!(writer, "\t{}: {:?}", input.name(), input.dtype())?;
        }
        Ok(())
    }

    /// Check model schema wrt. pipeline expectations
    fn check_schema<'a, P: Pipeline<'a>>(&'a self, pipeline: &P, params: &P::Parameters) -> Result<()> {
        if let Some(expected_inputs) = pipeline.expected_inputs(params) {
            // inputs should be exactly the same sets
            let expected_inputs = &expected_inputs.collect();
            let actual_inputs: HashSet<_> = self.session().inputs().iter().map(|i| i.name()).collect();
            if !actual_inputs.eq(expected_inputs) {
                return UnexpectedModelSchemaError::new_for_input(expected_inputs, &actual_inputs).into_err();
            }
        }
        if let Some(expected_outputs) = pipeline.expected_outputs(params) {            
            // for outputs, we just check that the expected ones are present (but having others is ok)
            let expected_outputs = &expected_outputs.collect();
            let actual_outputs: HashSet<_> = self.session().outputs().iter().map(|i| i.name()).collect();
            if !actual_outputs.is_superset(expected_outputs) {
                return UnexpectedModelSchemaError::new_for_output(expected_outputs, &actual_outputs).into_err();
            }
        }
        Ok(())
    }

    fn run(&self, input: SessionInputs<'_, '_>) -> Result<SessionOutputs<'_>> {
        // SAFETY: see the `session` field contract. The borrow ends when this
        // call returns; outputs are consumed by the caller's post-processor
        // before then.
        let session = unsafe { &mut *self.session.get() };
        Ok(session.run(input)?)
    }

}


/// References a model, a pipeline and some parameters to implement `Composable`
struct ComposableModel<'a, P: Pipeline<'a>> {
    model: &'a Model,
    pipeline: &'a P,
    params: &'a P::Parameters,
}


impl<'a, P: Pipeline<'a>> ComposableModel<'a, P> {
    pub fn new(model: &'a Model, pipeline: &'a P, params: &'a P::Parameters) -> Self {
        Self { model, pipeline, params }
    }
}


impl<'a, P: Pipeline<'a>> Composable<P::Input, P::Output> for ComposableModel<'a, P> {
    fn apply(&self, input: P::Input) -> Result<P::Output> {
        self.model.inference(input, self.pipeline, self.params)
    }
}
impl Model {
    fn session(&self) -> &Session {
        // SAFETY: shared reads only; no writer exists outside `run`, which
        // cannot overlap itself (ORT serializes kernels per session) or these
        // reads on the same thread.
        unsafe { &*self.session.get() }
    }
}

// SAFETY: `Model` shares its session across threads (GLiNER inference runs on
// a detector used concurrently by worker threads). The only interior mutation
// is `Session::run` for the duration of one call; ONNX Runtime documents
// sessions as thread-safe for concurrent Run, ort keeps
// `unsafe impl Send/Sync for Session`, and outputs never outlive the call.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}
